from typing import Dict, Optional

import torch
from torch import nn 

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward, MatrixAttention
from allennlp.modules import Seq2SeqEncoder, TimeDistributed, TextFieldEmbedder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask, last_dim_softmax, weighted_sum
from allennlp.training.metrics import CategoricalAccuracy


@Model.register("nli_encoder")
class NLIEncoder(Model):
    """
    This ``Model`` implements the Decomposable Attention model described in `"A Decomposable
    Attention Model for Natural Language Inference"
    <https://www.semanticscholar.org/paper/A-Decomposable-Attention-Model-for-Natural-Languag-Parikh-T%C3%A4ckstr%C3%B6m/07a9478e87a8304fc3267fa16e83e9f3bbd98b27>`_
    by Parikh et al., 2016, with some optional enhancements before the decomposable attention
    actually happens.  Parikh's original model allowed for computing an "intra-sentence" attention
    before doing the decomposable entailment step.  We generalize this to any
    :class:`Seq2SeqEncoder` that can be applied to the premise and/or the hypothesis before
    computing entailment.

    The basic outline of this model is to get an embedded representation of each word in the
    premise and hypothesis, align words between the two, compare the aligned phrases, and make a
    final entailment decision based on this aggregated comparison.  Each step in this process uses
    a feedforward network to modify the representation.

    Parameters
    ----------
    vocab : ``Vocabulary``
    text_field_embedder : ``TextFieldEmbedder``
        Used to embed the ``premise`` and ``hypothesis`` ``TextFields`` we get as input to the
        model.
    aggregate_feedforward : ``FeedForward``
        This final feedforward network is applied to the concatenated, summed result of the
        ``compare_feedforward`` network, and its output is used as the entailment class logits.
    encoder : ``Seq2SeqEncoder``, optional (default=``None``)
        Encoder of premise and hypothesis encoders.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 fc_dim: int,
                 nonlinear_fc: bool,
                 dropout_fc: float=0.0,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(NLIEncoder, self).__init__(vocab, regularizer)

        self._text_field_embedder = text_field_embedder
        self._encoder = encoder
        self._fc_dim = fc_dim
        self._nonlinear_fc = nonlinear_fc
        self._dropout_fc = dropout_fc
        if nonlinear_fc:
            self._classifier = nn.Sequential(
                    nn.Dropout(p=dropout_fc),
                    nn.Linear(4*encoder.get_output_dim(), fc_dim),
                    nn.Tanh(),
                    nn.Dropout(p=dropout_fc),
                    nn.Linear(fc_dim, fc_dim),
                    nn.Tanh(),
                    nn.Dropout(p=dropout_fc),
                    nn.Linear(fc_dim, 3),
                )
        else:
            self._classifier = nn.Sequential(
                    nn.Linear(4*encoder.get_output_dim(), fc_dim),
                    nn.Linear(fc_dim, fc_dim),
                    nn.Linear(fc_dim, 3)
                )

        self._num_labels = vocab.get_vocab_size(namespace="labels")
        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()
        initializer(self)

    def forward(self,  # type: ignore
                premise: Dict[str, torch.LongTensor],
                hypothesis: Dict[str, torch.LongTensor],
                label: torch.IntTensor = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        premise : Dict[str, torch.LongTensor]
            From a ``TextField``
        hypothesis : Dict[str, torch.LongTensor]
            From a ``TextField``
        label : torch.IntTensor, optional (default = None)
            From a ``LabelField``

        Returns
        -------
        An output dictionary consisting of:

        label_logits : torch.FloatTensor
            A tensor of shape ``(batch_size, num_labels)`` representing unnormalised log
            probabilities of the entailment label.
        label_probs : torch.FloatTensor
            A tensor of shape ``(batch_size, num_labels)`` representing probabilities of the
            entailment label.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """
        embedded_premise = self._text_field_embedder(premise)
        embedded_hypothesis = self._text_field_embedder(hypothesis)

        for k, v in premise.items():
            dims = len(v.size())
            break
        if dims > 2:
            premise_mask = get_text_field_mask({k: v[:,:,0] for k, v in premise.items()}).float()
            hypothesis_mask = get_text_field_mask({k: v[:,:,0] for k, v in hypothesis.items()}).float()
        else:
            premise_mask = get_text_field_mask(premise).float()
            hypothesis_mask = get_text_field_mask(hypothesis).float()

        premise_hidden = self._encoder(embedded_premise, premise_mask)
        hypothesis_hidden = self._encoder(embedded_hypothesis, hypothesis_mask)

        premise = torch.max(premise_hidden, 1)[0]
        hypothesis = torch.max(hypothesis_hidden, 1)[0]
        #premise = torch.sum(premise_hidden, 1)
        #hypothesis = torch.sum(hypothesis_hidden, 1)

        features = torch.cat((premise, hypothesis, torch.abs(premise-hypothesis), premise*hypothesis), 1)
        label_logits = self._classifier(features)        
        label_probs = torch.nn.functional.softmax(label_logits)

        output_dict = {"label_logits": label_logits, "label_probs": label_probs}

        if label is not None:
            loss = self._loss(label_logits, label.long().view(-1))
            self._accuracy(label_logits, label.squeeze(-1))
            output_dict["loss"] = loss

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
                'accuracy': self._accuracy.get_metric(reset),
                }

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'NLIEncoder':
        embedder_params = params.pop("text_field_embedder")
        text_field_embedder = TextFieldEmbedder.from_params(vocab, embedder_params)

        encoder_params = params.pop("encoder", None)
        if encoder_params is not None:
            encoder = Seq2SeqEncoder.from_params(encoder_params)
        else:
            encoder = None

        fc_dim = params.pop('fc_dim', 512)
        nonlinear_fc = params.pop('nonlinear_fc', True)
        dropout_fc = params.pop('dropout_fc', 0.0)

        init_params = params.pop('initializer', None)
        reg_params = params.pop('regularizer', None)
        initializer = (InitializerApplicator.from_params(init_params)
                       if init_params is not None
                       else InitializerApplicator())
        regularizer = RegularizerApplicator.from_params(reg_params) if reg_params is not None else None

        return cls(vocab=vocab,
                   text_field_embedder=text_field_embedder,
                   encoder=encoder,
                   fc_dim=fc_dim,
                   nonlinear_fc=nonlinear_fc,
                   dropout_fc=dropout_fc,
                   initializer=initializer,
                   regularizer=regularizer)
