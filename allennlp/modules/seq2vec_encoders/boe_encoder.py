from typing import Optional, Tuple

from overrides import overrides
import torch
from torch.nn import Conv1d, Linear

from allennlp.common import Params
from allennlp.modules.seq2vec_encoders.seq2vec_encoder import Seq2VecEncoder
from allennlp.nn import Activation


@Seq2VecEncoder.register("boe")
class BagOfEmbeddingsEncoder(Seq2VecEncoder):
    """
    A ``CnnEncoder`` is a combination of multiple convolution layers and max pooling layers.  As a
    :class:`Seq2VecEncoder`, the input to this module is of shape ``(batch_size, num_tokens,
    input_dim)``, and the output is of shape ``(batch_size, output_dim)``.

    The CNN has one convolution layer for each ngram filter size. Each convolution operation gives
    out a vector of size num_filters. The number of times a convolution layer will be used
    is ``num_tokens - ngram_size + 1``. The corresponding maxpooling layer aggregates all these
    outputs from the convolution layer and outputs the max.

    This operation is repeated for every ngram size passed, and consequently the dimensionality of
    the output after maxpooling is ``len(ngram_filter_sizes) * num_filters``.  This then gets
    (optionally) projected down to a lower dimensional output, specified by ``output_dim``.

    We then use a fully connected layer to project in back to the desired output_dim.  For more
    details, refer to "A Sensitivity Analysis of (and Practitioners’ Guide to) Convolutional Neural
    Networks for Sentence Classification", Zhang and Wallace 2016, particularly Figure 1.

    Parameters
    ----------
    embedding_dim : ``int``
        This is the input dimension to the encoder.  We need this because we can't do shape
        inference in pytorch, and we need to know what size filters to construct in the CNN.
    output_dim : ``Optional[int]``, optional (default=``None``)
        After doing convolutions and pooling, we'll project the collected features into a vector of
        this size.  If this value is ``None``, we will just return the result of the max pooling,
        giving an output of shape ``len(ngram_filter_sizes) * num_filters``.
    """
    def __init__(self,
                 embedding_dim: int,
                 comb_method: str,
                 activation: Activation = Activation.by_name('relu')(), 
                 output_dim: Optional[int] = None) -> None:
        super(BagOfEmbeddingsEncoder, self).__init__()
        self._embedding_dim = embedding_dim
        self._comb_method = comb_method
        self._activation = activation
        self._output_dim = output_dim

        # Make this check happen earlier
        assert(comb_method in ['sum', 'mean'])

        if self._output_dim:
            self.projection_layer = Linear(embedding_dim, self._output_dim)
            #self._activation(
        else:
            self.projection_layer = None
            self._output_dim = embedding_dim

    @overrides
    def get_input_dim(self) -> int:
        return self._embedding_dim

    @overrides
    def get_output_dim(self) -> int:
        return self._output_dim

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor):  # pylint: disable=arguments-differ
        if mask is not None:
            tokens = tokens * mask.unsqueeze(-1).float()

        if self.comb_method == 'sum':
            result = torch.sum(tokens, 1)
        elif self.comb_method == 'mean':
            result = torch.mean(tokens, 1)

        if self.projection_layer:
            result = self.projection_layer(result)
        return result

    @classmethod
    def from_params(cls, params: Params) -> 'BagOfEmbeddingsEncoder':
        embedding_dim = params.pop('embedding_dim')
        output_dim = params.pop('output_dim', None)
        comb_method = params.pop('comb_method', 'sum')
        activation = Activation.by_name(params.pop("activation", "relu"))()
        params.assert_empty(cls.__name__)
        return cls(embedding_dim=embedding_dim,
                   comb_method=comb_method,
                   output_dim=output_dim,
                   activation=activation)
