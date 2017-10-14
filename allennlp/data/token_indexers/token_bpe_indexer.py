from typing import Dict, List
import itertools

from overrides import overrides

from allennlp.common.checks import ConfigurationError
from allennlp.common.params import Params
from allennlp.common.util import pad_sequence_to_length
from allennlp.data.tokenizers.token import Token
from allennlp.data.token_indexers.token_indexer import TokenIndexer
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.tokenizers.bpe_tokenizer import BPETokenizer


@TokenIndexer.register("bpe")
class TokenBPEIndexer(TokenIndexer[List[int]]):
    """
    This :class:`TokenIndexer` represents tokens as lists of bpe indices.

    Parameters
    ----------
    namespace : ``str``, optional (default=``token_bpe``)
        We will use this namespace in the :class:`Vocabulary` to map the characters in each token
        to indices.
    bpe_tokenizer : ``BPETokenizer``, optional (default=``BPETokenizer()``)
        We use a :class:`CharacterTokenizer` to handle splitting tokens into characters, as it has
        options for byte encoding and other things.  The default here is to instantiate a
        ``CharacterTokenizer`` with its default parameters, which uses unicode characters and
        retains casing.
    """
    # pylint: disable=no-self-use
    def __init__(self,
                 namespace: str = 'token_bpe',
                 bpe_tokenizer: BPETokenizer = None) -> None:
        self._namespace = namespace
        self._bpe_tokenizer = bpe_tokenizer

    @overrides
    def count_vocab_items(self, token: Token, counter: Dict[str, Dict[str, int]]):
        if token.text is None:
            raise ConfigurationError('TokenBPEIndexer needs a tokenizer that retains text')
        for piece in self._bpe_tokenizer.tokenize(token.text):
            # If `text_id` is set on the character token (e.g., if we're using byte encoding), we
            # will not be using the vocab for this character.
            if getattr(piece, 'text_id', None) is None:
                counter[self._namespace][piece.text] += 1

    @overrides
    def token_to_indices(self, token: Token, vocabulary: Vocabulary) -> List[int]:
        indices = []
        if token.text is None:
            raise ConfigurationError('TokenBPEIndexer needs a tokenizer that retains text')
        for piece in self._bpe_tokenizer.tokenize(token.text):
            if getattr(piece, 'text_id', None) is not None:
                # `text_id` being set on the token means that we aren't using the vocab, we just
                # use this id instead.
                index = piece.text_id
            else:
                index = vocabulary.get_token_index(piece.text, self._namespace)
            indices.append(index)
        """
        print(indices)
        with open('/scratch/kz918/Data/GloVe/merge/yo2.txt', 'w', encoding='utf-8') as f:
            for t in tokens:
                f.write(t.text)
                f.write(" ")
            f.write("\n")
        import pdb; pdb.set_trace()
        """
        return indices

    @overrides
    def get_padding_lengths(self, token: List[int]) -> Dict[str, int]:
        return {'num_token_bpe': len(token)}

    @overrides
    def get_padding_token(self) -> List[int]:
        return []

    @overrides
    def pad_token_sequence(self,
                           tokens: List[List[int]],
                           desired_num_tokens: int,
                           padding_lengths: Dict[str, int]) -> List[List[int]]:
        padded_tokens = pad_sequence_to_length(tokens, desired_num_tokens, default_value=lambda: [])
        desired_token_length = padding_lengths['num_token_bpe']
        longest_token = max(tokens, key=len)
        padding_index = 0
        if desired_token_length > len(longest_token):
            # Since we want to pad to greater than the longest token, we add a
            # "dummy token" to get the speed of itertools.zip_longest.
            padded_tokens.append([padding_index] * desired_token_length)
        # pad the list of lists to the longest sublist, appending 0's
        padded_tokens = list(zip(*itertools.zip_longest(*padded_tokens, fillvalue=padding_index)))
        if desired_token_length > len(longest_token):
            # now we remove the "dummy token" if we appended one.
            padded_tokens.pop()

        # Now we need to truncate all of them to our desired length, and return the result.
        return [list(token[:desired_token_length]) for token in padded_tokens]

    @classmethod
    def from_params(cls, params: Params) -> 'TokenBPEIndexer':
        """
        Parameters
        ----------
        namespace : ``str``, optional (default=``token_bpes``)
            We will use this namespace in the :class:`Vocabulary` to map the characters in each token
            to indices.
        bpe_tokenizer : ``Params``, optional (default=``Params({})``)
            We use a :class:`CharacterTokenizer` to handle splitting tokens into characters, as it has
            options for byte encoding and other things.  These parameters get passed to the character
            tokenizer.  The default is to use unicode characters and to retain casing.
        """
        namespace = params.pop('namespace', 'token_bpe')
        bpe_tokenizer_params = params.pop('bpe_tokenizer', {})
        bpe_tokenizer = BPETokenizer.from_params(bpe_tokenizer_params)
        params.assert_empty(cls.__name__)
        return cls(namespace=namespace, bpe_tokenizer=bpe_tokenizer)
