from typing import List

from overrides import overrides

from allennlp.common import Params
from allennlp.data.tokenizers.token import Token
from allennlp.data.tokenizers.tokenizer import Tokenizer
from allennlp.data.tokenizers.word_splitter import WordSplitter, SpacyWordSplitter


@Tokenizer.register("bpe")
class BPETokenizer(Tokenizer):
    """
    A ``BPETokenizer`` splits strings into bpe tokens.

    Parameters
    ----------
    lowercase_tokens : ``bool``, optional (default=``False``)
        If ``True``, we will lowercase all of the tokens in the text before doing any other
        operation.
    start_tokens : ``List[str]``, optional
        If given, these tokens will be added to the beginning of every string we tokenize.  If
        using byte encoding, this should actually be a ``List[int]``, not a ``List[str]``.
    end_tokens : ``List[str]``, optional
        If given, these tokens will be added to the end of every string we tokenize.  If using byte
        encoding, this should actually be a ``List[int]``, not a ``List[str]``.
    """
    def __init__(self,
                 merges: str = None,
                 bpe_vocab: str = None,
                 word_split: bool = False,
                 word_splitter: WordSplitter = None,
                 lowercase_tokens: bool = False,
                 start_tokens: List[str] = None,
                 end_tokens: List[str] = None) -> None:
        self._lowercase_tokens = lowercase_tokens
        self._word_split = word_split
        if word_split:
            self._word_splitter = word_splitter or SpacyWordSplitter()
        self._start_tokens = start_tokens or []
        # We reverse the tokens here because we're going to insert them with `insert(0)` later;
        # this makes sure they show up in the right order.
        self._start_tokens.reverse()
        self._end_tokens = end_tokens or []
        
        # Merges
        with open(merges, 'r', encoding="utf-8") as f:
            lines = f.readlines()
        self.merges = [l.strip() for l in lines]

        # Vocabulary
        self.bpe_vocab = {}
        with open(bpe_vocab, 'r', encoding="utf-8") as f:
            lines = f.readlines()
        words = [l.strip().split('\t')[0] for l in lines]
        for w in words:
            self.bpe_vocab[w] = w

    @overrides
    def tokenize(self, text: str) -> List[Token]:
        if self._lowercase_tokens:
            text = text.lower()
        if self._word_split:
            words = self._word_splitter.split_words(text)
            tokens = [self.tokenize_word(w.text) for w in words]
            all_tokens = []
            for t in tokens:
                all_tokens += t
            return all_tokens
        else:
            return self.tokenize_word(text)

    def tokenize_word(self, text: str) -> List[Token]:
        text = "_"+text
 
        # Check if word in
        if text in self.bpe_vocab:
            raw_tokens = self.bpe_vocab[text]
        else:
            raw_tokens = self.merge_word(text)
        
        bpe_tokens = raw_tokens.split(" ")
        self.bpe_vocab[text] = raw_tokens

        # Make token objects
        tokens = [Token(t) for t in bpe_tokens]
        """
        START TOKEN NOT SUPPORTED
        """
        return tokens


    def merge_word(self, word):
        """Replace all occurrences of a symbol pair ('A', 'B') with a new symbol 'AB'"""
        chars = " ".join(list(word))
        for merge in self.merges:
            if merge in chars:
                chars = chars.replace(merge, "".join(merge.split(" ")))
        return chars

    @classmethod
    def from_params(cls, params: Params) -> 'BPETokenizer':
        merges = params.pop('merges', None)
        bpe_vocab = params.pop('bpe_vocab', None)
        word_split = params.pop('word_split', False)
        word_splitter = WordSplitter.from_params(params.pop('word_splitter', {}))
        lowercase_tokens = params.pop('lowercase_tokens', False)
        start_tokens = params.pop('start_tokens', None)
        end_tokens = params.pop('end_tokens', None)
        params.assert_empty(cls.__name__)
        return cls(merges=merges,
                   bpe_vocab=bpe_vocab,
                   word_split=word_split,
                   word_splitter=word_splitter,
                   lowercase_tokens=lowercase_tokens,
                   start_tokens=start_tokens,
                   end_tokens=end_tokens)
