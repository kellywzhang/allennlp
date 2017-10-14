# coding: utf8
from __future__ import unicode_literals

import spacy
from spacy import language_data as base
from spacy.language_data import update_exc, strings_to_exc, expand_exc

from spacy.en.word_sets import NUM_WORDS
from spacy.en.tokenizer_exceptions import ORTH_ONLY

from allennlp.data.tokenizers.tokenizer_exceptions import TOKENIZER_EXCEPTIONS

TOKENIZER_EXCEPTIONS = dict(TOKENIZER_EXCEPTIONS)
update_exc(TOKENIZER_EXCEPTIONS, strings_to_exc(ORTH_ONLY))
update_exc(TOKENIZER_EXCEPTIONS, expand_exc(TOKENIZER_EXCEPTIONS, "'", "’"))
update_exc(TOKENIZER_EXCEPTIONS, strings_to_exc(base.EMOTICONS))
update_exc(TOKENIZER_EXCEPTIONS, strings_to_exc(base.ABBREVIATIONS))
