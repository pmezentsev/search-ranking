import nltk
import string
import typing as t
import json


class Tokenizer:
    def __init__(self, vocab_path, max_len=30):
        with open(vocab_path, 'r') as f:
            self.vocab = json.load(f)
        self.oov_val = self.vocab['OOV']
        self.max_len = max_len
        self.stopwords = {'the', 'what', 'is', 'how', 'i', 'to', 'a', 'in', 'do', 'of', 'are',
                          'and', 'can', 'for', 'you', 'why', 'it', 'my', 'on', 'does',
                          's', 'which', 'be', 'if', 'some', 'or', 'get', 'that', 'with', 'should',
                          'your', 'have', 'an', 'from', 'will', 'who', 'when',
                          'like', 'would', 'there', 'at', 't', 'as',
                          'about', 'not', 'one', 'most', 'we', 'make', 'way', 'did', 'where',
                          'was', 'any', 'by', 'so',
                          'learn', 'they', 'me', 'has', 'someone',
                          'this', 'ever', 'new', 'much', 'all', 'use', 'more',
                          'their', 'many', 'than',
                          'other', 'am', 'out', 'mean', 'first'}

    def tokenize(self, text: str) -> t.List[str]:
        text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
        text = text.lower()
        tokens = nltk.word_tokenize(text)
        return tokens

    def token_to_ids(self, tokens: t.List[str]) -> t.List[int]:
        token_ids = [self.vocab.get(token, self.oov_val) for token in tokens[:self.max_len]]
        pad_len = self.max_len - len(token_ids)
        if pad_len > 0:
            token_ids = token_ids + [0] * pad_len
        return token_ids

    def text_to_ids(self, text: str) -> t.List[int]:
        tokens = self.tokenize(text)
        token_ids = self.token_to_ids(tokens)
        return token_ids

    def remove_stopwords(self, tokens):
        return [t for t in tokens if t not in self.stopwords]

