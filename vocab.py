
import nltk
import json
import os
from collections import Counter
nltk.download('punkt', quiet=True)

class Vocabulary:
    def __init__(self, freq_threshold=5, max_size=None):
        self.freq_threshold = freq_threshold
        self.max_size = max_size
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3:"<UNK>"}
        self.stoi = {v:k for k,v in self.itos.items()}
        self.freqs = Counter()

    def tokenizer(self, text):
        return nltk.tokenize.word_tokenize(text.lower())

    def build_vocabulary(self, sentence_list):
        for sentence in sentence_list:
            tokens = self.tokenizer(sentence)
            self.freqs.update(tokens)

        # most common sorted
        sorted_tokens = [token for token, cnt in self.freqs.items() if cnt >= self.freq_threshold]
        if self.max_size:
            sorted_tokens = sorted_tokens[: self.max_size - len(self.itos)]

        idx = max(self.itos.keys()) + 1
        for token in sorted_tokens:
            if token not in self.stoi:
                self.stoi[token] = idx
                self.itos[idx] = token
                idx += 1

    def numericalize(self, text):
        tokens = self.tokenizer(text)
        return [self.stoi.get(tok, self.stoi["<UNK>"]) for tok in tokens]
