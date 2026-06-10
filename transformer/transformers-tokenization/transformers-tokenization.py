from typing import Dict, List

import numpy as np


class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """

    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 4

        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
        for i, v in enumerate(
            [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        ):
            self.word_to_id[v] = i
            self.id_to_word[i] = v

    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        unique_words = set()
        for line in texts:
            line = line.lower()
            for word in line.split():
                unique_words.add(word)

        i = 4
        for word in sorted(unique_words):
            self.word_to_id[word] = i
            self.id_to_word[i] = word
            i += 1
            self.vocab_size += 1

    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        # YOUR CODE HERE
        return [
            self.word_to_id.get(word.lower(), self.word_to_id[self.unk_token])
            for word in text.split()
        ]

    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        # YOUR CODE HERE
        return " ".join([self.id_to_word.get(id, self.unk_token) for id in ids])










