from typing import List, Dict


class WordPieceTokenizer:
    """
    WordPiece tokenizer for BERT.
    """

    def __init__(
        self, vocab: Dict[str, int], unk_token: str = "[UNK]", max_word_len: int = 100
    ):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_word_len = max_word_len

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into WordPiece tokens.
        """
        tokens = []
        for word in text.lower().split():
            word_tokens = self._tokenize_word(word)
            tokens.extend(word_tokens)
        return tokens

    def _greedy_match(self, start, word):
        """
        function to greadly match the longest substring in word
        """
        prefix = "" if start == 0 else "##"
        end = len(word)

        while end >= start and prefix + word[start:end] not in self.vocab:
            end -= 1

        return end >= start, end

    def _tokenize_word(self, word: str) -> List[str]:
        """
        Tokenize a single word into subwords.
        """
        # YOUR CODE HERE

        if len(word) > self.max_word_len:
            return [self.unk_token]
        result = []
        start = 0

        while start < len(word):
            found, end = self._greedy_match(start, word)

            if found:
                prefix = "" if start == 0 else "##"
                result.append(prefix + word[start:end])
                start = end
            else:
                result.append(self.unk_token)
                return result

        return result
