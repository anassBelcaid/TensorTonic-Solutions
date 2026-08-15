import numpy as np


class BertEmbeddings:
    """
    BERT Embeddings = Token + Position + Segment
    """

    def __init__(self, vocab_size: int, max_position: int, hidden_size: int):
        self.hidden_size = hidden_size

        # Token embeddings
        self.token_embeddings = np.random.randn(vocab_size, hidden_size) * 0.02

        # Position embeddings (learned, not sinusoidal)
        self.position_embeddings = np.random.randn(max_position, hidden_size) * 0.02

        # Segment embeddings (just 2 segments: A and B)
        self.segment_embeddings = np.random.randn(2, hidden_size) * 0.02

    def forward(self, token_ids: np.ndarray, segment_ids: np.ndarray) -> np.ndarray:
        """
        Returns: np.ndarray of shape (batch, seq_len, hidden_size) with combined embeddings
        """
        # embedding
        token_embeds = self.token_embeddings[token_ids]

        # positional
        seq_len = token_ids.shape[1]
        pos_embeds = self.position_embeddings[np.arange(seq_len)]

        # segement ids
        segment_embeds = self.segment_embeddings[segment_ids]

        return token_embeds + pos_embeds + segment_embeds
