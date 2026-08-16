import numpy as np
import random
from typing import List, Tuple


def create_nsp_examples(
    documents: List[List[str]], num_examples: int, seed: int = 42
) -> List[Tuple[str, str, int]]:
    """
    Generate (sentence_A, sentence_B, is_next_label) tuples.

    Label 1 means IsNext and label 0 means NotNext.
    """
    random.seed(seed)
    np.random.seed(seed)

    non_empty_documents = [document for document in documents if document]
    if not non_empty_documents and num_examples > 0:
        raise ValueError("documents must contain at least one sentence")

    examples = []
    for _ in range(num_examples):
        is_next = random.random() < 0.5

        if is_next:
            document = random.choice(non_empty_documents)
            if len(document) == 1:
                sentence_a = sentence_b = document[0]
            else:
                index = random.randrange(len(document) - 1)
                sentence_a = document[index]
                sentence_b = document[index + 1]
            label = 1
        else:
            document_a = random.choice(non_empty_documents)
            document_b = random.choice(non_empty_documents)
            sentence_a = random.choice(document_a)
            sentence_b = random.choice(document_b)
            label = 0

        examples.append((sentence_a, sentence_b, label))

    return examples


def create_nsp_pairs(
    documents: List[List[str]], pair_specs: List[dict]
) -> List[Tuple[str, str, int]]:
    """Build deterministic NSP pairs from index specifications."""

    def get(spec: dict, *names: str):
        for name in names:
            if name in spec:
                return spec[name]
        raise KeyError(f"pair specification is missing one of: {', '.join(names)}")

    pairs = []
    for spec in pair_specs:
        doc_a = get(spec, "doc_a", "doc_a_idx", "document_a", "document_a_idx")
        sent_a = get(spec, "sent_a", "sent_a_idx", "sentence_a", "sentence_a_idx")
        doc_b = get(spec, "doc_b", "doc_b_idx", "document_b", "document_b_idx")
        sent_b = get(spec, "sent_b", "sent_b_idx", "sentence_b", "sentence_b_idx")
        label = next(
            (
                spec[name]
                for name in ("label", "is_next", "is_next_label")
                if name in spec
            ),
            None,
        )
        if label is None:
            label = int(
                doc_a == doc_b
                and (
                    sent_b == sent_a + 1
                    or (len(documents[doc_a]) == 1 and sent_a == sent_b == 0)
                )
            )

        pairs.append(
            (documents[doc_a][sent_a], documents[doc_b][sent_b], int(label))
        )

    return pairs


class NSPHead:
    """Next Sentence Prediction classification head."""

    def __init__(self, hidden_size: int):
        self.W = np.random.randn(hidden_size, 2) * 0.02
        self.b = np.zeros(2)

    def forward(self, cls_hidden: np.ndarray) -> np.ndarray:
        """
        Predict IsNext logits: cls_hidden @ W + b
        """
        return cls_hidden @ self.W + self.b


def softmax(x: np.ndarray) -> np.ndarray:
    """Compute softmax along last axis."""
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

