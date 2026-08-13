import numpy as np

from collections import Counter
import math


def bm25_score(query_tokens, docs, k1=1.2, b=0.75):
    """
    Returns numpy array of BM25 scores for each document.
    """
    number_of_docs = len(docs)
    if number_of_docs == 0:
        return np.array([], dtype=float)

    doc_term_counts = [Counter(doc) for doc in docs]
    doc_lengths = np.array([len(doc) for doc in docs], dtype=float)
    average_doc_length = doc_lengths.mean()
    scores = np.zeros(number_of_docs, dtype=float)

    # A repeated query term should not cause the same BM25 contribution to be
    # added more than once.
    for term in set(query_tokens):
        document_frequency = sum(term in counts for counts in doc_term_counts)
        if document_frequency == 0:
            continue

        idf = math.log(
            1
            + (number_of_docs - document_frequency + 0.5)
            / (document_frequency + 0.5)
        )

        for doc_idx, counts in enumerate(doc_term_counts):
            term_frequency = counts[term]
            if term_frequency == 0:
                continue

            if average_doc_length == 0:
                length_ratio = 0.0
            else:
                length_ratio = doc_lengths[doc_idx] / average_doc_length

            denominator = term_frequency + k1 * (1 - b + b * length_ratio)
            scores[doc_idx] += idf * (
                term_frequency * (k1 + 1) / denominator
            )

    return scores


# Input: query=["machine","learning"],
# docs=[["introduction","to","machine","learning"], ["deep","learning","basics"], ["cooking","pasta","guide"]]
#
# Output: [1.34111, 0.49005, 0.00000]
#
# scores[0] ≥ scores[1] ≫ scores[2] (