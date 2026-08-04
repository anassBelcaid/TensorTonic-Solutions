import numpy as np
from collections import Counter
import math


def get_vocabulary(documents):
    """
    function to get the vocabulary words in sorted order
    """

    words = set()

    for document in documents:
        for word in document.split():
            words.add(word)
    return list(sorted(words))


def tfidf_vectorizer(documents):
    """
    Build TF-IDF matrix from a list of text documents.
    Returns tuple of (tfidf_matrix, vocabulary).
    """
    # Write code here
    vocabulary = get_vocabulary(documents)

    # Build the document_to_idx
    word_to_idx = {word: i for (i, word) in enumerate(vocabulary)}
    N = len(vocabulary)
    D = len(documents)
    tf = np.zeros((D, N))

    for j, document in enumerate(documents):
        counter = Counter(document.split())
        for word, c in counter.items():
            idx = word_to_idx[word]
            tf[j, idx] = c
    # normalizing
    tf = tf / tf.sum(axis=1, keepdims=True)

    # nor the intersting part idf
    idf = (tf > 0).sum(axis=0, keepdims=True)
    idf = np.log(D / idf)
    return tf * idf, vocabulary


