from collections import Counter
from itertools import chain


def word_count_dict(sentences):
    """
    Returns: dict[str, int] - global word frequency across all sentences
    """
    return dict(Counter(chain.from_iterable(sentences)))
