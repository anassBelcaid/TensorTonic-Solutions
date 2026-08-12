def text_chunking(tokens, chunk_size, overlap):
    """
    Split tokens into chunks with optional overlap.

    The final chunk may be shorter than ``chunk_size``.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    step = chunk_size - overlap
    result = []

    for start in range(0, len(tokens), step):
        result.append(tokens[start : start + chunk_size])

        # Once a chunk reaches the end, another chunk would only repeat tokens
        # already included in this one.
        if start + chunk_size >= len(tokens):
            break

    return result


