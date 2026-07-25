import numpy as np


def f1_score(p, r):
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def classification_metrics(y_true, y_pred, average="micro", pos_label=1):
    """
    Compute accuracy, precision, recall and F1 score.

    Parameters
    ----------
    average : {"micro", "macro", "weighted", "binary"}
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Map labels to consecutive indices
    labels = np.unique(np.concatenate([y_true, y_pred]))
    K = len(labels)
    N = len(y_true)

    label_to_idx = {label: i for i, label in enumerate(labels)}

    true_idx = np.vectorize(label_to_idx.get)(y_true)
    pred_idx = np.vectorize(label_to_idx.get)(y_pred)

    # One-hot encodings
    one_true = np.zeros((N, K), dtype=int)
    one_pred = np.zeros((N, K), dtype=int)

    one_true[np.arange(N), true_idx] = 1
    one_pred[np.arange(N), pred_idx] = 1

    # Confusion matrix
    confusion = one_true.T @ one_pred

    # Accuracy
    accuracy = np.trace(confusion) / N

    # Per-class statistics
    tp = np.diag(confusion)
    fn = confusion.sum(axis=1) - tp
    fp = confusion.sum(axis=0) - tp

    precision = np.divide(
        tp,
        tp + fp,
        out=np.zeros(K, dtype=float),
        where=(tp + fp) != 0,
    )

    recall = np.divide(
        tp,
        tp + fn,
        out=np.zeros(K, dtype=float),
        where=(tp + fn) != 0,
    )

    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(K, dtype=float),
        where=(precision + recall) != 0,
    )

    result = {"accuracy": float(accuracy)}

    if average == "micro":
        tp_sum = tp.sum()
        fp_sum = fp.sum()
        fn_sum = fn.sum()

        p = tp_sum / (tp_sum + fp_sum) if tp_sum + fp_sum else 0.0
        r = tp_sum / (tp_sum + fn_sum) if tp_sum + fn_sum else 0.0

        result["precision"] = float(p)
        result["recall"] = float(r)
        result["f1"] = float(f1_score(p, r))

    elif average == "macro":
        result["precision"] = float(precision.mean())
        result["recall"] = float(recall.mean())
        result["f1"] = float(f1.mean())

    elif average == "weighted":
        weights = confusion.sum(axis=1) / N

        result["precision"] = float(np.sum(weights * precision))
        result["recall"] = float(np.sum(weights * recall))
        result["f1"] = float(np.sum(weights * f1))

    elif average == "binary":
        if pos_label not in label_to_idx:
            raise ValueError(f"pos_label={pos_label} not found.")

        i = label_to_idx[pos_label]

        result["precision"] = float(precision[i])
        result["recall"] = float(recall[i])
        result["f1"] = float(f1[i])

    else:
        raise ValueError(
            "average must be one of {'micro', 'macro', 'weighted', 'binary'}"
        )

    return result

