import numpy as np


def evaluate(y_true, y_score, threshold):
    """
    Evaluate predictions at a given threshold.

    Positive prediction if score >= threshold.
    Returns:
        (TPR, FPR)
    """
    mask_true = y_true == 1
    mask_pred = y_score >= threshold

    tp = np.sum(mask_true & mask_pred)
    tn = np.sum(~mask_true & ~mask_pred)
    fp = np.sum(~mask_true & mask_pred)
    fn = np.sum(mask_true & ~mask_pred)

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return tpr, fpr


def roc_curve(y_true, y_score):
    """
    Compute ROC curve from binary labels and scores.

    Returns:
        fpr, tpr, thresholds
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    # Start with threshold = +inf (everything predicted negative)
    thresholds = [np.inf]
    tpr = [0.0]
    fpr = [0.0]

    # Sweep threshold from largest score to smallest
    for threshold in np.sort(np.unique(y_score))[::-1]:
        t, f = evaluate(y_true, y_score, threshold)
        thresholds.append(float(threshold))
        tpr.append(float(t))
        fpr.append(float(f))

    return fpr, tpr, thresholds




