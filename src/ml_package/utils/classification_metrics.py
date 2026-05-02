"""Classification metric helpers."""

import numpy as np


def _validate_average(average):
    if average not in {"binary", "macro", "weighted", "micro"}:
        raise ValueError(
            "average must be one of 'binary', 'macro', 'weighted', or 'micro'"
        )


def _labels_from_inputs(y_true, y_pred, labels):
    if labels is not None:
        return np.asarray(labels)
    return np.unique(np.concatenate([np.asarray(y_true), np.asarray(y_pred)]))


def confusion_matrix(y_true, y_pred, labels=None):
    """
    Compute a confusion matrix with rows as true labels and columns as predictions.

    For binary labels ``[0, 1]`` or ``[-1, 1]``, the returned matrix is arranged
    as ``[[TN, FP], [FN, TP]]`` because the labels are sorted in ascending order.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = _labels_from_inputs(y_true, y_pred, labels)
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=int)

    for true, pred in zip(y_true, y_pred):
        matrix[label_to_idx[true], label_to_idx[pred]] += 1

    return matrix


def accuracy(y_true, y_pred):
    """Compute the proportion of correctly classified samples."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean(y_true == y_pred)


def precision(y_true, y_pred, average="binary", labels=None, positive_label=1):
    """Compute precision with binary, macro, weighted, or micro averaging."""
    _validate_average(average)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    labels = _labels_from_inputs(y_true, y_pred, labels)

    if average == "binary":
        if positive_label not in labels:
            return 0.0
        idx = int(np.where(labels == positive_label)[0][0])
        tp = cm[idx, idx]
        fp = np.sum(cm[:, idx]) - tp
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0

    tp = np.diag(cm)
    predicted_positive = np.sum(cm, axis=0)
    per_class = np.divide(
        tp,
        predicted_positive,
        out=np.zeros_like(tp, dtype=float),
        where=predicted_positive > 0,
    )

    if average == "macro":
        return np.mean(per_class)
    if average == "weighted":
        weights = np.sum(cm, axis=1)
        return np.average(per_class, weights=weights) if np.sum(weights) > 0 else 0.0

    tp_total = np.sum(tp)
    fp_total = np.sum(predicted_positive - tp)
    return tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0


def recall(y_true, y_pred, average="binary", labels=None, positive_label=1):
    """Compute recall with binary, macro, weighted, or micro averaging."""
    _validate_average(average)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    labels = _labels_from_inputs(y_true, y_pred, labels)

    if average == "binary":
        if positive_label not in labels:
            return 0.0
        idx = int(np.where(labels == positive_label)[0][0])
        tp = cm[idx, idx]
        fn = np.sum(cm[idx, :]) - tp
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0

    tp = np.diag(cm)
    actual_positive = np.sum(cm, axis=1)
    per_class = np.divide(
        tp,
        actual_positive,
        out=np.zeros_like(tp, dtype=float),
        where=actual_positive > 0,
    )

    if average == "macro":
        return np.mean(per_class)
    if average == "weighted":
        return (
            np.average(per_class, weights=actual_positive)
            if np.sum(actual_positive) > 0 else 0.0
        )

    tp_total = np.sum(tp)
    fn_total = np.sum(actual_positive - tp)
    return tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0


def f1_score(y_true, y_pred, average="binary", labels=None, positive_label=1):
    """Compute F1 score with binary, macro, weighted, or micro averaging."""
    _validate_average(average)

    if average in {"binary", "micro"}:
        p = precision(y_true, y_pred, average=average, labels=labels,
                      positive_label=positive_label)
        r = recall(y_true, y_pred, average=average, labels=labels,
                   positive_label=positive_label)
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    tp = np.diag(cm)
    predicted_positive = np.sum(cm, axis=0)
    actual_positive = np.sum(cm, axis=1)

    p = np.divide(
        tp,
        predicted_positive,
        out=np.zeros_like(tp, dtype=float),
        where=predicted_positive > 0,
    )
    r = np.divide(
        tp,
        actual_positive,
        out=np.zeros_like(tp, dtype=float),
        where=actual_positive > 0,
    )
    per_class = np.divide(
        2 * p * r,
        p + r,
        out=np.zeros_like(p, dtype=float),
        where=(p + r) > 0,
    )

    if average == "macro":
        return np.mean(per_class)

    weights = actual_positive
    return np.average(per_class, weights=weights) if np.sum(weights) > 0 else 0.0
