"""
Evaluation metrics from the paper.

- mCE  (mean Corruption Error): paper §4, uCE / AlexNet normalisation
- mFP  (mean Flip Probability): paper §4 perturbation robustness
- RMS Calibration Error: paper §4, Appendix E (Eq. 3)
- Brier Score: Appendix E
"""

import numpy as np


def mean_corruption_error(
    errors: dict[str, list[float]],
    alexnet_errors: dict[str, list[float]] | None = None,
) -> float:
    """
    Compute mCE over 15 corruptions × 5 severity levels.

    errors: {corruption_name: [err_sev1, ..., err_sev5]}  (as fractions, not %)
    alexnet_errors: same format; if None returns unnormalised mCE (uCE)
    """
    ce_values = []
    for corruption, errs in errors.items():
        uce = np.mean(errs)
        if alexnet_errors is not None:
            uce_alexnet = np.mean(alexnet_errors[corruption])
            ce_values.append(uce / uce_alexnet)
        else:
            ce_values.append(uce)
    return float(np.mean(ce_values))


def mean_flip_probability(flip_probs: dict[str, float]) -> float:
    """
    Average flip probability across perturbation types.

    flip_probs: {perturbation_name: flip_prob}
    """
    return float(np.mean(list(flip_probs.values())))


def rms_calibration_error(
    confidences: np.ndarray,
    correctness: np.ndarray,
    bin_size: int = 100,
) -> float:
    """
    RMS Calibration Error (Appendix E, Eq. 3).

    confidences: (N,) — predicted confidence (max softmax)
    correctness: (N,) — 1 if prediction correct, 0 otherwise
    bin_size: number of predictions per adaptive bin (paper uses 100)
    """
    n = len(confidences)
    # sort by confidence
    order = np.argsort(confidences)
    confidences = confidences[order]
    correctness = correctness[order]

    total = 0.0
    for i in range(0, n, bin_size):
        bin_conf = confidences[i:i + bin_size]
        bin_correct = correctness[i:i + bin_size]
        b = len(bin_conf)
        acc = bin_correct.mean()
        conf = bin_conf.mean()
        total += (b / n) * (acc - conf) ** 2

    return float(np.sqrt(total))


def brier_score(probs: np.ndarray, labels: np.ndarray, num_classes: int) -> float:
    """
    Brier score: mean squared error of probability vector vs one-hot label.

    probs:  (N, C) softmax probabilities
    labels: (N,)  integer class labels
    """
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(labels)), labels] = 1.0
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))
