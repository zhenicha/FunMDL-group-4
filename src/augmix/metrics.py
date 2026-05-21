"""
Evaluation metrics from the paper.

- mCE  (mean Corruption Error): paper 4, Eq. uCE / AlexNet normalisation
- mFP  (mean Flip Probability): paper 4 perturbation robustness
- RMS Calibration Error: paper 4, Appendix E (Eq. 3)
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
    raise NotImplementedError


def mean_flip_probability(flip_probs: dict[str, float]) -> float:
    """
    Average flip probability across perturbation types.

    flip_probs: {perturbation_name: flip_prob}
    """
    raise NotImplementedError


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
    raise NotImplementedError


def brier_score(probs: np.ndarray, labels: np.ndarray, num_classes: int) -> float:
    """
    Brier score: mean squared error of probability vector vs one-hot label.

    probs:  (N, C) softmax probabilities
    labels: (N,)  integer class labels
    """
    raise NotImplementedError
