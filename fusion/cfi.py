"""
Combined Fatigue Index (CFI) — fuses all six feature signals
from the sliding window into a single 0–100 score.

Feature vector layout (6 columns):
  [0] Blink Score       (BS)
  [1] PERCLOS
  [2] Posture Score     (PS)
  [3] Yawn Score        (YS)
  [4] Posture Change    (PCS)
  [5] Sitting Time      (TS)
"""

import numpy as np

# Default weights (overridable via config) — must sum to 1.0
DEFAULT_W_BLINK = 0.25
DEFAULT_W_PERCLOS = 0.25
DEFAULT_W_POSTURE = 0.15
DEFAULT_W_YAWN = 0.15
DEFAULT_W_PCS = 0.10
DEFAULT_W_SITTING = 0.10


def compute_cfi(
    window_tensor,
    w_blink=DEFAULT_W_BLINK,
    w_perclos=DEFAULT_W_PERCLOS,
    w_posture=DEFAULT_W_POSTURE,
    w_yawn=DEFAULT_W_YAWN,
    w_pcs=DEFAULT_W_PCS,
    w_sitting=DEFAULT_W_SITTING,
):
    """
    Compute the Combined Fatigue Index from a sliding-window tensor.

    Parameters
    ----------
    window_tensor : numpy.ndarray
        Shape ``(W, 6)`` — one row per time step, six feature columns.
    w_* : float
        Per-feature weights.  Must sum to 1.0 for a true [0, 100] range.

    Returns
    -------
    float
        CFI value clipped to [0, 100].
    """
    # Temporal aggregation (column means)
    mean_bs = np.mean(window_tensor[:, 0])
    mean_pc = np.mean(window_tensor[:, 1]) * 100   # scale PERCLOS (0–1) → 0–100
    mean_ps = np.mean(window_tensor[:, 2])
    mean_ys = np.mean(window_tensor[:, 3])
    mean_pcs = np.mean(window_tensor[:, 4])
    mean_ts = np.mean(window_tensor[:, 5])

    cfi = (
        w_blink * mean_bs +
        w_perclos * mean_pc +
        w_posture * mean_ps +
        w_yawn * mean_ys +
        w_pcs * mean_pcs +
        w_sitting * mean_ts
    )

    return float(min(100.0, cfi))
