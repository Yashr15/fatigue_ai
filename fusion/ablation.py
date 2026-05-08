"""
Ablation variants of the CFI for diagnostic / comparison purposes.

Feature vector layout (6 columns):
  [0] Blink Score       (BS)
  [1] PERCLOS
  [2] Posture Score     (PS)
  [3] Yawn Score        (YS)
  [4] Posture Change    (PCS)
  [5] Sitting Time      (TS)
"""

import numpy as np


def cfi_blink_only(window_tensor):
    """CFI using only the Blink Score channel (column 0)."""
    return float(np.mean(window_tensor[:, 0]))


def cfi_no_posture(window_tensor):
    """
    CFI using every signal *except* posture (column 2).

    Remaining channels are equally weighted.
    """
    cols = [0, 1, 3, 4, 5]
    values = [float(np.mean(window_tensor[:, c])) for c in cols]
    # Scale PERCLOS (index 1 in cols → values[1])
    values[1] *= 100
    return float(min(100.0, sum(values) / len(values)))
