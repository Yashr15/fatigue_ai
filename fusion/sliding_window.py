"""
Fixed-size sliding window for temporal feature aggregation.

Each frame appends a 6-element feature vector:
  [BS, PERCLOS, PS, YS, PCS, TS]

When the window is full, ``get_tensor()`` returns a numpy
array of shape ``(W, 6)`` ready for fusion.
"""

from collections import deque
import numpy as np


class SlidingWindow:
    def __init__(self, window_size=30):
        self.window_size = window_size
        self.buffer = deque(maxlen=window_size)

    def add(self, feature_vector):
        """
        Append one time-step.

        Parameters
        ----------
        feature_vector : list or array
            Length-6 vector ``[BS, PERCLOS, PS, YS, PCS, TS]``.
        """
        self.buffer.append(feature_vector)

    def is_full(self):
        return len(self.buffer) == self.window_size

    def get_tensor(self):
        """
        Returns
        -------
        numpy.ndarray
            Shape ``(W, 6)`` — the current window contents.
        """
        return np.array(self.buffer)
