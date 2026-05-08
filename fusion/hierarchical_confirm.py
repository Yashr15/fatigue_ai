import time
from collections import deque
import numpy as np

class HierarchicalConfirm:
    def __init__(self,
                 short_window_sec=10,
                 long_window_sec=300,
                 long_thresh=55):
        self.short_window_sec = short_window_sec
        self.long_window_sec = long_window_sec
        self.long_thresh = long_thresh

        self.history = deque()
        self.confirmed = False

    def update(self, cfi, short_alert):
        now = time.time()

        # Keep long-term history
        if cfi is not None:
            self.history.append((now, cfi))

        # Remove old values
        while self.history and (now - self.history[0][0] > self.long_window_sec):
            self.history.popleft()

        # Long-term condition
        if len(self.history) < 10:
            return False  # insufficient data

        values = np.array([v for _, v in self.history])
        mean_cfi = values.mean()

        trend = np.polyfit(range(len(values)), values, 1)[0]

        self.confirmed = (
            short_alert and
            mean_cfi >= self.long_thresh and
            trend >= 0
        )

        return self.confirmed

    def get_long_stats(self):
        if not self.history:
            return None, None
        values = np.array([v for _, v in self.history])
        return values.mean(), values.max()
