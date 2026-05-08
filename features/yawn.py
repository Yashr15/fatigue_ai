"""
Yawn detection via Mouth Aspect Ratio (MAR).
"""

import numpy as np
import time

# Mouth landmarks (MediaPipe Face Mesh)
UPPER_LIP = [13]
LOWER_LIP = [14]
LEFT_MOUTH = [78]
RIGHT_MOUTH = [308]

# Defaults (overridable via config)
DEFAULT_MAR_THRESHOLD = 0.75
DEFAULT_MIN_DURATION = 1.0   # seconds
DEFAULT_WINDOW_SEC = 300     # 5 minutes


def compute_mar(landmarks, shape):
    """
    Mouth Aspect Ratio (MAR) = vertical / horizontal mouth opening.
    """
    h, w = shape

    def pt(i):
        return np.array([landmarks[i].x * w, landmarks[i].y * h])

    vertical = np.linalg.norm(pt(UPPER_LIP[0]) - pt(LOWER_LIP[0]))
    horizontal = np.linalg.norm(pt(LEFT_MOUTH[0]) - pt(RIGHT_MOUTH[0]))

    if horizontal == 0:
        return 0.0

    return vertical / horizontal


class YawnTracker:
    """
    Counts yawns within a rolling time window and produces a
    fatigue-oriented yawn score (0–100).

    Parameters
    ----------
    mar_thresh : float
        MAR above which the mouth is considered "open".
    min_duration : float
        Minimum seconds the mouth must stay open for a yawn.
    window_sec : int
        Rolling window (seconds) to count recent yawns.
    """

    def __init__(
        self,
        mar_thresh=DEFAULT_MAR_THRESHOLD,
        min_duration=DEFAULT_MIN_DURATION,
        window_sec=DEFAULT_WINDOW_SEC,
    ):
        self.mar_thresh = mar_thresh
        self.min_duration = min_duration
        self.window_sec = window_sec

        self.yawn_start = None
        self.yawn_times = []

    def update(self, mar):
        now = time.time()

        if mar > self.mar_thresh:
            if self.yawn_start is None:
                self.yawn_start = now
        else:
            if self.yawn_start is not None:
                duration = now - self.yawn_start
                if duration >= self.min_duration:
                    self.yawn_times.append(now)
                self.yawn_start = None

        # Keep only recent yawns
        self.yawn_times = [
            t for t in self.yawn_times if now - t <= self.window_sec
        ]

    def yawn_count(self):
        return len(self.yawn_times)

    def yawn_score(self):
        """Simple fatigue-oriented yawn score (0–100)."""
        return min(100.0, self.yawn_count() * 20.0)
