"""
Posture Change Score (PCS) — tracks significant changes in
neck angle over a rolling time window.
"""

import numpy as np
import time

# MediaPipe landmarks
CHIN = 152
LEFT_SHOULDER = 234
RIGHT_SHOULDER = 454

# Defaults (overridable via config)
DEFAULT_ANGLE_DELTA = 10.0   # degrees
DEFAULT_WINDOW_SEC = 300     # 5 minutes


class PostureChangeTracker:
    """
    Tracks abrupt posture changes (neck angle jumps) and produces
    a PCS score (0–100).

    Parameters
    ----------
    angle_delta : float
        Minimum angle change (degrees) to register as a posture shift.
    window_sec : int
        Rolling window (seconds) to count recent changes.
    """

    def __init__(
        self,
        angle_delta=DEFAULT_ANGLE_DELTA,
        window_sec=DEFAULT_WINDOW_SEC,
    ):
        self.angle_delta = angle_delta
        self.window_sec = window_sec

        self.last_angle = None
        self.change_times = []

    def _compute_neck_angle(self, landmarks, shape):
        h, w = shape

        def pt(i):
            return np.array([landmarks[i].x * w, landmarks[i].y * h])

        chin = pt(CHIN)
        shoulder_mid = (pt(LEFT_SHOULDER) + pt(RIGHT_SHOULDER)) / 2

        vec = chin - shoulder_mid
        vertical = np.array([0, 1])

        norm = np.linalg.norm(vec)
        if norm == 0:
            return 0.0

        cos_angle = np.dot(vec / norm, vertical)
        angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
        return angle

    def update(self, landmarks, shape):
        """
        Process a new frame.

        Parameters
        ----------
        landmarks : list
            MediaPipe face mesh landmarks.
        shape : tuple[int, int]
            (height, width) of the frame.
        """
        now = time.time()
        theta = self._compute_neck_angle(landmarks, shape)

        if self.last_angle is not None:
            if abs(theta - self.last_angle) >= self.angle_delta:
                self.change_times.append(now)

        self.last_angle = theta

        # Keep only recent changes
        self.change_times = [
            t for t in self.change_times if now - t <= self.window_sec
        ]

    def pcs_score(self):
        """Posture Change Score (0–100)."""
        return min(100.0, len(self.change_times) * 10.0)
