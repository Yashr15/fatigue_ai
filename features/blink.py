"""
Blink detection and tracking using Eye Aspect Ratio (EAR).

Uses *both* eyes (averaged) for robustness against head tilt.
PERCLOS is computed as a true time-fraction, not a frame-count ratio.
"""

import numpy as np
import time
from collections import deque

# MediaPipe Face Mesh eye landmark indices
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE = [362, 385, 387, 263, 373, 380]

# Defaults (overridable via config)
DEFAULT_EAR_THRESHOLD = 0.21
DEFAULT_MICROSLEEP_THRESHOLD = 0.5  # seconds
DEFAULT_PERCLOS_WINDOW = 60.0       # seconds


def _euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def _eye_ear(landmarks, indices, image_shape):
    """Compute EAR for a single eye given its landmark indices."""
    h, w = image_shape
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices]
    p1, p2, p3, p4, p5, p6 = pts
    ear = (_euclidean(p2, p6) + _euclidean(p3, p5)) / (2.0 * _euclidean(p1, p4))
    return ear


def compute_ear(landmarks, image_shape):
    """
    Compute the average Eye Aspect Ratio across both eyes.

    Returns
    -------
    float
        Mean EAR of left and right eyes.
    """
    right = _eye_ear(landmarks, RIGHT_EYE, image_shape)
    left = _eye_ear(landmarks, LEFT_EYE, image_shape)
    return (right + left) / 2.0


class BlinkTracker:
    """
    Tracks blinks, blink scores, microsleeps, and PERCLOS.

    Parameters
    ----------
    ear_threshold : float
        EAR value below which the eyes are considered closed.
    microsleep_threshold : float
        Seconds of continuous eye closure that count as a microsleep.
    perclos_window_sec : float
        Rolling window (seconds) for the PERCLOS calculation.
    """

    def __init__(
        self,
        ear_threshold=DEFAULT_EAR_THRESHOLD,
        microsleep_threshold=DEFAULT_MICROSLEEP_THRESHOLD,
        perclos_window_sec=DEFAULT_PERCLOS_WINDOW,
    ):
        self.ear_threshold = ear_threshold
        self.microsleep_threshold = microsleep_threshold
        self.perclos_window_sec = perclos_window_sec

        self.eye_closed = False
        self.blink_start_time = None

        # Blink stats
        self.blink_count = 0
        self.blink_records = []

        # Latched values
        self.last_blink_score = 0.0
        self.last_blink_duration_ms = 0.0

        # PERCLOS buffer: stores (timestamp, is_closed)
        self.perclos_window = deque()
        self._last_perclos_time = None

    # ------------------------------------------------------------------ #
    @staticmethod
    def _compute_blink_score(duration_sec):
        D_norm = 0.2   # normal blink (s)
        D_max = 0.5    # drowsy blink (s)
        score = 100 * (duration_sec - D_norm) / (D_max - D_norm)
        return max(0.0, min(100.0, score))

    # ------------------------------------------------------------------ #
    def update(self, ear):
        now = time.time()
        microsleep = False
        blink_duration = None
        blink_score = None

        # Eye closes
        if ear < self.ear_threshold and not self.eye_closed:
            self.eye_closed = True
            self.blink_start_time = now

        # Eye opens → blink confirmed
        elif ear >= self.ear_threshold and self.eye_closed:
            self.eye_closed = False
            blink_duration = now - self.blink_start_time
            blink_score = self._compute_blink_score(blink_duration)

            self.blink_count += 1
            self.last_blink_score = blink_score
            self.last_blink_duration_ms = blink_duration * 1000

            self.blink_records.append({
                "timestamp": now,
                "duration_ms": self.last_blink_duration_ms,
                "blink_score": blink_score,
            })
            self.blink_start_time = None

        # Microsleep detection
        if self.eye_closed and self.blink_start_time:
            if (now - self.blink_start_time) >= self.microsleep_threshold:
                microsleep = True

        # Update PERCLOS window (time-weighted)
        self.perclos_window.append((now, self.eye_closed))
        while self.perclos_window and now - self.perclos_window[0][0] > self.perclos_window_sec:
            self.perclos_window.popleft()

        return blink_duration, blink_score, microsleep

    # ------------------------------------------------------------------ #
    def compute_perclos(self):
        """
        Percentage of time eyes were closed in the PERCLOS window.

        Uses actual elapsed time between samples rather than
        a simple frame count, so the result is accurate even if
        the frame rate fluctuates.
        """
        if len(self.perclos_window) < 2:
            return 0.0

        closed_time = 0.0
        items = list(self.perclos_window)

        for i in range(1, len(items)):
            dt = items[i][0] - items[i - 1][0]
            if items[i - 1][1]:          # previous sample was "eyes closed"
                closed_time += dt

        total_time = items[-1][0] - items[0][0]
        if total_time <= 0:
            return 0.0

        return closed_time / total_time

    # ------------------------------------------------------------------ #
    def average_blink_score_last_n(self, n=10):
        if not self.blink_records:
            return 0.0
        recent = self.blink_records[-n:]
        return sum(b["blink_score"] for b in recent) / len(recent)
