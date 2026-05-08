"""
Sitting time tracker — estimates fatigue contribution from
prolonged sitting without posture breaks.
"""

import time

# Defaults (overridable via config)
DEFAULT_RESET_THRESHOLD = 20.0   # posture score below this → break detected
DEFAULT_MAX_MINUTES = 60.0       # minutes at which score saturates to 100


class SittingTimeTracker:
    """
    Parameters
    ----------
    reset_threshold : float
        If `posture_score` drops below this, the timer resets
        (assumes the user stood up or stretched).
    max_minutes : float
        Number of continuous sitting minutes that map to score 100.
    """

    def __init__(
        self,
        reset_threshold=DEFAULT_RESET_THRESHOLD,
        max_minutes=DEFAULT_MAX_MINUTES,
    ):
        self.reset_threshold = reset_threshold
        self.max_minutes = max_minutes
        self.start_time = time.time()

    def update(self, posture_score):
        """Reset timer if posture improves significantly (proxy for stand/stretch)."""
        if posture_score < self.reset_threshold:
            self.start_time = time.time()

    def ts_score(self):
        """Sitting Time Score (0–100)."""
        minutes = (time.time() - self.start_time) / 60
        return min(100.0, (minutes / self.max_minutes) * 100)
