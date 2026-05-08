"""
Fatigue episode logger.

Each session writes to a unique file so that daily analytics
can aggregate across all sessions without data loss.
"""

import csv
import time
import os
import logging

logger = logging.getLogger(__name__)


class FatigueEpisodeLogger:
    """
    Logs confirmed fatigue episodes (start / end / duration) to CSV.

    Each instance creates a timestamped file so previous sessions
    are never overwritten.

    Parameters
    ----------
    save_dir : str
        Directory for CSV files (created if missing).
    """

    def __init__(self, save_dir="data"):
        os.makedirs(save_dir, exist_ok=True)

        filename = f"fatigue_episodes_{int(time.time())}.csv"
        self.filepath = os.path.join(save_dir, filename)

        self.active = False
        self.start_time = None

        self._file = open(self.filepath, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(["start_time", "end_time", "duration_sec"])
        self._file.flush()
        logger.info("Fatigue logger → %s", self.filepath)

    def update(self, fatigue_confirmed):
        now = time.time()

        if fatigue_confirmed and not self.active:
            self.active = True
            self.start_time = now

        elif not fatigue_confirmed and self.active:
            duration = now - self.start_time
            self._writer.writerow([self.start_time, now, round(duration, 2)])
            self._file.flush()

            self.active = False
            self.start_time = None

    def close(self):
        """Flush and close the underlying file."""
        if self._file and not self._file.closed:
            self._file.flush()
            self._file.close()

    def __del__(self):
        self.close()
