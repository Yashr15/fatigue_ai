"""
CSV logger for blink events.

Keeps the file handle open and flushes periodically
to avoid per-blink open/close overhead.
"""

import csv
import os
import time
import logging

logger = logging.getLogger(__name__)


class BlinkCSVLogger:
    """
    Parameters
    ----------
    save_dir : str
        Directory to store CSV files (created if missing).
    filename : str or None
        Explicit filename; auto-generated with a timestamp if None.
    """

    def __init__(self, save_dir="data", filename=None):
        os.makedirs(save_dir, exist_ok=True)

        if filename is None:
            filename = f"blink_log_{int(time.time())}.csv"

        self.filepath = os.path.join(save_dir, filename)

        # Open file and write header — kept open for the session
        self._file = open(self.filepath, mode="w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow([
            "timestamp",
            "blink_count",
            "duration_ms",
            "blink_score",
        ])
        self._file.flush()
        logger.info("Blink logger → %s", self.filepath)

    def log(self, blink_count, duration_ms, blink_score):
        """Append one blink event row."""
        self._writer.writerow([
            time.time(),
            blink_count,
            round(duration_ms, 2),
            round(blink_score, 2),
        ])
        self._file.flush()

    def close(self):
        """Flush and close the underlying file."""
        if self._file and not self._file.closed:
            self._file.flush()
            self._file.close()

    def __del__(self):
        self.close()
