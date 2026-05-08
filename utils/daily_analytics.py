"""
Cross-session daily summary report.

Aggregates blink and fatigue data across all session CSVs
in the data directory.
"""

import os
import logging
import pandas as pd

logger = logging.getLogger(__name__)


def generate_daily_report(data_root="data", report_dir="reports"):
    os.makedirs(report_dir, exist_ok=True)

    blink_logs = []
    fatigue_logs = []

    if not os.path.isdir(data_root):
        logger.warning("Data directory '%s' not found — skipping daily report.", data_root)
        return

    for f in os.listdir(data_root):
        path = os.path.join(data_root, f)
        if f.startswith("blink_log") and f.endswith(".csv"):
            try:
                df = pd.read_csv(path)
                if not df.empty:
                    blink_logs.append(df)
            except Exception as exc:
                logger.warning("Could not read %s: %s", path, exc)
        if f.startswith("fatigue_episodes") and f.endswith(".csv"):
            try:
                df = pd.read_csv(path)
                if not df.empty:
                    fatigue_logs.append(df)
            except Exception as exc:
                logger.warning("Could not read %s: %s", path, exc)

    if not blink_logs:
        logger.info("No blink session data found — skipping daily report.")
        return

    blink_df = pd.concat(blink_logs, ignore_index=True)

    summary = {
        "Total Sessions": len(blink_logs),
        "Total Blinks": len(blink_df),
        "Mean Blink Score": blink_df["blink_score"].mean(),
        "Max Blink Score": blink_df["blink_score"].max(),
        "Total Fatigue Episodes": sum(len(df) for df in fatigue_logs),
    }

    pd.DataFrame([summary]).to_csv(
        os.path.join(report_dir, "daily_summary.csv"), index=False
    )

    logger.info("Daily productivity report saved.")
