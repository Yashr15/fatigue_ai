"""
Post-session analytics — generates summary metrics, plots, and text reports.
"""

import os
import logging
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for headless use
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def generate_session_report(data_dir="data", report_dir="reports"):
    os.makedirs(report_dir, exist_ok=True)

    if not os.path.isdir(data_dir):
        logger.warning("Data directory '%s' not found — skipping session report.", data_dir)
        return

    # ── Load data ────────────────────────────────────────
    blink_files = sorted([f for f in os.listdir(data_dir) if f.startswith("blink_log") and f.endswith(".csv")])
    fatigue_files = sorted([f for f in os.listdir(data_dir) if f.startswith("fatigue_episodes") and f.endswith(".csv")])

    if not blink_files:
        logger.info("No blink data found — skipping session report.")
        return

    try:
        blink_df = pd.read_csv(os.path.join(data_dir, blink_files[-1]))
    except Exception as exc:
        logger.warning("Could not read latest blink log: %s", exc)
        return

    if blink_df.empty:
        logger.info("Latest blink log is empty — skipping session report.")
        return

    blink_df["time_sec"] = blink_df["timestamp"] - blink_df["timestamp"].iloc[0]

    # ── Smoothing ────────────────────────────────────────
    blink_df["bs_avg"] = blink_df["blink_score"].rolling(10, min_periods=1).mean()

    # ── Summary metrics ──────────────────────────────────
    metrics = {
        "Total Blinks": len(blink_df),
        "Mean Blink Score": blink_df["blink_score"].mean(),
        "Max Blink Score": blink_df["blink_score"].max(),
        "Mean Blink Duration (ms)": blink_df["duration_ms"].mean(),
        "Session Duration (min)": blink_df["time_sec"].iloc[-1] / 60,
    }

    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(report_dir, "metrics_table.csv"), index=False)

    # ── Plot 1: blink fatigue progression ────────────────
    plt.figure(figsize=(10, 4))
    plt.plot(blink_df["time_sec"] / 60, blink_df["blink_score"], alpha=0.3, label="Blink Score")
    plt.plot(blink_df["time_sec"] / 60, blink_df["bs_avg"], linewidth=2, label="Avg Blink Score")
    plt.xlabel("Time (minutes)")
    plt.ylabel("Blink Score")
    plt.title("Blink-Based Fatigue Progression")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(report_dir, "blink_fatigue_trend.png"))
    plt.close()

    # ── Plot 2: fatigue episodes ─────────────────────────
    if fatigue_files:
        try:
            fatigue_df = pd.read_csv(os.path.join(data_dir, fatigue_files[-1]))
            if not fatigue_df.empty:
                fatigue_df["start_min"] = (fatigue_df["start_time"] - blink_df["timestamp"].iloc[0]) / 60
                fatigue_df["end_min"] = (fatigue_df["end_time"] - blink_df["timestamp"].iloc[0]) / 60

                plt.figure(figsize=(10, 2))
                for _, row in fatigue_df.iterrows():
                    plt.axvspan(row["start_min"], row["end_min"], color="red", alpha=0.4)

                plt.xlabel("Time (minutes)")
                plt.title("Confirmed Fatigue Episodes")
                plt.tight_layout()
                plt.savefig(os.path.join(report_dir, "fatigue_timeline.png"))
                plt.close()
        except Exception as exc:
            logger.warning("Could not process fatigue episodes: %s", exc)

    # ── Summary text ─────────────────────────────────────
    with open(os.path.join(report_dir, "session_summary.txt"), "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v:.2f}\n")

    logger.info("Session analytics generated.")
