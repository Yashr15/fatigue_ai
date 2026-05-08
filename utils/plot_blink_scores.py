import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

def plot_blink_scores(csv_path):
    if not os.path.exists(csv_path):
        print("CSV file not found.")
        return

    df = pd.read_csv(csv_path)

    if df.empty:
        print("CSV file is empty.")
        return

    # Convert timestamp to relative time (seconds)
    t0 = df["timestamp"].iloc[0]
    df["time_sec"] = df["timestamp"] - t0

    plt.figure()

    # Raw blink scores
    plt.plot(
        df["time_sec"],
        df["blink_score"],
        marker='o',
        linestyle='-',
        alpha=0.4,
        label="Raw Blink Score"
    )

    # -------- ADD THESE TWO LINES HERE --------
    df["bs_avg"] = df["blink_score"].rolling(window=5, min_periods=1).mean()
    plt.plot(df["time_sec"], df["bs_avg"], linewidth=2, label="Avg Blink Score")
    # -----------------------------------------

    plt.xlabel("Time (seconds)")
    plt.ylabel("Blink Score (BSᵢ)")
    plt.title("Blink Score vs Time")
    plt.legend()
    plt.grid(True)
    plt.show()



if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_blink_scores.py <path_to_csv>")
    else:
        plot_blink_scores(sys.argv[1])


