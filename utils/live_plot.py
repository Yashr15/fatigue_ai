import matplotlib.pyplot as plt
import time
from collections import deque

class LiveFatiguePlot:
    def __init__(self, window_size=60):
        plt.ion()

        self.start_time = time.time()
        self.times = deque(maxlen=window_size)
        self.blink_scores = deque(maxlen=window_size)
        self.perclos_values = deque(maxlen=window_size)

        self.fig, self.ax = plt.subplots()
        self.line_bs, = self.ax.plot([], [], label="Blink Score", linewidth=2)
        self.line_pc, = self.ax.plot([], [], label="PERCLOS", linewidth=2)

        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Value")
        self.ax.set_ylim(0, 1.05)
        self.ax.legend()
        self.ax.set_title("Live Fatigue Signals")

    def update(self, blink_score, perclos):
        t = time.time() - self.start_time

        self.times.append(t)
        self.blink_scores.append(blink_score / 100.0)  # normalize
        self.perclos_values.append(perclos)

        self.line_bs.set_xdata(self.times)
        self.line_bs.set_ydata(self.blink_scores)

        self.line_pc.set_xdata(self.times)
        self.line_pc.set_ydata(self.perclos_values)

        self.ax.relim()
        self.ax.autoscale_view()

        plt.pause(0.01)
