# Fatigue AI

Real-time driver / operator fatigue detection using webcam-based face analysis.

## Features

- **Blink tracking** — Eye Aspect Ratio (EAR) averaged across both eyes, blink scoring, microsleep alerts
- **PERCLOS** — Time-weighted percentage of eye closure over a 60-second rolling window
- **Yawn detection** — Mouth Aspect Ratio (MAR) with configurable duration thresholds
- **Posture scoring** — Forward head lean, neck inclination, and posture instability
- **Sitting time** — Cumulative sitting duration with automatic break detection
- **Combined Fatigue Index (CFI)** — Weighted fusion of all six signals via sliding window
- **Trigger–Confirm logic** — Short-term trigger with hierarchical long-term confirmation
- **Session & daily reports** — Automated CSV and plot generation on exit

## Quick Start

```bash
# 1. Create virtual environment
python -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the monitor
python -m vision.main
```

Press **ESC** or **Ctrl+C** to stop.  Session reports are saved to `reports/`.

## Project Structure

```
fatigue_ai/
├── config.py                # Centralized configuration
├── requirements.txt
├── vision/
│   └── main.py              # Entry point (camera loop + overlay)
├── engine/
│   └── fatigue_engine.py    # Core pipeline (feature → fusion → logging)
├── features/
│   ├── blink.py             # EAR, BlinkTracker, PERCLOS
│   ├── blink_logger.py      # Blink CSV logger
│   ├── yawn.py              # MAR, YawnTracker
│   ├── posture.py           # Ergonomic posture score
│   ├── posture_change.py    # Posture Change Score (PCS)
│   ├── sitting_time.py      # Sitting time tracker
│   └── overload.py          # Overload score
├── fusion/
│   ├── sliding_window.py    # Fixed-size temporal window
│   ├── cfi.py               # Combined Fatigue Index
│   ├── trigger_confirm.py   # Short-term trigger–confirm
│   ├── hierarchical_confirm.py  # Long-term hierarchical confirm
│   └── ablation.py          # Diagnostic CFI variants
├── utils/
│   ├── fatigue_logger.py    # Fatigue episode logger
│   ├── session_analytics.py # Post-session plots & CSV
│   ├── daily_analytics.py   # Cross-session daily summary
│   ├── live_plot.py         # Real-time matplotlib plot
│   └── plot_blink_scores.py # Standalone blink-score plotter
├── data/                    # Runtime CSV logs (gitignored)
└── reports/                 # Generated reports (gitignored)
```

## Configuration

All tunable parameters are in [`config.py`](config.py).  Edit `FatigueConfig` fields to
adjust thresholds, fusion weights, window sizes, and data paths.
