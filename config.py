"""
Centralized configuration for the Fatigue AI system.

All tunable parameters live here so they can be adjusted
without touching module internals.
"""

from dataclasses import dataclass, field
import os


@dataclass
class FatigueConfig:
    # ── Blink detection ──────────────────────────────────
    ear_threshold: float = 0.21
    microsleep_threshold_sec: float = 0.5
    perclos_window_sec: float = 60.0

    # ── Yawn detection ───────────────────────────────────
    mar_threshold: float = 0.75
    min_yawn_duration_sec: float = 1.0
    yawn_window_sec: int = 300

    # ── Posture scoring ──────────────────────────────────
    forward_dist_normalizer: float = 120.0
    neck_angle_normalizer: float = 45.0

    # ── Posture change tracking ──────────────────────────
    posture_change_angle_delta: float = 10.0
    posture_change_window_sec: int = 300

    # ── Sitting time ─────────────────────────────────────
    sitting_reset_threshold: float = 20.0
    sitting_max_minutes: float = 60.0

    # ── Overload weights ─────────────────────────────────
    overload_w_yawn: float = 0.4
    overload_w_pcs: float = 0.3
    overload_w_ts: float = 0.3

    # ── CFI fusion weights (must sum to 1.0) ─────────────
    cfi_w_blink: float = 0.25
    cfi_w_perclos: float = 0.25
    cfi_w_posture: float = 0.15
    cfi_w_yawn: float = 0.15
    cfi_w_pcs: float = 0.10
    cfi_w_sitting: float = 0.10

    # ── Trigger / Confirm ────────────────────────────────
    trigger_thresh: float = 60.0
    confirm_duration_sec: float = 10.0

    # ── Hierarchical confirm ─────────────────────────────
    short_window_sec: float = 10.0
    long_window_sec: float = 300.0
    long_thresh: float = 55.0

    # ── Sliding window ───────────────────────────────────
    window_size: int = 30

    # ── Data paths (resolved relative to project root) ───
    data_dir: str = field(default_factory=lambda: os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data"
    ))
    report_dir: str = field(default_factory=lambda: os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "reports"
    ))
