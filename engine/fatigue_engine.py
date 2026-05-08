"""
FatigueEngine — the single source of truth for per-frame
fatigue analysis.  Integrates all feature extractors, fusion
logic, and data logging.
"""

from typing import Optional

import cv2
import mediapipe as mp
import logging

from config import FatigueConfig

from features.blink import compute_ear, BlinkTracker
from features.yawn import compute_mar, YawnTracker
from features.posture_change import PostureChangeTracker
from features.sitting_time import SittingTimeTracker
from features.overload import OverloadScore
from features.posture import compute_posture_score
from features.blink_logger import BlinkCSVLogger

from fusion.sliding_window import SlidingWindow
from fusion.cfi import compute_cfi
from fusion.trigger_confirm import TriggerConfirm
from fusion.hierarchical_confirm import HierarchicalConfirm

from utils.fatigue_logger import FatigueEpisodeLogger
from utils.session_analytics import generate_session_report
from utils.daily_analytics import generate_daily_report

logger = logging.getLogger(__name__)


class FatigueEngine:
    """
    Processes webcam frames end-to-end and returns a fatigue metrics dict.

    Parameters
    ----------
    config : FatigueConfig
        Central configuration object.  Uses defaults if not supplied.
    """

    def __init__(self, config: Optional[FatigueConfig] = None):
        self.cfg = config or FatigueConfig()

        # ── MediaPipe ────────────────────────────────────
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # ── Feature trackers ─────────────────────────────
        self.blink_tracker = BlinkTracker(
            ear_threshold=self.cfg.ear_threshold,
            microsleep_threshold=self.cfg.microsleep_threshold_sec,
            perclos_window_sec=self.cfg.perclos_window_sec,
        )
        self.yawn_tracker = YawnTracker(
            mar_thresh=self.cfg.mar_threshold,
            min_duration=self.cfg.min_yawn_duration_sec,
            window_sec=self.cfg.yawn_window_sec,
        )
        self.pcs_tracker = PostureChangeTracker(
            angle_delta=self.cfg.posture_change_angle_delta,
            window_sec=self.cfg.posture_change_window_sec,
        )
        self.sit_tracker = SittingTimeTracker(
            reset_threshold=self.cfg.sitting_reset_threshold,
            max_minutes=self.cfg.sitting_max_minutes,
        )
        self.os_tracker = OverloadScore(
            w_yawn=self.cfg.overload_w_yawn,
            w_pcs=self.cfg.overload_w_pcs,
            w_ts=self.cfg.overload_w_ts,
        )

        # ── Fusion ───────────────────────────────────────
        self.window = SlidingWindow(window_size=self.cfg.window_size)
        self.tc = TriggerConfirm(
            trigger_thresh=self.cfg.trigger_thresh,
            confirm_duration=self.cfg.confirm_duration_sec,
        )
        self.hc = HierarchicalConfirm(
            short_window_sec=self.cfg.short_window_sec,
            long_window_sec=self.cfg.long_window_sec,
            long_thresh=self.cfg.long_thresh,
        )

        # ── Logging ──────────────────────────────────────
        self.blink_logger = BlinkCSVLogger(save_dir=self.cfg.data_dir)
        self.fatigue_logger = FatigueEpisodeLogger(save_dir=self.cfg.data_dir)

    # ------------------------------------------------------------------ #
    def process_frame(self, frame):
        """
        Run the full pipeline on a single BGR frame.

        Returns
        -------
        dict or None
            Metrics dictionary, or ``None`` if no face was detected.
        """
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return None

        landmarks = results.multi_face_landmarks[0].landmark

        # ── Blink ────────────────────────────────────────
        ear = compute_ear(landmarks, (h, w))
        blink_duration, _, microsleep = self.blink_tracker.update(ear)
        perclos = self.blink_tracker.compute_perclos()
        avg_bs = self.blink_tracker.average_blink_score_last_n(10)

        # ── Yawn ─────────────────────────────────────────
        mar = compute_mar(landmarks, (h, w))
        self.yawn_tracker.update(mar)
        yawn_count = self.yawn_tracker.yawn_count()
        yawn_score = self.yawn_tracker.yawn_score()

        # ── Posture change ───────────────────────────────
        self.pcs_tracker.update(landmarks, (h, w))
        pcs = self.pcs_tracker.pcs_score()

        # ── Posture score ────────────────────────────────
        posture_score = compute_posture_score(
            landmarks,
            pcs_score=pcs,
            shape=(h, w),
            forward_dist_normalizer=self.cfg.forward_dist_normalizer,
            neck_angle_normalizer=self.cfg.neck_angle_normalizer,
        )

        # ── Sitting time ─────────────────────────────────
        self.sit_tracker.update(posture_score)
        sitting_score = self.sit_tracker.ts_score()

        # ── Overload ─────────────────────────────────────
        overload_score = self.os_tracker.compute(yawn_score, pcs, sitting_score)

        # ── Sliding window ───────────────────────────────
        feature_vector = [
            avg_bs,
            perclos,
            posture_score,
            yawn_score,
            pcs,
            sitting_score,
        ]
        self.window.add(feature_vector)

        cfi = None
        if self.window.is_full():
            X_t = self.window.get_tensor()
            cfi = compute_cfi(
                X_t,
                w_blink=self.cfg.cfi_w_blink,
                w_perclos=self.cfg.cfi_w_perclos,
                w_posture=self.cfg.cfi_w_posture,
                w_yawn=self.cfg.cfi_w_yawn,
                w_pcs=self.cfg.cfi_w_pcs,
                w_sitting=self.cfg.cfi_w_sitting,
            )

        # ── Trigger logic ────────────────────────────────
        short_alert = self.tc.update(cfi)
        final_alert = self.hc.update(cfi, short_alert)

        # ── Data logging ─────────────────────────────────
        self.fatigue_logger.update(final_alert)

        if blink_duration:
            self.blink_logger.log(
                blink_count=self.blink_tracker.blink_count,
                duration_ms=self.blink_tracker.last_blink_duration_ms,
                blink_score=self.blink_tracker.last_blink_score,
            )

        return {
            "blink_score": avg_bs,
            "blink_count": self.blink_tracker.blink_count,
            "perclos": perclos,
            "yawn_count": yawn_count,
            "yawn_score": yawn_score,
            "posture_score": posture_score,
            "posture_change_score": pcs,
            "sitting_time_score": sitting_score,
            "overload_score": overload_score,
            "cfi": cfi,
            "fatigue_confirmed": final_alert,
            "microsleep": microsleep,
        }

    # ------------------------------------------------------------------ #
    def generate_reports(self):
        """Generate session and daily analytics reports."""
        logger.info("Generating session report …")
        generate_session_report(
            data_dir=self.cfg.data_dir,
            report_dir=self.cfg.report_dir,
        )
        logger.info("Generating daily report …")
        generate_daily_report(
            data_root=self.cfg.data_dir,
            report_dir=self.cfg.report_dir,
        )

    # ------------------------------------------------------------------ #
    def release(self):
        """Clean up resources."""
        self.face_mesh.close()
        self.blink_logger.close()
        self.fatigue_logger.close()
        logger.info("Engine resources released.")
