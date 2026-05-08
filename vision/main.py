"""
Fatigue Monitor — single entry point.

Run from project root:
    python -m vision.main
"""

import cv2
import signal
import sys
import logging

from config import FatigueConfig
from engine.fatigue_engine import FatigueEngine

# ── Logging setup ────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Graceful shutdown ────────────────────────────────────
RUNNING = True


def _handle_exit(sig, frame):
    global RUNNING
    RUNNING = False


signal.signal(signal.SIGINT, _handle_exit)


# ── Main ─────────────────────────────────────────────────
def main():
    config = FatigueConfig()
    engine = FatigueEngine(config)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Cannot open camera (index 0). Check permissions or connection.")
        sys.exit(1)

    logger.info("Fatigue monitoring started. Press Ctrl+C or ESC to stop.")

    while RUNNING:
        ret, frame = cap.read()
        if not ret:
            logger.warning("Camera read failed — stopping.")
            break

        h, w, _ = frame.shape
        metrics = engine.process_frame(frame)

        # ── Minimal overlay ──────────────────────────────
        y = 30
        dy = 22

        def draw(text, color=(255, 255, 255)):
            nonlocal y
            cv2.putText(
                frame, text, (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
            )
            y += dy

        if metrics:
            cfi = metrics["cfi"]
            confirmed = metrics["fatigue_confirmed"]

            draw(
                f"CFI: {cfi:.1f}" if cfi is not None else "CFI: --",
                (0, 255, 0) if not confirmed else (0, 0, 255),
            )

            draw(f"Blink Count: {metrics['blink_count']}")
            draw(f"Blink Score: {metrics['blink_score']:.1f}")
            draw(f"PERCLOS: {metrics['perclos']:.2f}")

            draw(f"Yawns (5 min): {metrics['yawn_count']}")
            draw(f"Yawn Score: {metrics['yawn_score']:.1f}")

            draw(f"Posture Score: {metrics['posture_score']:.1f}")
            draw(f"Posture Change Score: {metrics['posture_change_score']:.1f}")

            draw(f"Sitting Time Score: {metrics['sitting_time_score']:.1f}")
            draw(f"Overload Score: {metrics['overload_score']:.1f}")

            draw(
                f"Fatigue: {'CONFIRMED' if confirmed else 'NORMAL'}",
                (0, 0, 255) if confirmed else (0, 255, 0),
            )

        cv2.imshow("Fatigue Monitor", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    # ── Cleanup ──────────────────────────────────────────
    logger.info("Stopping …")
    cap.release()
    cv2.destroyAllWindows()

    engine.generate_reports()
    engine.release()

    logger.info("Session report saved. Goodbye.")


if __name__ == "__main__":
    main()
