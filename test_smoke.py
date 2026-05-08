"""Quick smoke test: verify all imports work."""
import sys
sys.path.insert(0, ".")

try:
    from config import FatigueConfig
    print("1. config OK")

    from features.blink import compute_ear, BlinkTracker
    print("2. features.blink OK")

    from features.yawn import compute_mar, YawnTracker
    print("3. features.yawn OK")

    from features.posture import compute_posture_score
    print("4. features.posture OK")

    from features.posture_change import PostureChangeTracker
    print("5. features.posture_change OK")

    from features.sitting_time import SittingTimeTracker
    print("6. features.sitting_time OK")

    from features.overload import OverloadScore
    print("7. features.overload OK")

    from features.blink_logger import BlinkCSVLogger
    print("8. features.blink_logger OK")

    from fusion.sliding_window import SlidingWindow
    print("9. fusion.sliding_window OK")

    from fusion.cfi import compute_cfi
    print("10. fusion.cfi OK")

    from fusion.trigger_confirm import TriggerConfirm
    print("11. fusion.trigger_confirm OK")

    from fusion.hierarchical_confirm import HierarchicalConfirm
    print("12. fusion.hierarchical_confirm OK")

    from fusion.ablation import cfi_blink_only, cfi_no_posture
    print("13. fusion.ablation OK")

    from utils.fatigue_logger import FatigueEpisodeLogger
    print("14. utils.fatigue_logger OK")

    from utils.session_analytics import generate_session_report
    print("15. utils.session_analytics OK")

    from utils.daily_analytics import generate_daily_report
    print("16. utils.daily_analytics OK")

    from engine.fatigue_engine import FatigueEngine
    print("17. engine.fatigue_engine OK")

    # Quick functional test: CFI with 6-feature vector
    import numpy as np
    sw = SlidingWindow(window_size=5)
    for _ in range(5):
        sw.add([10.0, 0.05, 30.0, 0.0, 5.0, 10.0])
    assert sw.is_full()
    tensor = sw.get_tensor()
    assert tensor.shape == (5, 6), f"Expected (5,6), got {tensor.shape}"
    cfi_val = compute_cfi(tensor)
    assert 0 <= cfi_val <= 100, f"CFI out of range: {cfi_val}"
    print(f"18. CFI functional test OK (CFI={cfi_val:.1f})")

    print()
    print("ALL TESTS PASSED")
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
