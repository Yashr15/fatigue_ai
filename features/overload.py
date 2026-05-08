"""
Overload Score — weighted combination of yawn, posture-change,
and sitting-time signals.
"""

# Defaults (overridable via config)
DEFAULT_W_YAWN = 0.4
DEFAULT_W_PCS = 0.3
DEFAULT_W_TS = 0.3


class OverloadScore:
    """
    Parameters
    ----------
    w_yawn, w_pcs, w_ts : float
        Weights for yawn score, posture change score, and sitting time score.
    """

    def __init__(
        self,
        w_yawn=DEFAULT_W_YAWN,
        w_pcs=DEFAULT_W_PCS,
        w_ts=DEFAULT_W_TS,
    ):
        self.w_yawn = w_yawn
        self.w_pcs = w_pcs
        self.w_ts = w_ts

    def compute(self, ys, pcs, ts):
        """Weighted overload score (0–100)."""
        os_i = (
            self.w_yawn * ys +
            self.w_pcs * pcs +
            self.w_ts * ts
        )
        return min(100.0, os_i)
