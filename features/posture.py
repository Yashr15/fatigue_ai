"""
Ergonomic posture scoring based on forward head lean,
neck inclination, and posture instability (PCS).
"""

import numpy as np

# MediaPipe landmarks
CHIN = 152
LEFT_SHOULDER = 234
RIGHT_SHOULDER = 454
LEFT_EYE = 33
RIGHT_EYE = 263

# Defaults (overridable via config)
DEFAULT_FORWARD_DIST_NORM = 120.0
DEFAULT_NECK_ANGLE_NORM = 45.0


def compute_posture_score(
    landmarks,
    pcs_score,
    shape,
    forward_dist_normalizer=DEFAULT_FORWARD_DIST_NORM,
    neck_angle_normalizer=DEFAULT_NECK_ANGLE_NORM,
):
    """
    Ergonomic posture score (0-100) based on:
      - Forward head displacement
      - Neck inclination angle
      - Posture instability (PCS)

    Parameters
    ----------
    landmarks : list
        MediaPipe face mesh landmarks.
    pcs_score : float
        Current posture change score (0-100).
    shape : tuple[int, int]
        (height, width) of the image.
    forward_dist_normalizer : float
        Pixel distance at which forward_score = 100.
    neck_angle_normalizer : float
        Angle (degrees) at which neck_score = 100.
    """
    h, w = shape

    def pt(i):
        return np.array([landmarks[i].x * w, landmarks[i].y * h])

    chin = pt(CHIN)
    l_sh = pt(LEFT_SHOULDER)
    r_sh = pt(RIGHT_SHOULDER)

    shoulder_mid = (l_sh + r_sh) / 2

    # Forward head displacement
    forward_dist = np.linalg.norm(chin - shoulder_mid)
    forward_score = min(100, (forward_dist / forward_dist_normalizer) * 100)

    # Neck inclination
    neck_vec = chin - shoulder_mid
    vertical = np.array([0, 1])
    norm = np.linalg.norm(neck_vec)
    if norm == 0:
        angle = 0.0
    else:
        cos_angle = np.dot(neck_vec / norm, vertical)
        angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
    neck_score = min(100, (angle / neck_angle_normalizer) * 100)

    # Combine with instability
    posture_score = (
        0.4 * forward_score +
        0.4 * neck_score +
        0.2 * pcs_score
    )

    return float(np.clip(posture_score, 0, 100))
