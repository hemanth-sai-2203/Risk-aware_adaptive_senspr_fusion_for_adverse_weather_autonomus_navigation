"""
uncertainty.py — S.D.T Formula + Risk-Aware Speed Control
==========================================================
"Our Approach: Hand-crafted metrics (blur, point density, SNR)"
"Our Approach: S.D.T formula (derived from physical reasoning)"

The S.D.T (Sensor Degradation Trust) formula computes a single uncertainty
score [0.0 = perfect, 1.0 = completely blind] by combining three physical
metrics into a weighted sum:

    U = alpha * H_health + beta * H_disagree + gamma * H_jitter

Where:
    H_health   = 1 - normalized sensor health score
    H_disagree = sigmoid(lambda * min_distance) between unmatched detections
    H_jitter   = temporal variance of detection count

The Risk-Aware Speed is then computed as:
    speed = MAX_SPEED * (1 - U) ^ POWER
"""

import collections
import logging
import numpy as np

from config import (
    ALPHA_HEALTH, BETA_DISAGREEMENT, GAMMA_JITTER,
    DISAGREEMENT_LAMBDA, MAX_SPEED, EMERGENCY_BRAKE_THRESHOLD,
    SPEED_SCALE_POWER,
)

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# HAND-CRAFTED SENSOR METRICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_blur_score(img: np.ndarray) -> float:
    """
    Laplacian variance → measures image sharpness.
    High value = sharp (clear weather). Low value = blurry (fog/rain).
    Normalized to [0, 1] where 1 = perfectly clear.
    """
    if img is None or img.size == 0:
        return 0.0
    gray = img.mean(axis=2) if img.ndim == 3 else img
    lap = np.array([
        [0,  1, 0],
        [1, -4, 1],
        [0,  1, 0],
    ], dtype=np.float32)
    # Manual convolution (avoids cv2 dependency)
    from numpy.lib.stride_tricks import sliding_window_view
    try:
        windows = sliding_window_view(gray.astype(np.float32), (3, 3))
        response = (windows * lap).sum(axis=(-1, -2))
        variance = float(response.var())
    except Exception:
        variance = 0.0
    # Normalize: variance > 500 is "perfectly sharp"
    return min(1.0, variance / 500.0)


def compute_lidar_density_score(lidar_pts: np.ndarray) -> float:
    """
    LiDAR point density metric.
    Returns [0, 1] where 1 = full expected density.
    """
    if lidar_pts is None or lidar_pts.shape[0] == 0:
        return 0.0
    EXPECTED_PTS = 4000   # calibrated for LIDAR_PPS=5000
    return min(1.0, lidar_pts.shape[0] / EXPECTED_PTS)


def compute_radar_snr_score(radar_pts: np.ndarray) -> float:
    """
    Radar Signal-to-Noise Ratio: checks if detections are in a valid range.
    Returns [0, 1] where 1 = good SNR.
    """
    if radar_pts is None or radar_pts.shape[0] == 0:
        return 0.0
    EXPECTED_DETS = 6
    return min(1.0, radar_pts.shape[0] / EXPECTED_DETS)


def compute_health(img, lidar_pts, radar_pts) -> dict:
    """
    Combines all three sensor metrics into a single health report.
    Returns {cam_score, lidar_score, radar_score, overall_health, active_mode}
    """
    cam_score   = compute_blur_score(img)
    lidar_score = compute_lidar_density_score(lidar_pts)
    radar_score = compute_radar_snr_score(radar_pts)

    overall = (0.4 * cam_score + 0.4 * lidar_score + 0.2 * radar_score)

    # Active fusion mode (for display)
    if overall >= 0.70:
        mode = "GOLD"   # All sensors working well
    elif cam_score >= 0.50 and lidar_score >= 0.40:
        mode = "M1"     # Camera + LiDAR
    elif cam_score >= 0.30 and radar_score >= 0.30:
        mode = "M2"     # Camera + Radar
    else:
        mode = "DEGRADED"

    return {
        "cam_score":    round(cam_score, 3),
        "lidar_score":  round(lidar_score, 3),
        "radar_score":  round(radar_score, 3),
        "overall":      round(overall, 3),
        "active_mode":  mode,
    }


# ══════════════════════════════════════════════════════════════════════════════
# S.D.T FORMULA — UNCERTAINTY ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class UncertaintyEngine:
    """
    Computes the S.D.T Uncertainty score using physical reasoning.

    U = alpha * H_health + beta * H_disagree + gamma * H_jitter
    """

    def __init__(self):
        self._det_history = collections.deque(maxlen=10)
        logger.info(
            "UncertaintyEngine ready (alpha=%.2f beta=%.2f gamma=%.2f lam=%.4f)",
            ALPHA_HEALTH, BETA_DISAGREEMENT, GAMMA_JITTER, DISAGREEMENT_LAMBDA,
        )

    def compute(self, health: dict, unmatched_dets: list,
                n_detections: int) -> dict:
        """
        Args:
            health:         output of compute_health()
            unmatched_dets: list of detection indices not matched to any sensor
            n_detections:   total number of detections this frame

        Returns dict with: uncertainty, target_speed, mode
        """
        # ── Component 1: Health ──────────────────────────────────────────────
        H_health = 1.0 - health["overall"]

        # ── Component 2: Sensor Disagreement ─────────────────────────────────
        # Fraction of detections that could not be corroborated by LiDAR/Radar
        if n_detections > 0:
            disagree_frac = len(unmatched_dets) / n_detections
        else:
            disagree_frac = 0.0
        # Apply sigmoid scaling
        H_disagree = 1.0 / (1.0 + np.exp(-DISAGREEMENT_LAMBDA * 100 * disagree_frac))
        H_disagree = (H_disagree - 0.5) * 2.0   # re-scale to [0, 1]

        # ── Component 3: Temporal Jitter ──────────────────────────────────────
        self._det_history.append(n_detections)
        if len(self._det_history) >= 3:
            H_jitter = min(1.0, float(np.std(list(self._det_history))) / 5.0)
        else:
            H_jitter = 0.0

        # ── S.D.T Formula ────────────────────────────────────────────────────
        uncertainty = (
            ALPHA_HEALTH      * H_health   +
            BETA_DISAGREEMENT * H_disagree +
            GAMMA_JITTER      * H_jitter
        )
        
        # --- ENVIRONMENTAL RISK FLOOR ---
        # Even if 0 objects detected, force uncertainty up in bad weather
        if health.get("cam_score", 1.0) < 0.35: # Heavy Fog / Rain
            uncertainty = max(uncertainty, 0.45) # Force CAUTION (Orange)
        if health.get("cam_score", 1.0) < 0.22: # Heavy Rain
            uncertainty = max(uncertainty, 0.75) # Force DANGER (Red)

        uncertainty = float(np.clip(uncertainty, 0.0, 1.0))

        # ── Risk-Aware Speed ──────────────────────────────────────────────────
        if uncertainty >= EMERGENCY_BRAKE_THRESHOLD:
            target_speed = 0.0
            mode = "EMERGENCY_STOP"
        else:
            target_speed = MAX_SPEED * ((1.0 - uncertainty) ** SPEED_SCALE_POWER)
            mode = health["active_mode"]

        return {
            "uncertainty":  round(uncertainty, 4),
            "H_health":     round(H_health, 4),
            "H_disagree":   round(H_disagree, 4),
            "H_jitter":     round(H_jitter, 4),
            "target_speed": round(target_speed, 2),
            "mode":         mode,
        }
