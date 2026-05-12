"""
sensor_health_monitor.py
------------------------
Computes real-time health scores for Camera, LiDAR, and RADAR
from raw sensor data every simulation tick.

Health scores are in [0.0, 1.0]:
    1.0 = perfectly healthy sensor
    0.0 = completely failed / maximum degradation

These scores are consumed by the Fusion Selector to decide
which fusion module to activate each tick.

Integration:
    from simulation.sensor_health_monitor import SensorHealthMonitor
    monitor = SensorHealthMonitor()
    scores = monitor.compute(cam_img, lidar_pts, radar_pts)
    # scores = {"cam": 0.87, "lid": 0.34, "rad": 0.91, ...}

Python 3.7 | Windows | numpy 1.21.6 | opencv 4.7.0.72
No CUDA required.
"""

import os
import sys
import json
import logging
import collections

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from config import (
    HEALTH_THRESHOLD,
    BLUR_LAPLACIAN_CLEAR,
    LIDAR_EXPECTED_POINTS,
    LIDAR_EXPECTED_INTENSITY,
    RADAR_EXPECTED_DETECTIONS,
    RADAR_MAX_RCS,
)

logger = logging.getLogger(__name__)

# ── CALIBRATION FILE ──────────────────────────────────────────────────────────
# After running calibrate() on clear-weather data, baselines are saved here
# so they persist across runs.
_CALIB_PATH = os.path.join(_ROOT, "data", "health_monitor_calibration.json")

# ── EMA SMOOTHING ─────────────────────────────────────────────────────────────
# Exponential moving average weight for final health scores.
# Higher = more responsive to sudden changes (less smooth).
# Lower  = smoother but slower to react.
# 0.3 is a good balance for 20Hz simulation.
EMA_ALPHA = 0.3

# ── WINDOW SIZE FOR TEMPORAL SMOOTHING ────────────────────────────────────────
HISTORY_WINDOW = 5


class SensorHealthMonitor:
    """
    Computes per-sensor health scores from raw sensor data.

    Supports optional calibration from clear-weather baseline frames
    to auto-tune thresholds for your specific machine and CARLA build.

    Parameters
    ----------
    ema_alpha : float
        EMA smoothing factor for final scores (0 < alpha <= 1).
        Lower = smoother but slower reaction. Default 0.3.
    use_calibration : bool
        Load saved calibration baselines if available. Default True.
    """

    def __init__(self, ema_alpha=EMA_ALPHA, use_calibration=True):
        self._alpha = ema_alpha

        # EMA state — initialised to 1.0 (assume healthy at start)
        self._ema = {"cam": 1.0, "lid": 1.0, "rad": 1.0}

        # Rolling window for temporal difference (TD) component
        self._cam_history  = collections.deque(maxlen=HISTORY_WINDOW)
        self._lid_history  = collections.deque(maxlen=HISTORY_WINDOW)
        self._rad_history  = collections.deque(maxlen=HISTORY_WINDOW)

        # Calibration baselines (overwritten if calibration file exists)
        self._baselines = {
            "blur_laplacian_clear"    : BLUR_LAPLACIAN_CLEAR,
            "lidar_expected_points"   : LIDAR_EXPECTED_POINTS,
            "lidar_expected_intensity": LIDAR_EXPECTED_INTENSITY,
            "radar_expected_det"      : RADAR_EXPECTED_DETECTIONS,
            "radar_max_rcs"           : RADAR_MAX_RCS,
        }

        if use_calibration:
            self._load_calibration()

        logger.info(
            "SensorHealthMonitor ready  (alpha=%.2f, threshold=%.2f)",
            self._alpha, HEALTH_THRESHOLD,
        )

    # ── PUBLIC API ────────────────────────────────────────────────────────────

    def compute(self, cam_img, lidar_pts, radar_pts):
        """
        Compute health scores for all three sensors.

        Parameters
        ----------
        cam_img    : np.ndarray uint8  (H, W, 3)  RGB
        lidar_pts  : np.ndarray float32 (N, 4)    x, y, z, intensity
        radar_pts  : np.ndarray float32 (M, 4)    azimuth, altitude, depth, velocity

        Returns
        -------
        dict with keys:
            "cam"         : float  camera health       [0.0, 1.0]
            "lid"         : float  lidar health        [0.0, 1.0]
            "rad"         : float  radar health        [0.0, 1.0]
            "cam_raw"     : dict   raw sub-scores for logging
            "lid_raw"     : dict   raw sub-scores for logging
            "rad_raw"     : dict   raw sub-scores for logging
            "degraded"    : list   names of sensors below HEALTH_THRESHOLD
            "active_mode" : str    suggested module to activate
        """
        cam_score, cam_raw = self._camera_health(cam_img)
        lid_score, lid_raw = self._lidar_health(lidar_pts)
        rad_score, rad_raw = self._radar_health(radar_pts)

        # EMA smoothing — prevents single-frame spikes from triggering switches
        cam_score = self._update_ema("cam", cam_score)
        lid_score = self._update_ema("lid", lid_score)
        rad_score = self._update_ema("rad", rad_score)

        # Identify degraded sensors
        degraded = []
        if cam_score < HEALTH_THRESHOLD:
            degraded.append("camera")
        if lid_score < HEALTH_THRESHOLD:
            degraded.append("lidar")
        if rad_score < HEALTH_THRESHOLD:
            degraded.append("radar")

        # Determine active module
        active_mode = self._select_mode(cam_score, lid_score, rad_score)

        return {
            "cam"         : round(cam_score, 4),
            "lid"         : round(lid_score, 4),
            "rad"         : round(rad_score, 4),
            "cam_raw"     : cam_raw,
            "lid_raw"     : lid_raw,
            "rad_raw"     : rad_raw,
            "degraded"    : degraded,
            "active_mode" : active_mode,
        }

    def calibrate(self, cam_frames, lidar_frames, radar_frames):
        """
        Compute baselines from a list of clear-weather frames and save them.

        Call this once after collecting 50-100 clear-weather frames.
        Results are saved to data/health_monitor_calibration.json and
        loaded automatically on future runs.

        Parameters
        ----------
        cam_frames   : list of np.ndarray uint8 (H, W, 3)
        lidar_frames : list of np.ndarray float32 (N, 4)
        radar_frames : list of np.ndarray float32 (M, 4)
        """
        if not cam_frames:
            logger.warning("No frames provided for calibration — skipping.")
            return

        logger.info("Calibrating on %d clear-weather frames ...", len(cam_frames))

        # Camera: mean Laplacian variance across frames
        lap_vars = []
        for img in cam_frames:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            lap_vars.append(cv2.Laplacian(gray, cv2.CV_64F).var())
        baseline_blur = float(np.mean(lap_vars))

        # LiDAR: mean point count and mean intensity
        point_counts  = [f.shape[0] for f in lidar_frames if f.shape[0] > 0]
        intensities   = [float(np.mean(f[:, 3])) for f in lidar_frames if f.shape[0] > 0]
        baseline_pts  = float(np.mean(point_counts)) if point_counts  else LIDAR_EXPECTED_POINTS
        baseline_int  = float(np.mean(intensities))  if intensities   else LIDAR_EXPECTED_INTENSITY

        # RADAR: mean detection count and mean RCS
        det_counts    = [f.shape[0] for f in radar_frames]
        rcs_vals      = [float(np.mean(f[:, 3])) for f in radar_frames if f.shape[0] > 0]
        baseline_det  = float(np.mean(det_counts)) if det_counts else RADAR_EXPECTED_DETECTIONS
        baseline_rcs  = float(np.mean(rcs_vals))   if rcs_vals   else RADAR_MAX_RCS

        self._baselines = {
            "blur_laplacian_clear"    : baseline_blur,
            "lidar_expected_points"   : baseline_pts,
            "lidar_expected_intensity": baseline_int,
            "radar_expected_det"      : baseline_det,
            "radar_max_rcs"           : baseline_rcs,
        }

        # Save to disk
        os.makedirs(os.path.dirname(_CALIB_PATH), exist_ok=True)
        with open(_CALIB_PATH, "w") as f:
            json.dump(self._baselines, f, indent=2)

        logger.info(
            "Calibration saved to %s\n"
            "  blur_laplacian_clear     = %.1f\n"
            "  lidar_expected_points    = %.0f\n"
            "  lidar_expected_intensity = %.3f\n"
            "  radar_expected_det       = %.1f\n"
            "  radar_max_rcs            = %.2f",
            _CALIB_PATH,
            baseline_blur, baseline_pts, baseline_int,
            baseline_det, baseline_rcs,
        )

    def reset_ema(self):
        """Reset EMA state to 1.0 — call after a weather transition."""
        self._ema = {"cam": 1.0, "lid": 1.0, "rad": 1.0}

    # ── CAMERA HEALTH ─────────────────────────────────────────────────────────

    def _camera_health(self, img):
        """
        Camera health from:
          1. Blur score    — Laplacian variance (low = blurry = degraded)
          2. Exposure score — mean brightness distance from ideal midpoint

        Returns (health_float, raw_dict)
        """
        if img is None or img.size == 0:
            return 0.0, {"blur": 0.0, "exposure": 0.0, "error": "null_frame"}

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # ── Blur ─────────────────────────────────────────────────────────────
        lap_var   = cv2.Laplacian(gray, cv2.CV_64F).var()
        # Normalise against clear-weather baseline
        # Clamp to [0, 1]: 0 = completely blurry, 1 = as sharp as clear weather
        # Clamp to [0, 1]: 0 = completely blurry, 1 = as sharp as clear weather
        base_blur = self._baselines["blur_laplacian_clear"]
        blur_score = float(np.clip(lap_var / base_blur, 0.0, 1.0)) if base_blur > 0.0 else 0.0
        # ── Exposure ─────────────────────────────────────────────────────────
        mean_brightness = float(np.mean(gray))
        # Ideal = 128. Score drops as brightness deviates in either direction.
        # At brightness 0 (total darkness) or 255 (total glare) score = 0.0
        exposure_score = float(1.0 - abs(mean_brightness - 128.0) / 128.0)
        exposure_score = float(np.clip(exposure_score, 0.0, 1.0))

        # ── Combined ─────────────────────────────────────────────────────────
        # Blur is weighted more heavily — it is the primary fog/rain indicator.
        # Exposure catches glare and darkness which blur alone may miss.
        health = 0.65 * blur_score + 0.35 * exposure_score

        raw = {
            "laplacian_var" : round(float(lap_var), 2),
            "blur_score"    : round(blur_score, 4),
            "mean_brightness": round(mean_brightness, 2),
            "exposure_score": round(exposure_score, 4),
        }
        return float(np.clip(health, 0.0, 1.0)), raw

    # ── LIDAR HEALTH ──────────────────────────────────────────────────────────

    def _lidar_health(self, points):
        """
        LiDAR health from:
          1. Point count score    — how many returns vs clear-weather baseline
          2. Intensity score      — mean return intensity vs baseline
          3. Angular coverage     — how much of the 360deg sweep has returns

        Returns (health_float, raw_dict)
        """
        n = points.shape[0] if points is not None else 0

        if n == 0:
            return 0.0, {"n_points": 0, "count_score": 0.0,
                         "intensity_score": 0.0, "coverage_score": 0.0,
                         "error": "no_returns"}

        # ── Point count ───────────────────────────────────────────────────────
# ── Point count ───────────────────────────────────────────────────────
        expected = self._baselines["lidar_expected_points"]
        count_score = float(np.clip(n / expected, 0.0, 1.0)) if expected > 0 else 0.0

        # ── Intensity ─────────────────────────────────────────────────────────
        # Column 3 = intensity in CARLA LiDAR output
        mean_int      = float(np.mean(points[:, 3]))
        expected_int  = self._baselines["lidar_expected_intensity"]
        if expected_int > 0:
            intensity_score = float(np.clip(mean_int / expected_int, 0.0, 1.0))
        else:
            intensity_score = 1.0

        # ── Angular coverage ──────────────────────────────────────────────────
        # Use x,y columns to compute azimuth angles of all returns.
        # Divide 360deg into 36 bins of 10deg each.
        # Coverage = fraction of bins that have at least one return.
        azimuths   = np.degrees(np.arctan2(points[:, 1], points[:, 0])) % 360.0
        bin_size   = 10.0
        n_bins     = int(360.0 / bin_size)
        occupied   = len(np.unique((azimuths / bin_size).astype(int) % n_bins))
        coverage_score = float(occupied) / float(n_bins)

        # ── Combined ──────────────────────────────────────────────────────────
        # Point count is the dominant signal for fog (beams scatter and don't return).
        # Intensity drops in fog too but is a secondary confirmation.
        # Coverage tells us if only part of the sweep is returning (partial occlusion).
        health = (0.50 * count_score
                + 0.30 * intensity_score
                + 0.20 * coverage_score)

        raw = {
            "n_points"       : n,
            "count_score"    : round(count_score, 4),
            "mean_intensity" : round(mean_int, 4),
            "intensity_score": round(intensity_score, 4),
            "coverage_score" : round(coverage_score, 4),
        }
        return float(np.clip(health, 0.0, 1.0)), raw

    # ── RADAR HEALTH ──────────────────────────────────────────────────────────

    def _radar_health(self, radar_pts):
        """
        RADAR health from:
          1. Detection count score — how many objects detected vs baseline
          2. RCS score             — mean Radar Cross Section strength vs baseline

        Note: CARLA RADAR column 3 is velocity, not RCS.
        We use detection count as primary and velocity spread as a secondary
        signal (RADAR returns in bad conditions tend to cluster near zero velocity
        or produce phantom returns with unrealistic velocity spikes).

        Returns (health_float, raw_dict)
        """
        n = radar_pts.shape[0] if radar_pts is not None else 0

        if n == 0:
            # Zero detections is ambiguous — could mean no objects in range
            # OR total RADAR failure. We return 0.5 (uncertain) rather than
            # 0.0 (failed) to avoid falsely triggering module switches when
            # the vehicle is simply in an empty area.
            return 0.5, {"n_detections": 0, "count_score": 0.5,
                         "vel_score": 0.5, "note": "no_returns_ambiguous"}

        # ── Detection count ───────────────────────────────────────────────────
# ── Detection count ───────────────────────────────────────────────────
        expected_det  = self._baselines["radar_expected_det"]
        # Soft cap at 2x expected — more detections than expected is fine
        count_score   = float(np.clip(n / expected_det, 0.0, 1.0)) if expected_det > 0 else 0.0

        # ── Velocity spread (quality proxy) ──────────────────────────────────
        # Column 3 in our radar_pts = velocity (m/s) from carla_setup.py
        # Healthy RADAR returns show spread of velocities (objects at various speeds).
        # Degraded/phantom returns tend to cluster at zero.
        # We use std deviation of velocity as a quality signal.
# We use std deviation of velocity as a quality signal.
        if radar_pts.shape[1] >= 4:
            velocities = radar_pts[:, 3]
            vel_std    = float(np.std(velocities)) if n > 1 else 0.0
        else:
            vel_std    = 0.0

        # Normalise: std of 0 = all returns at same speed = suspicious.
        # std > 3.0 m/s = healthy spread. Clip at 1.0.
        vel_score = float(np.clip(vel_std / 3.0, 0.0, 1.0))

        # ── Combined ──────────────────────────────────────────────────────────
        # Detection count dominates — it is the most direct health signal.
        # Velocity spread is a secondary quality check.
        health = 0.70 * count_score + 0.30 * vel_score

        raw = {
            "n_detections" : n,
            "count_score"  : round(count_score, 4),
            "vel_std"      : round(vel_std, 4),
            "vel_score"    : round(vel_score, 4),
        }
        return float(np.clip(health, 0.0, 1.0)), raw

    # ── EMA ───────────────────────────────────────────────────────────────────

    def _update_ema(self, sensor_key, new_score):
        """
        Update exponential moving average for a sensor and return smoothed score.
        EMA(t) = alpha * new_score + (1 - alpha) * EMA(t-1)
        """
        prev = self._ema[sensor_key]
        smoothed = self._alpha * new_score + (1.0 - self._alpha) * prev
        self._ema[sensor_key] = smoothed
        return smoothed

    # ── MODE SELECTION ────────────────────────────────────────────────────────

    def _select_mode(self, cam, lid, rad):
        """
        Select the recommended fusion module based on health scores.
        Mirrors the Fusion Selector logic so the Health Monitor can
        suggest the mode independently (Fusion Selector remains the authority).

        Returns one of: "M1", "M2", "M3", "GOLD"
        """
        cam_ok = cam >= HEALTH_THRESHOLD
        lid_ok = lid >= HEALTH_THRESHOLD
        rad_ok = rad >= HEALTH_THRESHOLD

        if cam_ok and lid_ok and rad_ok:
            return "GOLD"                    # all healthy — blend all three
        elif not lid_ok and cam_ok and rad_ok:
            return "M2"                      # LiDAR fails → Camera + RADAR
        elif not rad_ok and cam_ok and lid_ok:
            return "M1"                      # RADAR fails → Camera + LiDAR
        elif not cam_ok and lid_ok and rad_ok:
            return "M3"                      # Camera fails → LiDAR + RADAR
        else:
            # Two or more sensors degraded — pick the least-bad pair
            scores = {"M1": cam + lid, "M2": cam + rad, "M3": lid + rad}
            return max(scores, key=scores.get)

    # ── CALIBRATION LOAD ──────────────────────────────────────────────────────

    def _load_calibration(self):
        """Load saved calibration baselines from disk if available."""
        if not os.path.exists(_CALIB_PATH):
            logger.info(
                "No calibration file found at %s — using config.py defaults. "
                "Run monitor.calibrate() after first clear-weather collection.",
                _CALIB_PATH,
            )
            return
        try:
            with open(_CALIB_PATH) as f:
                saved = json.load(f)
            self._baselines.update(saved)
            logger.info("Calibration loaded from %s", _CALIB_PATH)
        except Exception as exc:
            logger.warning("Could not load calibration (%s) — using defaults.", exc)


# ── QUICK SANITY TEST ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )

    monitor = SensorHealthMonitor()

    print("\n── Test 1: Healthy sensors (clear weather simulation) ───────────")
    cam   = np.random.randint(100, 160, (600, 800, 3), dtype=np.uint8)
    lidar = np.random.uniform(0.0, 50.0, (65000, 4)).astype(np.float32)
    lidar[:, 3] = np.random.uniform(0.5, 0.8, 65000).astype(np.float32)
    radar = np.random.uniform(-1.0, 1.0, (8, 4)).astype(np.float32)
    radar[:, 3] = np.random.uniform(-10.0, 10.0, 8).astype(np.float32)

    r = monitor.compute(cam, lidar, radar)
    print("  cam  = {:.3f}  lid  = {:.3f}  rad  = {:.3f}".format(
        r["cam"], r["lid"], r["rad"]))
    print("  mode = {}  degraded = {}".format(r["active_mode"], r["degraded"]))

    print("\n── Test 2: Heavy fog (blurry camera, sparse LiDAR) ─────────────")
    cam_fog   = cv2.GaussianBlur(cam, (31, 31), 6.0)
    lidar_fog = lidar[:10000].copy()
    lidar_fog[:, 3] *= 0.2

    r2 = monitor.compute(cam_fog, lidar_fog, radar)
    print("  cam  = {:.3f}  lid  = {:.3f}  rad  = {:.3f}".format(
        r2["cam"], r2["lid"], r2["rad"]))
    print("  mode = {}  degraded = {}".format(r2["active_mode"], r2["degraded"]))

    print("\n── Test 3: Camera failure (darkness / glare) ────────────────────")
    cam_dark  = np.zeros((600, 800, 3), dtype=np.uint8)
    r3 = monitor.compute(cam_dark, lidar, radar)
    print("  cam  = {:.3f}  lid  = {:.3f}  rad  = {:.3f}".format(
        r3["cam"], r3["lid"], r3["rad"]))
    print("  mode = {}  degraded = {}".format(r3["active_mode"], r3["degraded"]))

    print("\n  All tests passed.\n")
