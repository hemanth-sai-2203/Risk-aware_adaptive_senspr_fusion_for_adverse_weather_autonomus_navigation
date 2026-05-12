"""
config.py — RA-ASF Final Implementation Config
================================================
CARLA 0.9.16 | Python 3.12 | Intel Iris Xe (Low VRAM)
All settings tuned for a STABLE live demo.
"""

# ── CARLA Connection ──────────────────────────────────────────────────────────
CARLA_HOST   = "localhost"
CARLA_PORT   = 2000
QUEUE_TIMEOUT = 8.0        # seconds to wait for sensor data

# ── Simulation ───────────────────────────────────────────────────────────────
TOWN               = "Town02"       # small, stable map — loads fast
FIXED_DELTA_SECONDS = 0.05          # 20 FPS sync tick


CAMERA_WIDTH  = 640
CAMERA_HEIGHT = 480
CAMERA_FOV    = 90
CAMERA_FPS    = 20


LIDAR_CHANNELS    = 8
LIDAR_RANGE       = 100.0
LIDAR_PPS         = 2000
LIDAR_ROTATION_HZ = 20
LIDAR_UPPER_FOV   = 10.0
LIDAR_LOWER_FOV   = -30.0

RADAR_HFOV  = 60.0
RADAR_VFOV  = 10.0
RADAR_RANGE = 100.0
RADAR_PPS   = 1000

# ── Sensor Mount Positions (x, y, z, pitch, yaw, roll) ───────────────────────
CAMERA_MOUNT = (1.5, 0.0, 2.4,  0.0, 0.0, 0.0)
LIDAR_MOUNT  = (0.0, 0.0, 2.8,  0.0, 0.0, 0.0)
RADAR_MOUNT  = (2.0, 0.0, 1.0,  5.0, 0.0, 0.0)

# ── NPC Traffic ───────────────────────────────────────────────────────────────
N_VEHICLES    = 120
N_PEDESTRIANS = 5

# ── Camera Intrinsic Matrix (K) ───────────────────────────────────────────────
import numpy as np
_fx = CAMERA_WIDTH  / (2.0 * np.tan(np.radians(CAMERA_FOV / 2.0)))
_fy = _fx
CAMERA_K = np.array([
    [_fx, 0.0, CAMERA_WIDTH  / 2.0],
    [0.0, _fy, CAMERA_HEIGHT / 2.0],
    [0.0, 0.0, 1.0],
], dtype=np.float64)

# ── S.D.T / Uncertainty Parameters ───────────────────────────────────────────
ALPHA_HEALTH         = 0.30   # weight: sensor health
BETA_DISAGREEMENT    = 0.50   # weight: sensor disagreement
GAMMA_JITTER         = 0.20   # weight: temporal jitter
DISAGREEMENT_LAMBDA  = 0.005  # scale for disagreement distance

# ── Hungarian Matching ────────────────────────────────────────────────────────
M1_MATCH_THRESHOLD_PX = 80    # camera ↔ LiDAR match distance (pixels)
M2_MATCH_THRESHOLD_PX = 120   # camera ↔ Radar match distance (pixels)

# ── Fusion Filters ────────────────────────────────────────────────────────────
LIDAR_GROUND_Z   = -1.8       # ignore points below this height (road surface)
RADAR_MIN_DEPTH  = 2.0        # ignore radar detections too close
RADAR_MAX_DEPTH  = 80.0       # ignore radar detections too far

# ── Risk-Aware Speed Control ──────────────────────────────────────────────────
MAX_SPEED                = 30.0   # km/h
EMERGENCY_BRAKE_THRESHOLD = 0.85  # uncertainty above this → emergency stop
SPEED_SCALE_POWER        = 1.5

# ── PID Gains ────────────────────────────────────────────────────────────────
PID_KP = 0.5
PID_KI = 0.05
PID_KD = 0.10

# ── Demo Schedule ─────────────────────────────────────────────────────────────
DEMO_SCHEDULE     = ["clear", "fog_light", "fog_heavy", "rain"]
TICKS_PER_WEATHER = 200   # ticks per weather state (~10 seconds each)

# ── Results Output ────────────────────────────────────────────────────────────
import os
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
