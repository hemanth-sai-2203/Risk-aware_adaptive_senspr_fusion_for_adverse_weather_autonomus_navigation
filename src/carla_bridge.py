"""
carla_bridge.py — CARLA Connection + Sensor Parsing
=====================================================
Connects to CARLA, spawns the ego vehicle, attaches Camera / LiDAR / Radar,
and provides a single tick() method that returns parsed numpy arrays.
Compatible with CARLA 0.9.16 + Python 3.12.

FIX (D3D Crash): Camera replaced with GNSS sensor to prevent Intel Iris Xe crash.
                  tick() now returns a synthetic dark frame instead of None,
                  so the demo loop never gets a None image and skips frames.
"""

import queue
import time
import logging
import numpy as np
import carla

from config import (
    CARLA_HOST, CARLA_PORT, QUEUE_TIMEOUT,
    FIXED_DELTA_SECONDS,
    CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FOV, CAMERA_FPS,
    LIDAR_CHANNELS, LIDAR_RANGE, LIDAR_PPS, LIDAR_ROTATION_HZ,
    LIDAR_UPPER_FOV, LIDAR_LOWER_FOV,
    RADAR_HFOV, RADAR_VFOV, RADAR_RANGE, RADAR_PPS,
    CAMERA_MOUNT, LIDAR_MOUNT, RADAR_MOUNT,
)

logger = logging.getLogger(__name__)


def _tf(x, y, z, pitch=0.0, yaw=0.0, roll=0.0):
    """Make a CARLA Transform from plain numbers."""
    return carla.Transform(
        carla.Location(x=x, y=y, z=z),
        carla.Rotation(pitch=pitch, yaw=yaw, roll=roll),
    )


def _make_synthetic_frame() -> np.ndarray:
    """
    Creates a synthetic dark-blue 'camera' frame (RGB).
    Used because the real camera sensor is replaced with GNSS to prevent
    the D3D11/Intel Iris Xe fatal crash. All perception still works via
    CARLA ground-truth, so this is purely a visualisation placeholder.
    """
    img = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
    # Dark navy gradient to look like a night/dusk road scene
    for row in range(CAMERA_HEIGHT):
        intensity = int(10 + 25 * (row / CAMERA_HEIGHT))
        img[row, :] = (intensity, intensity + 8, intensity + 20)
    # Subtle horizon line
    h_row = CAMERA_HEIGHT // 2
    img[h_row - 1 : h_row + 1, :] = (30, 40, 60)
    return img


def parse_image(image) -> np.ndarray:
    """CARLA BGRA → RGB numpy (H, W, 3)."""
    raw  = np.frombuffer(image.raw_data, dtype=np.uint8)
    bgra = raw.reshape((image.height, image.width, 4))
    return bgra[:, :, :3][:, :, ::-1].copy()


def parse_lidar(meas) -> np.ndarray:
    """CARLA LiDAR → (N, 4) float32 [x, y, z, intensity]."""
    raw = np.frombuffer(meas.raw_data, dtype=np.float32)
    return raw.reshape((-1, 4)).copy()


def parse_radar(meas) -> np.ndarray:
    """CARLA Radar → (N, 4) float32 [azimuth, altitude, depth, velocity]."""
    n = len(meas)
    if n == 0:
        return np.zeros((0, 4), dtype=np.float32)
    data = np.zeros((n, 4), dtype=np.float32)
    for i, d in enumerate(meas):
        data[i] = [d.azimuth, d.altitude, d.depth, d.velocity]
    return data


class Carlabridge:
    """Manages the connection to CARLA and all sensor actors."""

    def __init__(self):
        self.client  = None
        self.world   = None
        self.tm      = None
        self.vehicle = None
        self.camera  = None   # Actually GNSS — avoids D3D crash
        self.lidar   = None
        self.radar   = None
        self._actors = []
        self._cam_q  = queue.Queue()
        self._lid_q  = queue.Queue()
        self._rad_q  = queue.Queue()

    # ── 1. CONNECT ────────────────────────────────────────────────────────────
    def connect(self):
        self.client = carla.Client(CARLA_HOST, CARLA_PORT)
        self.client.set_timeout(60.0)
        self.world  = self.client.get_world()

        # ── FORCE lightweight Town01 map to prevent GPU memory overflow ───────
        current_map = self.world.get_map().name
        if "Town01" not in current_map:
            logger.info("Map is '%s' — switching to Town01 (lightweight)...", current_map)
            self.world = self.client.load_world("Town01")
            time.sleep(5.0)
            logger.info("Map switched to Town01 successfully.")
        else:
            logger.info("Map is already Town01.")

        self.tm = self.client.get_trafficmanager(8000)
        self.tm.set_synchronous_mode(True)

        settings = self.world.get_settings()
        settings.synchronous_mode    = True
        settings.fixed_delta_seconds = FIXED_DELTA_SECONDS
        self.world.apply_settings(settings)
        logger.info("Connected to CARLA. Sync mode ON.")

    # ── 2. SPAWN VEHICLE ──────────────────────────────────────────────────────
    def spawn_vehicle(self):
        bp = self.world.get_blueprint_library().find("vehicle.tesla.model3")
        sp = self.world.get_map().get_spawn_points()
        self.vehicle = self.world.spawn_actor(bp, sp[0])
        self._actors.append(self.vehicle)
        self.vehicle.set_autopilot(True, 8000)
        self.tm.ignore_lights_percentage(self.vehicle, 100)
        self.tm.vehicle_percentage_speed_difference(self.vehicle, -20)
        logger.info("Ego vehicle spawned & autopilot ON.")
        return self.vehicle

    # ── 3. ATTACH SENSORS ─────────────────────────────────────────────────────
    def attach_sensors(self):
        lib = self.world.get_blueprint_library()

        # ── Camera replaced with GNSS to prevent D3D11 crash on Intel Iris Xe ──
        # ── REAL RGB CAMERA FOR YOLO ──────────────────────────────────────────
        # Warning: This increases GPU load drastically.
        cam_bp = lib.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(CAMERA_WIDTH))
        cam_bp.set_attribute("image_size_y", str(CAMERA_HEIGHT))
        cam_bp.set_attribute("fov", str(CAMERA_FOV))
        self.camera = self.world.spawn_actor(
            cam_bp, _tf(*CAMERA_MOUNT), attach_to=self.vehicle
        )
        
        # Helper to process raw CARLA BGRA to Numpy RGB
        def process_img(image):
            import numpy as np
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1] # BGR to RGB
            self._cam_q.put(array)
            
        self.camera.listen(process_img)
        self._actors.append(self.camera)

        # ── LiDAR ─────────────────────────────────────────────────────────────
        lid_bp = lib.find("sensor.lidar.ray_cast")
        lid_bp.set_attribute("channels",           str(LIDAR_CHANNELS))
        lid_bp.set_attribute("range",              str(LIDAR_RANGE))
        lid_bp.set_attribute("points_per_second",  str(LIDAR_PPS))
        lid_bp.set_attribute("rotation_frequency", str(LIDAR_ROTATION_HZ))
        lid_bp.set_attribute("upper_fov",          str(LIDAR_UPPER_FOV))
        lid_bp.set_attribute("lower_fov",          str(LIDAR_LOWER_FOV))
        self.lidar = self.world.spawn_actor(
            lid_bp, _tf(*LIDAR_MOUNT), attach_to=self.vehicle
        )
        self.lidar.listen(self._lid_q.put)
        self._actors.append(self.lidar)

        # ── Radar ─────────────────────────────────────────────────────────────
        rad_bp = lib.find("sensor.other.radar")
        rad_bp.set_attribute("horizontal_fov",    str(RADAR_HFOV))
        rad_bp.set_attribute("vertical_fov",      str(RADAR_VFOV))
        rad_bp.set_attribute("range",             str(RADAR_RANGE))
        rad_bp.set_attribute("points_per_second", str(RADAR_PPS))
        self.radar = self.world.spawn_actor(
            rad_bp, _tf(*RADAR_MOUNT), attach_to=self.vehicle
        )
        self.radar.listen(self._rad_q.put)
        self._actors.append(self.radar)

        logger.info("GNSS (camera-proxy) + LiDAR + Radar attached.")

    # ── 4. TICK ───────────────────────────────────────────────────────────────
    def tick(self):
        """
        Advance the simulation by one step and return sensor data.

        Returns:
            (img, lidar_pts, radar_pts)
            img       : (H, W, 3) uint8 RGB — synthetic frame (never None)
            lidar_pts : (N, 4) float32 or None
            radar_pts : (N, 4) float32 or None
        """
        self.world.tick()
        try:
            _c = self._cam_q.get(timeout=QUEUE_TIMEOUT)   # GNSS data — discard
            l  = self._lid_q.get(timeout=QUEUE_TIMEOUT)
            r  = self._rad_q.get(timeout=QUEUE_TIMEOUT)

            # Flush any stale frames that built up
            while not self._cam_q.empty(): self._cam_q.get_nowait()
            while not self._lid_q.empty(): self._lid_q.get_nowait()
            while not self._rad_q.empty(): self._rad_q.get_nowait()

            return _c, parse_lidar(l), parse_radar(r)

        except Exception as e:
            logger.warning("tick() sensor timeout: %s", e)
            return _make_synthetic_frame(), None, None

    # ── 5. DESTROY ────────────────────────────────────────────────────────────
    def destroy(self):
        for a in reversed(self._actors):
            try:
                if a and a.is_alive:
                    a.destroy()
            except Exception:
                pass
        self._actors = []
        try:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            self.world.apply_settings(settings)
        except Exception:
            pass
        logger.info("Cleanup complete.")
