"""
carla_setup.py
--------------
CarlaSetup : connects to CARLA 0.9.15 on Windows.
v3 (Tank Mode): Designed to survive severe Intel Iris map-loading disconnects.
"""

import sys
import os
import time
import queue
import logging

import carla
import numpy as np

from config import (
    CARLA_HOST, CARLA_PORT, CARLA_TIMEOUT,
    TOWN, FIXED_DELTA_SECONDS, SEED,
    CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FOV, CAMERA_FPS,
    LIDAR_CHANNELS, LIDAR_RANGE, LIDAR_PPS,
    LIDAR_ROTATION_HZ, LIDAR_UPPER_FOV, LIDAR_LOWER_FOV,
    RADAR_HFOV, RADAR_VFOV, RADAR_RANGE, RADAR_PPS,
    CAMERA_MOUNT, LIDAR_MOUNT, RADAR_MOUNT,
    QUEUE_TIMEOUT,
)

logger = logging.getLogger(__name__)

def _make_transform(x, y, z, pitch=0.0, yaw=0.0, roll=0.0):
    return carla.Transform(carla.Location(x=x, y=y, z=z), carla.Rotation(pitch=pitch, yaw=yaw, roll=roll))

def parse_camera_image(image):
    raw  = np.frombuffer(image.raw_data, dtype=np.uint8)
    bgra = raw.reshape((image.height, image.width, 4))
    return bgra[:, :, :3][:, :, ::-1].copy()

def parse_lidar(measurement):
    raw = np.frombuffer(measurement.raw_data, dtype=np.float32)
    return raw.reshape((-1, 4)).copy()

def parse_radar(measurement):
    n = len(measurement)
    if n == 0: return np.zeros((0, 4), dtype=np.float32)
    data = np.zeros((n, 4), dtype=np.float32)
    for i, det in enumerate(measurement):
        data[i] = [det.azimuth, det.altitude, det.depth, det.velocity]
    return data

class CarlaSetup:
    def __init__(self):
        self.client = None
        self.world = None
        self.tm = None
        self.vehicle = None
        self.camera = None
        self.lidar = None
        self.radar = None
        self._actors = []
        self._camera_queue = queue.Queue()
        self._lidar_queue = queue.Queue()
        self._radar_queue = queue.Queue()
        self._sync_active = False

    def connect(self):
        # 1. Initial Connection
        for _ in range(5):
            try:
                self.client = carla.Client(CARLA_HOST, CARLA_PORT)
                self.client.set_timeout(10.0)
                logger.info("Connected to CARLA %s", self.client.get_server_version())
                break
            except Exception:
                time.sleep(3)
        else:
            raise RuntimeError("Make sure CARLA is running.")

        # 2. Bruteforce Map Loading
        self.world = self.client.get_world()
        if TOWN not in self.world.get_map().name:
            logger.info("Commanding map change to %s. Python will likely lose connection...", TOWN)
            self.client.set_timeout(200.0)
            try:
                self.client.load_world(TOWN)
            except Exception:
                logger.warning("Connection lost during map swap (Expected on Intel Iris). Waiting for recovery...")

            # 3. Recovery Loop
            loaded = False
            for attempt in range(60): # wait up to 2 minutes
                try:
                    self.client = carla.Client(CARLA_HOST, CARLA_PORT)
                    self.client.set_timeout(5.0)
                    self.world = self.client.get_world()
                    if TOWN in self.world.get_map().name:
                        loaded = True
                        break
                except Exception:
                    pass
                time.sleep(2.0)
                logger.info("  polling CARLA server... (attempt %d/60)", attempt+1)

            if not loaded:
                raise RuntimeError(
                    f"\n\n[FATAL] Your Intel Iris GPU physically cannot hold {TOWN} in memory.\n"
                    "You MUST change the map. Open config.py and change TOWN='Town03' to TOWN='Town02'.\n"
                )
            logger.info("Server recovered! Map is now %s", TOWN)
            time.sleep(3.0)

        # 4. Traffic Manager Setup
        self.client.set_timeout(CARLA_TIMEOUT)
        self.tm = self.client.get_trafficmanager(8000)
        self.tm.set_random_device_seed(SEED)
        self.tm.set_synchronous_mode(True)

        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = FIXED_DELTA_SECONDS
        self.world.apply_settings(settings)
        self._sync_active = True
        logger.info("Sync mode ON.")

        # --- ADD THIS TRAFFIC LIGHT HACK HERE ---
        traffic_lights = self.world.get_actors().filter('*traffic_light*')
        for light in traffic_lights:
            light.set_red_time(3.0)
            light.set_green_time(10.0)
            light.set_yellow_time(1.0)
        logger.info("Traffic light durations aggressively reduced.")
        # ----------------------------------------

    def spawn_vehicle(self):
        lib = self.world.get_blueprint_library()
        bp  = lib.find("vehicle.tesla.model3")
        bp.set_attribute("role_name", "ego")
        spawn_points = self.world.get_map().get_spawn_points()
        rng = np.random.default_rng(SEED)
        indices = rng.permutation(len(spawn_points))
        for idx in indices[:10]:
            try:
                self.vehicle = self.world.spawn_actor(bp, spawn_points[int(idx)])
                self._actors.append(self.vehicle)
                self.vehicle.set_autopilot(True, 8000)
                self.tm.ignore_lights_percentage(self.vehicle, 0)
                self.tm.distance_to_leading_vehicle(self.vehicle, 2.0)
                return self.vehicle
            except RuntimeError:
                continue
        raise RuntimeError("Could not spawn vehicle.")

    def attach_sensors(self):
        self.camera = self._attach_camera()
        self.lidar  = self._attach_lidar()
        self.radar  = self._attach_radar()
        self.world.tick()

    def _attach_camera(self):
        bp = self.world.get_blueprint_library().find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", str(CAMERA_WIDTH))
        bp.set_attribute("image_size_y", str(CAMERA_HEIGHT))
        bp.set_attribute("fov", str(CAMERA_FOV))
        bp.set_attribute("sensor_tick", str(1.0 / CAMERA_FPS))
        sensor = self.world.spawn_actor(bp, _make_transform(*CAMERA_MOUNT), attach_to=self.vehicle)
        sensor.listen(self._camera_queue.put)
        self._actors.append(sensor)
        return sensor

    def _attach_lidar(self):
        bp = self.world.get_blueprint_library().find("sensor.lidar.ray_cast")
        bp.set_attribute("channels",           str(LIDAR_CHANNELS))
        bp.set_attribute("range",              str(LIDAR_RANGE))
        bp.set_attribute("points_per_second",  str(LIDAR_PPS))
        bp.set_attribute("rotation_frequency", str(LIDAR_ROTATION_HZ))
        bp.set_attribute("upper_fov",          str(LIDAR_UPPER_FOV))
        bp.set_attribute("lower_fov",          str(LIDAR_LOWER_FOV))
        bp.set_attribute("noise_stddev",       "0.0")
        
        sensor = self.world.spawn_actor(
            bp, _make_transform(*LIDAR_MOUNT), attach_to=self.vehicle
        )
        sensor.listen(self._lidar_queue.put)
        self._actors.append(sensor)
        return sensor

    def _attach_radar(self):
        bp = self.world.get_blueprint_library().find("sensor.other.radar")
        bp.set_attribute("horizontal_fov", str(RADAR_HFOV))
        bp.set_attribute("vertical_fov", str(RADAR_VFOV))
        bp.set_attribute("points_per_second", str(RADAR_PPS))
        bp.set_attribute("range", str(RADAR_RANGE))
        sensor = self.world.spawn_actor(bp, _make_transform(*RADAR_MOUNT), attach_to=self.vehicle)
        sensor.listen(self._radar_queue.put)
        self._actors.append(sensor)
        return sensor

    def tick(self):
        self.world.tick()
        try:
            cam_raw = self._camera_queue.get(timeout=QUEUE_TIMEOUT)
            lid_raw = self._lidar_queue.get(timeout=QUEUE_TIMEOUT)
            rad_raw = self._radar_queue.get(timeout=QUEUE_TIMEOUT)
        except queue.Empty as exc:
            raise queue.Empty("Sensor timed out.") from exc
        return parse_camera_image(cam_raw), parse_lidar(lid_raw), parse_radar(rad_raw)

    def get_bounding_boxes(self):
        labels = []
        for actor in self.world.get_actors():
            if actor.id == self.vehicle.id: continue
            if "vehicle" in actor.type_id: atype = "vehicle"
            elif "walker" in actor.type_id: atype = "pedestrian"
            else: continue
            bb = actor.bounding_box
            loc = actor.get_transform().location
            labels.append({
                "id": actor.id, "type": atype,
                "location": [round(loc.x, 3), round(loc.y, 3), round(loc.z, 3)],
                "extent": [round(bb.extent.x, 3), round(bb.extent.y, 3), round(bb.extent.z, 3)],
                "yaw": round(actor.get_transform().rotation.yaw, 2),
            })
        return labels

    def destroy(self):
        for s in [self.camera, self.lidar, self.radar]:
            if s:
                try: s.stop()
                except Exception: pass
        if self._sync_active and self.world:
            try:
                self.tm.set_synchronous_mode(False)
                settings = self.world.get_settings()
                settings.synchronous_mode = False
                self.world.apply_settings(settings)
            except Exception: pass
        for a in reversed(self._actors):
            try:
                if a.is_alive: a.destroy()
            except Exception: pass
        self._actors.clear()