"""
data_collector.py
-----------------
Collects 2000 synchronized frames (500 x 4 weather states) from CARLA 0.9.15.
Saves Camera, LiDAR, RADAR, and 2D projected bounding box labels to disk.

Fixes applied in this version:
  1. NPC vehicles + pedestrians spawned so scene is populated
  2. Ego vehicle ignores traffic lights (ignore_lights_percentage = 100)
  3. Ego vehicle speed increased for better map coverage
  4. Labels now use LabelGenerator for proper 2D bbox projection
  5. NPC respawn between weather states to keep scene populated

Windows: run from project root
    cd C:\\Users\\heman\\Music\\ra_asf
    python simulation\\data_collector.py
"""

import os
import sys
import json
import queue
import logging
import math
from turtle import setup

import cv2
import carla
import numpy as np
from tqdm import tqdm

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from config import (
    DATA_ROOT, WEATHER_STATES, FRAMES_PER_WEATHER, DEGRADATION,
)
from simulation.carla_setup     import CarlaSetup
from simulation.weather_engine  import WeatherEngine
from simulation.npc_manager     import NpcManager
from simulation.label_generator import LabelGenerator

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt = "%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── SPECTATOR ─────────────────────────────────────────────────────────────────

def spectator_follow(world, vehicle):
    t   = vehicle.get_transform()
    yaw = math.radians(t.rotation.yaw)
    world.get_spectator().set_transform(carla.Transform(
        carla.Location(
            x=t.location.x - 10 * math.cos(yaw),
            y=t.location.y - 10 * math.sin(yaw),
            z=t.location.z + 6,
        ),
        carla.Rotation(pitch=-20, yaw=t.rotation.yaw),
    ))


# ── DEGRADATION ───────────────────────────────────────────────────────────────

def apply_camera_degradation(img, state):
    sigma = DEGRADATION[state]["blur_sigma"]
    if sigma <= 0.0:
        return img
    ksize = int(6 * sigma + 1) | 1
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)


def apply_lidar_degradation(points, state):
    if points.shape[0] == 0:
        return points
    dropout = DEGRADATION[state]["lidar_dropout"]
    scale   = DEGRADATION[state]["lidar_intensity_scale"]
    result  = points.copy()
    result[:, 3] *= scale
    if dropout > 0.0:
        keep   = np.random.random(len(result)) > dropout
        result = result[keep]
    return result


# ── SAVING ────────────────────────────────────────────────────────────────────

def create_directories():
    for weather in WEATHER_STATES:
        for sub in ["images", "lidar", "radar", "labels"]:
            os.makedirs(os.path.join(DATA_ROOT, weather, sub), exist_ok=True)
    logger.info("Output directories ready: %s", DATA_ROOT)


def save_frame(frame_id, weather, img, lidar_pts, radar_pts, labels):
    base  = os.path.join(DATA_ROOT, weather)
    fname = "frame_{:06d}".format(frame_id)

    cv2.imwrite(
        os.path.join(base, "images", fname + ".jpg"),
        cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, 95],
    )
    np.save(os.path.join(base, "lidar",  fname + ".npy"), lidar_pts)
    np.save(os.path.join(base, "radar",  fname + ".npy"), radar_pts)

    with open(os.path.join(base, "labels", fname + ".json"), "w") as f:
        json.dump({
            "frame_id"  : frame_id,
            "weather"   : weather,
            "num_points": int(lidar_pts.shape[0]),
            "num_radar" : int(radar_pts.shape[0]),
            "objects"   : labels,
        }, f, indent=2)


# ── WARM-UP ───────────────────────────────────────────────────────────────────

def warm_up(setup, n_ticks=40):
    """Discard first n_ticks to let NPCs spread out and sensors stabilise."""
    logger.info("Warming up (%d ticks) ...", n_ticks)
    for _ in range(n_ticks):
        try:
            setup.tick()
        except queue.Empty:
            pass




# ── MAIN ──────────────────────────────────────────────────────────────────────

def collect():
    create_directories()
    setup = CarlaSetup()
    npc   = None

    try:
        setup.connect()
        setup.spawn_vehicle()

        # ── FIX 1: ego ignores ALL traffic lights ─────────────────────────
        # This was the root cause of the vehicle stopping and only covering
        # 200m. Now the ego drives through the map continuously.
        setup.tm.ignore_lights_percentage(setup.vehicle, 100)

        # ── FIX 2: increase ego speed ─────────────────────────────────────
        # Negative % = faster than TM default speed.
        # -30 means 30% FASTER than the default — better map coverage.
        setup.tm.vehicle_percentage_speed_difference(setup.vehicle, -30)

        setup.attach_sensors()

        # ── FIX 3: spawn NPCs so scene has objects ────────────────────────
        npc = NpcManager(setup.client, setup.world, tm_port=8000)
        npc.spawn_all(setup.vehicle)

        # ── FIX 4: proper 2D label generator ─────────────────────────────
        labeler = LabelGenerator(
            world         = setup.world,
            ego_vehicle   = setup.vehicle,
            camera_sensor = setup.camera,
        )

        engine = WeatherEngine(setup.world)
        total  = 0

        for weather in WEATHER_STATES:
            logger.info("=" * 56)
            logger.info(
                "Weather: %-12s  target: %d frames",
                weather, FRAMES_PER_WEATHER,
            )
            logger.info("=" * 56)

            # Respawn NPCs between weather states — keeps population fresh
            npc.destroy_all()
            npc.spawn_all(setup.vehicle)

            engine.set_weather(weather)
            warm_up(setup, n_ticks=40)

            collected  = 0
            errors     = 0
            MAX_ERRORS = 20

            with tqdm(total=FRAMES_PER_WEATHER, desc=weather, unit="fr") as pbar:
                while collected < FRAMES_PER_WEATHER:
                    try:
                        cam_img, lidar_pts, radar_pts = setup.tick()
                    except queue.Empty:
                        errors += 1
                        logger.warning(
                            "Timeout %d/%d — skipping tick", errors, MAX_ERRORS
                        )
                        if errors >= MAX_ERRORS:
                            logger.error(
                                "Too many timeouts. "
                                "Reduce LIDAR_PPS in config.py."
                            )
                            break
                        continue

                    errors = 0

                    spectator_follow(setup.world, setup.vehicle)

                    # ── FRAME SKIPPING LOGIC ──────────────────────────────────
                    # 1. Frame Decimation: Only process every 2nd tick to increase physical distance
                    if total % 2 != 0:
                        total += 1
                        continue

                    # 2. Smart Frame Dropping: Don't bloat data if stuck in traffic
                    velocity = setup.vehicle.get_velocity()
                    speed_mps = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
                    
                    if speed_mps < 0.5: # Vehicle is effectively stopped
                        if total % 20 != 0: # Only save 1 out of every 20 frames while stopped
                            total += 1
                            continue
                    # ──────────────────────────────────────────────────────────

                    cam_img   = apply_camera_degradation(cam_img, weather)
                    lidar_pts = apply_lidar_degradation(lidar_pts, weather)

                    # 2D bounding boxes projected from 3D world → camera plane
                    labels = labeler.get_labels()

                    save_frame(
                        total, weather, cam_img, lidar_pts, radar_pts, labels
                    )

                    collected += 1
                    total     += 1
                    pbar.update(1)
                    pbar.set_postfix({
                        "pts"  : lidar_pts.shape[0],
                        "radar": radar_pts.shape[0],
                        "objs" : len(labels),
                    })

            logger.info(
                "Done: %-12s  %d/%d frames collected",
                weather, collected, FRAMES_PER_WEATHER,
            )

        logger.info("Collection complete.  Total frames: %d", total)
        logger.info("Data root: %s", DATA_ROOT)

    finally:
        if npc is not None:
            try:
                npc.destroy_all()
            except Exception:
                pass
        setup.destroy()


if __name__ == "__main__":
    collect()
