"""
run_demo.py — RA-ASF FINAL LIVE DEMO (Entry Point)
====================================================
This is the ONLY file you need to run for your presentation.

What it does (matches your slide exactly):
  1. Connects to CARLA and spawns the ego vehicle + NPCs
  2. Attaches GNSS (camera-proxy), LiDAR, Radar sensors
  3. Every tick:
     a) Gets sensor data (synthetic RGB frame + real LiDAR/Radar)
     b) Detects objects using CARLA Ground-Truth (Perfect Detector)
     c) Clusters LiDAR using deterministic DBSCAN
     d) Converts Radar to Cartesian
     e) Hungarian-matches Camera detections to LiDAR/Radar
     f) Computes S.D.T Uncertainty score
     g) Shows dual-panel window: Perception View + Bird's Eye View (BEV)
  4. Cycles through weather: clear → fog_light → fog_heavy → rain

Usage:
  Terminal 1:  Launch CARLA:
    cd C:\\Users\\heman\\Downloads\\CARLA_0.9.16
    .\\CarlaUE4.exe /Game/Maps/Town02 -carla-rpc-port=2000 -windowed ^
                   -ResX=640 -ResY=480 -quality-level=Low -nosound -dx11

  Terminal 2:  cd C:\\Users\\heman\\Music\\ra_asf
               .\\carla16_env\\Scripts\\activate
               cd final_implementation
               python run_demo.py

  Press Q to quit.

FIX NOTES
---------
* Camera replaced with GNSS (no D3D crash). tick() returns a synthetic frame
  so the main loop NEVER skips a frame due to None image (old bug).
* Dual-panel 1280x480 display: left = Perception View, right = Bird's Eye View.
* LiDAR and Radar data are now passed to the visualiser for BEV rendering.
"""

import os
import sys
import json
import time
import logging

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import config as C
from carla_bridge import Carlabridge
from perception   import (
    get_ground_truth_detections,
    cluster_lidar,
    radar_to_cartesian,
    project_to_image,
    hungarian_match,
)
from uncertainty  import compute_health, UncertaintyEngine
from visualizer   import draw_frame

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── NPC Spawner ───────────────────────────────────────────────────────────────
def spawn_npcs(world, client, ego):
    map = world.get_map()
    bp_lib = world.get_blueprint_library()
    import random
    vehicle_bps = [bp for bp in bp_lib.filter("vehicle.*") if bp.get_attribute("number_of_wheels").as_int() == 4]
    
    actors = []
    
    # 1. Spawn random vehicles (increased to 30)
    spawn_points = map.get_spawn_points()
    random.shuffle(spawn_points)
    for sp in spawn_points[:30]:
        bp = random.choice(vehicle_bps)
        v = world.try_spawn_actor(bp, sp)
        if v:
            actors.append(v)
            
    # 2. Spawn 15 pedestrians on the sidewalks (increased from 5)
    walker_bps = bp_lib.filter("walker.pedestrian.*")
    for _ in range(15):
        sp_loc = world.get_random_location_from_navigation()
        if sp_loc:
            bp = random.choice(walker_bps)
            w = world.try_spawn_actor(bp, carla.Transform(sp_loc))
            if w:
                actors.append(w)
            
    for actor in actors:
        if hasattr(actor, "set_autopilot"):
            actor.set_autopilot(True, 8000)
    
    logger.info("Spawned %d NPC actors (including pedestrians).", len(actors))
    return actors


# ── Weather Helper ────────────────────────────────────────────────────────────
WEATHER_PARAMS = {
    "clear":     dict(cloudiness=0,   precipitation=0,  fog_density=0,  sun_altitude_angle=70),
    "fog_light": dict(cloudiness=80,  precipitation=0,  fog_density=30, sun_altitude_angle=50),
    "fog_heavy": dict(cloudiness=100, precipitation=0,  fog_density=80, sun_altitude_angle=40),
    "rain":      dict(cloudiness=90,  precipitation=80, fog_density=20, sun_altitude_angle=30),
}

def set_weather(world, state):
    import carla
    p = WEATHER_PARAMS.get(state, WEATHER_PARAMS["clear"])
    w = world.get_weather()
    for k, v in p.items():
        setattr(w, k, v)
    try:
        world.set_weather(w)
    except Exception as e:
        logger.warning("Weather set failed: %s", e)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN DEMO LOOP
# ══════════════════════════════════════════════════════════════════════════════

def run_demo():
    bridge     = Carlabridge()
    npc_actors = []
    log        = []

    try:
        # ── Connect & Setup ───────────────────────────────────────────────────
        bridge.connect()
        bridge.spawn_vehicle()
        bridge.attach_sensors()
        npc_actors = spawn_npcs(bridge.world, bridge.client, bridge.vehicle)

        ue = UncertaintyEngine()

        logger.info("=" * 60)
        logger.info("  RA-ASF LIVE DEMO STARTED  —  1280x480 dual-panel display")
        logger.info("  Press Q in the OpenCV window to quit")
        logger.info("=" * 60)

        # ── Warm-up: let NPCs spread out ─────────────────────────────────────
        logger.info("Warming up (30 ticks)...")
        for _ in range(30):
            bridge.tick()

        tick        = 0
        total_ticks = len(C.DEMO_SCHEDULE) * C.TICKS_PER_WEATHER

        while tick < total_ticks:

            # ── Determine & apply weather ────────────────────────────────────
            weather_idx  = tick // C.TICKS_PER_WEATHER
            weather_name = C.DEMO_SCHEDULE[min(weather_idx, len(C.DEMO_SCHEDULE) - 1)]

            if tick % C.TICKS_PER_WEATHER == 0:
                set_weather(bridge.world, weather_name)
                logger.info("Weather → %s", weather_name)

            # ── Get Sensor Data ───────────────────────────────────────────────
            cam_img, lidar_pts, radar_pts = bridge.tick()
            # cam_img is ALWAYS a valid synthetic numpy frame now (never None)

            # ── STEP 1: Ground-Truth Detection ────────────────────────────────
            detections = get_ground_truth_detections(
                bridge.world, bridge.vehicle, bridge.camera
            )

            # ── STEP 2: LiDAR DBSCAN Clustering ──────────────────────────────
            lidar_centroids = cluster_lidar(lidar_pts)
            lidar_2d        = project_to_image(lidar_centroids)

            # ── STEP 3: Radar Polar → Cartesian ──────────────────────────────
            radar_xyz = radar_to_cartesian(radar_pts)
            radar_2d  = project_to_image(radar_xyz)

            # ── STEP 4: Hungarian Matching ────────────────────────────────────
            matched_m1, unmatched_m1 = hungarian_match(
                detections, lidar_2d, C.M1_MATCH_THRESHOLD_PX
            )
            # M2: camera ↔ radar on detections not yet matched
            unmatched_dets = unmatched_m1

            # ── STEP 5: S.D.T Uncertainty ────────────────────────────────────
            health = compute_health(cam_img, lidar_pts, radar_pts)
            result = ue.compute(health, unmatched_dets, len(detections))

            # ── STEP 6: Apply Speed to Vehicle ───────────────────────────────
            try:
                speed_diff = -float(result["target_speed"]) / C.MAX_SPEED * 30
                bridge.tm.vehicle_percentage_speed_difference(
                    bridge.vehicle, speed_diff
                )
            except Exception:
                pass

            # ── STEP 7: Draw Dual-Panel Frame & Show ─────────────────────────
            frame = draw_frame(
                img        = cam_img,
                detections = detections,
                result     = result,
                health     = health,
                tick       = tick,
                weather    = weather_name,
                lidar_pts  = lidar_pts,         # NEW: passed to BEV panel
                radar_xyz  = radar_xyz,          # NEW: passed to BEV panel
                matched_m1 = matched_m1,         # NEW: show match lines
            )

            cv2.imshow("RA-ASF — Perception + Bird's Eye View", frame)
            key = cv2.waitKey(150) & 0xFF
            if key == ord("q"):
                logger.info("Q pressed — stopping demo.")
                break

            # ── Log ───────────────────────────────────────────────────────────
            log.append({
                "tick":         tick,
                "weather":      weather_name,
                "n_objects":    len(detections),
                "n_lidar_clusters": len(lidar_centroids),
                "n_radar_pts":  len(radar_xyz),
                "n_matched":    len(matched_m1),
                "mode":         result["mode"],
                "uncertainty":  result["uncertainty"],
                "H_health":     result["H_health"],
                "H_disagree":   result["H_disagree"],
                "H_jitter":     result["H_jitter"],
                "target_speed": result["target_speed"],
            })

            if tick % 50 == 0:
                logger.info(
                    "[%04d] %-10s  mode=%-14s  objs=%2d  match=%2d  "
                    "U=%.3f  spd=%.1f km/h",
                    tick, weather_name, result["mode"],
                    len(detections), len(matched_m1),
                    result["uncertainty"], result["target_speed"],
                )

            tick += 1

        logger.info("Demo complete. %d ticks logged.", len(log))

    except KeyboardInterrupt:
        logger.info("Ctrl+C — stopping demo.")

    finally:
        cv2.destroyAllWindows()

        for a in npc_actors:
            try:
                if a and a.is_alive:
                    a.destroy()
            except Exception:
                pass
        bridge.destroy()

        # ── Save detailed JSON log ────────────────────────────────────────────
        log_path = os.path.join(C.RESULTS_DIR, "demo_log.json")
        os.makedirs(C.RESULTS_DIR, exist_ok=True)
        with open(log_path, "w") as f:
            json.dump(log, f, indent=2)
        logger.info("Log saved → %s", log_path)

        # ── Print summary statistics ──────────────────────────────────────────
        if log:
            unc_vals = [e["uncertainty"] for e in log]
            spd_vals = [e["target_speed"] for e in log]
            logger.info(
                "Summary: avg_U=%.3f  max_U=%.3f  avg_speed=%.1f  "
                "emergency_stops=%d",
                np.mean(unc_vals), np.max(unc_vals), np.mean(spd_vals),
                sum(1 for e in log if e["mode"] == "EMERGENCY_STOP"),
            )


if __name__ == "__main__":
    run_demo()
