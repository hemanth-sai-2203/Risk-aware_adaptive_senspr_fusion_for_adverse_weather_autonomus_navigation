# safe_demo.py  (NO CRASH VERSION)

import time
import logging
import sys
import os

# Ensure config and other modules are in path
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from carla_bridge import Carlabridge
from perception import (
    get_ground_truth_detections,
    cluster_lidar,
    radar_to_cartesian,
    project_to_image,
    hungarian_match,
)
from uncertainty import compute_health, UncertaintyEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


import config as C

def spawn_npcs(world, client, ego):
    map = world.get_map()
    ego_loc = ego.get_location()
    ego_wp = map.get_waypoint(ego_loc)
    
    bp_lib = world.get_blueprint_library()
    import random
    vehicle_blueprints = [bp for bp in bp_lib.filter("vehicle.*")
                          if bp.get_attribute("number_of_wheels").as_int() == 4]
    
    actors = []
    
    # 1. Force spawn 3 vehicles directly in front of the Ego vehicle along the lane
    current_wp = ego_wp
    for i in range(3):
        next_wps = current_wp.next(15.0)  # 15 meters ahead
        if next_wps:
            current_wp = next_wps[0]
            bp = random.choice(vehicle_blueprints)
            spawn_tf = current_wp.transform
            spawn_tf.location.z += 0.5  # prevent ground collision
            v = world.try_spawn_actor(bp, spawn_tf)
            if v:
                actors.append(v)
                
    # 2. Spawn a few more randomly
    spawn_points = map.get_spawn_points()
    random.shuffle(spawn_points)
    for sp in spawn_points[:5]:
        bp = random.choice(vehicle_blueprints)
        v = world.try_spawn_actor(bp, sp)
        if v:
            actors.append(v)
            
    # Enable autopilot
    for v in actors:
        v.set_autopilot(True, 8000)
    
    logger.info("Spawned %d NPC vehicles (guaranteed in front).", len(actors))
    return actors

def run_safe_demo():
    bridge = Carlabridge()
    npc_actors = []

    try:
        print("\n🚀 Starting SAFE CARLA Demo (No Rendering)\n")

        bridge.connect()
        bridge.spawn_vehicle()
        bridge.attach_sensors()
        npc_actors = spawn_npcs(bridge.world, bridge.client, bridge.vehicle)

        ue = UncertaintyEngine()

        print("✅ Connected to CARLA")
        print("✅ Sensors attached")
        print("✅ Simulation running...\n")

        tick = 0

        while tick < 300:   # run 300 steps for demo (approx 30 seconds)

            cam_img, lidar_pts, radar_pts = bridge.tick()

            # In -nullrhi mode, cam_img might be None or empty. We ignore it.
            # We still use the ground truth loop.

            # STEP 1: Detection
            detections = get_ground_truth_detections(
                bridge.world, bridge.vehicle, bridge.camera
            )

            # STEP 2: LiDAR clustering
            lidar_centroids = cluster_lidar(lidar_pts)
            lidar_2d = project_to_image(lidar_centroids)

            # STEP 3: Radar
            radar_xyz = radar_to_cartesian(radar_pts)
            radar_2d = project_to_image(radar_xyz)

            # STEP 4: Matching
            matched, unmatched = hungarian_match(
                detections, lidar_2d, 80
            )

            # STEP 5: Uncertainty
            # Note: Even without rendering, we pass the image data we get.
            # If the image is empty due to -nullrhi, the blur score gracefully defaults to 0.
            health = compute_health(cam_img, lidar_pts, radar_pts)
            result = ue.compute(health, unmatched, len(detections))

            # 🔥 PRINT OUTPUT (THIS IS YOUR DEMO)
            print(f"Tick {tick:03d} | Objects: {len(detections)} | "
                  f"Mode: {result['mode']:<8} | "
                  f"Uncertainty: {result['uncertainty']:.4f} | "
                  f"Speed: {result['target_speed']:.1f} km/h")

            time.sleep(0.1)
            tick += 1

        print("\n✅ Demo completed successfully!")

    except Exception as e:
        print("❌ Error:", e)

    finally:
        for a in npc_actors:
            try:
                if a and a.is_alive:
                    a.destroy()
            except Exception:
                pass
        bridge.destroy()


if __name__ == "__main__":
    run_safe_demo()
