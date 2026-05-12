import time
import logging
import sys
import os
import carla

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import config as C
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

def spawn_npcs(world, client, ego):
    map = world.get_map()
    bp_lib = world.get_blueprint_library()
    import random
    vehicle_bps = [bp for bp in bp_lib.filter("vehicle.*") if bp.get_attribute("number_of_wheels").as_int() == 4]
    
    actors = []
    ego_loc = ego.get_location()
    
    # 1. Spawn vehicles randomly across the entire city
    spawn_points = map.get_spawn_points()
    random.shuffle(spawn_points)
    
    count = 0
    for sp in spawn_points[:C.N_VEHICLES]:
        bp = random.choice(vehicle_bps)
        v = world.try_spawn_actor(bp, sp)
        if v:
            actors.append(v)
            count += 1
            
    print(f"✅ Successfully spawned {count} random vehicles across the city.")
            
    # 2. Spawn 5 pedestrians near the ego vehicle too
    walker_bps = bp_lib.filter("walker.pedestrian.*")
    for _ in range(5):
        # Find a random location near the ego (within 50 meters)
        sp_loc = world.get_random_location_from_navigation()
        if sp_loc and sp_loc.distance(ego_loc) < 50.0:
            bp = random.choice(walker_bps)
            w = world.try_spawn_actor(bp, carla.Transform(sp_loc))
            if w:
                actors.append(w)
            
    for actor in actors:
        if hasattr(actor, "set_autopilot"):
            actor.set_autopilot(True, 8000)
    
    return actors

def draw_3d_boxes(world, detections):
    debug = world.debug
    # In Ground Truth detections, we have the actor ID.
    # We can fetch the actor and draw a 3D box.
    for det in detections:
        actor = world.get_actor(det["id"])
        if actor:
            bb = actor.bounding_box
            tf = actor.get_transform()
            color = carla.Color(0, 255, 0) if det["class"] == "vehicle" else carla.Color(255, 165, 0)
            debug.draw_box(carla.BoundingBox(tf.location, bb.extent), tf.rotation, 0.1, color, 0.1)

def run_3d_demo():
    bridge = Carlabridge()
    npc_actors = []

    try:
        print("\n🚀 Starting 3D VISUAL CARLA Demo\n")
        bridge.connect()
        bridge.spawn_vehicle()
        bridge.attach_sensors()
        npc_actors = spawn_npcs(bridge.world, bridge.client, bridge.vehicle)

        ue = UncertaintyEngine()
        tick = 0

        # Move spectator camera behind ego vehicle so faculty can watch
        spectator = bridge.world.get_spectator()
        
        weather_schedule = ["clear", "clear", "clear", "fog_light", "fog_heavy", "rain"]
        ticks_per_weather = 150

        # --- INIT YOLO MODEL ---
        from ultralytics import YOLO
        yolo_path = r"C:\Users\heman\Music\ra_asf\final\results\yolo_carla_v1-2\weights\best.pt"
        print(f"Loading YOLO Model from: {yolo_path}")
        try:
            model = YOLO(yolo_path)
            print("✅ YOLO Model Loaded Successfully!")
        except Exception as e:
            print(f"❌ Failed to load YOLO Model: {e}")
            model = None

        while tick < 1500:
            # 1. Weather Cycling
            weather_idx = tick // ticks_per_weather
            weather_name = weather_schedule[weather_idx % len(weather_schedule)]
            
            if tick % ticks_per_weather == 0:
                import carla
                if weather_name == "clear": 
                    w = carla.WeatherParameters.ClearNoon
                elif weather_name == "fog_light": 
                    w = carla.WeatherParameters.CloudyNoon
                    w.fog_density = 30.0
                elif weather_name == "fog_heavy": 
                    w = carla.WeatherParameters.CloudyNoon
                    w.fog_density = 80.0
                    w.cloudiness = 100.0
                elif weather_name == "rain": 
                    w = carla.WeatherParameters.HardRainNoon
                bridge.world.set_weather(w)

            cam_img, lidar_pts, radar_pts = bridge.tick()

            # Update spectator camera to follow the car
            ego_tf = bridge.vehicle.get_transform()
            spec_tf = carla.Transform(
                ego_tf.location + carla.Location(z=3.0) - ego_tf.get_forward_vector() * 6.0,
                carla.Rotation(pitch=-15.0, yaw=ego_tf.rotation.yaw)
            )
            spectator.set_transform(spec_tf)

            # SWAP: Use YOLO instead of Ground Truth if model loaded!
            if model is not None:
                from perception import get_yolo_detections
                detections = get_yolo_detections(cam_img, model)
            else:
                detections = get_ground_truth_detections(bridge.world, bridge.vehicle, bridge.camera)
            
            # Draw 3D bounding boxes correctly (Only works if we have Ground Truth Actor IDs)
            debug = bridge.world.debug
            for det in detections:
                # Hack to distinguish CARLA actor IDs from our YOLO hash
                if isinstance(det["id"], int) and det["id"] < 100000: 
                    actor = bridge.world.get_actor(det["id"])
                    if actor:
                        try:
                            bb = actor.bounding_box
                            tf = actor.get_transform()
                            box_center = tf.transform(bb.location)
                            if det["depth"] < 50.0:
                                color = carla.Color(0, 255, 0) if det["class"] == "vehicle" else carla.Color(255, 165, 0)
                                debug.draw_box(carla.BoundingBox(box_center, bb.extent), tf.rotation, 0.05, color, 0.1)
                        except Exception:
                            pass

            # --- YOLO OPENCV PREVIEW ---
            import cv2
            import numpy as np
            preview_img = np.copy(cam_img)
            for det in detections:
                color = (0, 255, 0) if det["class"] == "vehicle" else (0, 165, 255)
                cv2.rectangle(preview_img, (det["x1"], det["y1"]), (det["x2"], det["y2"]), color, 2)
                cv2.putText(preview_img, f"{det['class']} {det['confidence']:.2f}", (det["x1"], det["y1"] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # OpenCV expects BGR colors, but cam_img is RGB. Convert it for display:
            preview_bgr = cv2.cvtColor(preview_img, cv2.COLOR_RGB2BGR)
            cv2.imshow("YOLO AI Vision", preview_bgr)
            cv2.waitKey(1)

            lidar_centroids = cluster_lidar(lidar_pts)
            lidar_2d = project_to_image(lidar_centroids)
            radar_xyz = radar_to_cartesian(radar_pts)
            matched, unmatched = hungarian_match(detections, lidar_2d, 80)
            
            # Artificial health modulation based on weather for the demo
            health = compute_health(cam_img, lidar_pts, radar_pts)
            if weather_name == "clear": health["cam_score"] = 0.9
            elif weather_name == "fog_light": health["cam_score"] = 0.6; health["lidar_score"] *= 0.8
            elif weather_name == "fog_heavy": health["cam_score"] = 0.3; health["lidar_score"] *= 0.5
            elif weather_name == "rain": health["cam_score"] = 0.2; health["radar_score"] *= 0.7
            
            result = ue.compute(health, unmatched, len(detections))

            # Highly descriptive terminal output
            u_status = "SAFE" if result['uncertainty'] < 0.4 else ("CAUTION" if result['uncertainty'] < 0.7 else "DANGER")
            print(f"[Tick {tick:03d}] Weather: {weather_name.upper():<9} | Objects Detected: {len(detections)} | Fusion Mode: {result['mode']:<10} | Uncertainty: {result['uncertainty']:.3f} ({u_status}) | AI Target Speed: {result['target_speed']:.1f} km/h")

            # Apply speed (Slower for better YOLO observation)
            slowed_speed = float(result["target_speed"]) * 0.6 # 40% slower than original
            bridge.tm.vehicle_percentage_speed_difference(bridge.vehicle, -slowed_speed / C.MAX_SPEED * 30)

            time.sleep(0.05)
            tick += 1

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
    run_3d_demo()
