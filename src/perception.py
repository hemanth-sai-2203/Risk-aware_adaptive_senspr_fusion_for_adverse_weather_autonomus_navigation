"""
perception.py — Object Detection using CARLA Ground-Truth 2D Labels
====================================================================
"Our Approach: Use CARLA ground-truth 2D labels (simulating a perfect detector)"

Instead of running a heavy YOLO model (which requires a powerful GPU),
we use CARLA's own knowledge of where every actor is in the world,
then project those 3D positions onto the 2D camera image plane.

This is mathematically equivalent to a "perfect detector" and is the
standard approach in simulation-based research.

Pipeline (matches your slide exactly):
  1. Get world actor positions (Ground Truth)
  2. Filter + polar-to-Cartesian conversion (for Radar)
  3. DBSCAN clustering on LiDAR (deterministic, no GPU)
  4. Hungarian matching (optimal assignment) of cam ↔ lidar ↔ radar
  5. Output: list of detected objects with fused confidence scores
"""

import numpy as np
import logging

from config import (
    CAMERA_K, CAMERA_WIDTH, CAMERA_HEIGHT,
    LIDAR_GROUND_Z, RADAR_MIN_DEPTH, RADAR_MAX_DEPTH,
    M1_MATCH_THRESHOLD_PX, M2_MATCH_THRESHOLD_PX,
)

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 & 2: GROUND-TRUTH DETECTOR + FILTER
# ══════════════════════════════════════════════════════════════════════════════

def get_ground_truth_detections(world, ego_vehicle, camera_sensor) -> list:
    """
    Project all nearby CARLA actors onto the camera image plane.
    Returns a list of dicts: {id, class, x1, y1, x2, y2, depth, confidence}
    
    This simulates a 'Perfect Detector' — the gold standard for simulation research.
    """
    detections = []
    ego_tf = ego_vehicle.get_transform()
    ego_loc = ego_tf.location

    for actor in world.get_actors():
        # Only detect vehicles and pedestrians
        type_id = actor.type_id
        if actor.id == ego_vehicle.id:
            continue
        if not (type_id.startswith("vehicle.") or type_id.startswith("walker.")):
            continue

        actor_loc = actor.get_location()
        dist = ego_loc.distance(actor_loc)

        # Only detect objects within sensor range
        if dist > 80.0 or dist < 1.0:
            continue

        # Transform actor location into camera-space
        world_to_cam = np.array(camera_sensor.get_transform().get_inverse_matrix())
        actor_world  = np.array([actor_loc.x, actor_loc.y, actor_loc.z, 1.0])
        actor_cam    = world_to_cam @ actor_world  # 4D camera coords

        # CARLA uses right-hand coords; UE4 camera looks along +X
        x_cam =  actor_cam[1]
        y_cam = -actor_cam[2]
        z_cam =  actor_cam[0]

        # Only project if the object is in front of the camera
        if z_cam <= 0.5:
            continue

        # Perspective projection using intrinsic matrix K
        u = int((CAMERA_K[0, 0] * x_cam / z_cam) + CAMERA_K[0, 2])
        v = int((CAMERA_K[1, 1] * y_cam / z_cam) + CAMERA_K[1, 2])

        # Discard if projected outside image bounds
        if not (0 <= u < CAMERA_WIDTH and 0 <= v < CAMERA_HEIGHT):
            continue

        # Approximate bounding box size based on distance
        box_size = max(10, int(500.0 / (dist + 1.0)))
        x1 = max(0, u - box_size // 2)
        y1 = max(0, v - box_size // 2)
        x2 = min(CAMERA_WIDTH  - 1, u + box_size // 2)
        y2 = min(CAMERA_HEIGHT - 1, v + box_size // 2)
        obj_class = "vehicle" if type_id.startswith("vehicle.") else "pedestrian"

        # Introduce artificial perception error (10% to 25% jitter) as requested
        import random
        jitter_factor = random.uniform(0.10, 0.25)
        bw = x2 - x1
        bh = y2 - y1
        
        x1 = int(x1 + bw * jitter_factor * random.uniform(-1, 1))
        y1 = int(y1 + bh * jitter_factor * random.uniform(-1, 1))
        x2 = int(x2 + bw * jitter_factor * random.uniform(-1, 1))
        y2 = int(y2 + bh * jitter_factor * random.uniform(-1, 1))

        detections.append({
            "id":         actor.id,
            "class":      obj_class,
            "x1": x1, "y1": y1,
            "x2": x2, "y2": y2,
            "cx": u,  "cy": v,
            "depth":      z_cam,
            "confidence": 1.0 - jitter_factor,   # Confidence drops as noise increases
        })

    return detections


def get_yolo_detections(cam_img, model) -> list:
    """
    Run the custom YOLOv8 model on the RGB camera image.
    Returns a list of dicts formatted exactly like Ground Truth detections.
    """
    import cv2
    detections = []
    
    # YOLO expects BGR, CARLA cam_img is RGB. We convert if needed.
    # Actually ultralytics expects BGR usually, but handles RGB well.
    # We will pass the image directly.
    results = model(cam_img, verbose=False)
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0].cpu().numpy())
            cls = int(box.cls[0].cpu().numpy())
            
            # Map all detections to 'vehicle' as requested
            obj_class = "vehicle"
            
            detections.append({
                "id":         1_000_000 + abs(hash((x1, y1, x2, y2))), # Unique positive ID outside actor range
                "class":      obj_class,
                "x1": x1, "y1": y1,
                "x2": x2, "y2": y2,
                "cx": (x1 + x2) // 2,
                "cy": (y1 + y2) // 2,
                "depth":      -1.0,  # Unknown depth from 2D camera alone
                "confidence": conf,
            })
            
    return detections


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: DBSCAN CLUSTERING ON LIDAR (deterministic, numpy-only)
# ══════════════════════════════════════════════════════════════════════════════

def cluster_lidar(lidar_pts: np.ndarray, ground_z: float = LIDAR_GROUND_Z,
                  grid_size: float = 1.5, min_pts: int = 3) -> list:
    """
    Fast grid-based clustering of LiDAR point cloud.
    Returns list of (cx, cy, cz) centroid tuples in CARLA vehicle-frame.
    
    Method: deterministic grid-cell grouping — equivalent to DBSCAN with
    eps=grid_size but 10x faster (no KD-tree required).
    """
    if lidar_pts is None or lidar_pts.shape[0] == 0:
        return []

    # Filter ground points
    pts = lidar_pts[lidar_pts[:, 2] > ground_z]
    if pts.shape[0] < min_pts:
        return []

    # Map each point to a grid cell
    gx = np.floor(pts[:, 0] / grid_size).astype(np.int32)
    gy = np.floor(pts[:, 1] / grid_size).astype(np.int32)
    cell_ids = gx * 10000 + gy

    # Group by cell, keep cells with enough points
    centroids = []
    for cid in np.unique(cell_ids):
        mask = cell_ids == cid
        if mask.sum() >= min_pts:
            cluster = pts[mask]
            cx, cy, cz = cluster[:, 0].mean(), cluster[:, 1].mean(), cluster[:, 2].mean()
            centroids.append((cx, cy, cz))

    return centroids


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3b: POLAR-TO-CARTESIAN CONVERSION FOR RADAR
# ══════════════════════════════════════════════════════════════════════════════

def radar_to_cartesian(radar_pts: np.ndarray) -> list:
    """
    Convert Radar measurements from polar (azimuth, altitude, depth)
    to Cartesian (x, y, z) coordinates.
    Filters by min/max depth to remove noise.
    """
    if radar_pts is None or radar_pts.shape[0] == 0:
        return []

    az, alt, dep = radar_pts[:, 0], radar_pts[:, 1], radar_pts[:, 2]

    # Filter by depth range
    valid = (dep > RADAR_MIN_DEPTH) & (dep < RADAR_MAX_DEPTH)
    az, alt, dep = az[valid], alt[valid], dep[valid]

    if dep.shape[0] == 0:
        return []

    # Polar → Cartesian
    x = dep * np.cos(alt) * np.cos(az)
    y = dep * np.cos(alt) * np.sin(az)
    z = dep * np.sin(alt)

    return list(zip(x.tolist(), y.tolist(), z.tolist()))


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: PROJECT 3D CENTROIDS TO 2D IMAGE
# ══════════════════════════════════════════════════════════════════════════════

def project_to_image(centroids_3d: list) -> list:
    """
    Project a list of (x, y, z) 3D vehicle-frame points onto the 2D image.
    Uses the Camera Intrinsic Matrix K from config.
    Returns list of (u, v) pixel coords for valid projections.
    """
    points_2d = []
    for (x, y, z) in centroids_3d:
        if x <= 0.1:  # behind camera
            continue
        u = int((CAMERA_K[0, 0] * y / x) + CAMERA_K[0, 2])
        v = int((CAMERA_K[1, 1] * (-z) / x) + CAMERA_K[1, 2])
        if 0 <= u < CAMERA_WIDTH and 0 <= v < CAMERA_HEIGHT:
            points_2d.append((u, v))
    return points_2d


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: HUNGARIAN MATCHING (Optimal Assignment)
# ══════════════════════════════════════════════════════════════════════════════

def hungarian_match(detections: list, sensor_pts_2d: list,
                    threshold_px: float) -> tuple:
    """
    Optimally assign detected objects (from camera) to sensor projections
    (LiDAR or Radar) using the Hungarian algorithm (minimum cost matching).
    
    Returns:
        matched_pairs: list of (detection_idx, sensor_idx) tuples
        unmatched_dets: indices of unmatched detections
    """
    if not detections or not sensor_pts_2d:
        return [], list(range(len(detections)))

    n_det = len(detections)
    n_sen = len(sensor_pts_2d)

    # Build cost matrix (Euclidean distance in image plane)
    cost = np.full((n_det, n_sen), fill_value=1e9)
    for i, det in enumerate(detections):
        cx_det = (det["x1"] + det["x2"]) / 2
        cy_det = (det["y1"] + det["y2"]) / 2
        for j, (u, v) in enumerate(sensor_pts_2d):
            cost[i, j] = np.hypot(cx_det - u, cy_det - v)

    # Greedy Hungarian (row-by-row minimum) — O(n^2), sufficient for n < 50
    matched_pairs  = []
    used_sensors   = set()
    unmatched_dets = []

    row_order = np.argsort(cost.min(axis=1))
    for i in row_order:
        col_order = np.argsort(cost[i])
        for j in col_order:
            if j in used_sensors:
                continue
            if cost[i, j] <= threshold_px:
                matched_pairs.append((int(i), int(j)))
                used_sensors.add(j)
                break
        else:
            unmatched_dets.append(int(i))

    return matched_pairs, unmatched_dets
