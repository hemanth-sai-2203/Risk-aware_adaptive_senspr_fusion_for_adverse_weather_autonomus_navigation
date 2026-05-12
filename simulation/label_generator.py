"""
label_generator.py
------------------
Projects CARLA actor 3D bounding boxes into 2D image-space
bounding boxes [x1, y1, x2, y2] using the camera's intrinsic
and extrinsic matrices.

This is what makes labels actually usable for training a detection model.

Output format per object:
    {
        "class"     : "vehicle" | "pedestrian",
        "actor_id"  : int,
        "bbox_2d"   : [x1, y1, x2, y2],   # pixel coords, clipped to image
        "bbox_3d"   : [cx, cy, cz, ext_x, ext_y, ext_z, yaw],
        "distance"  : float,               # metres from ego vehicle
        "occluded"  : false                # placeholder for future use
    }

Python 3.7 | Windows | CARLA 0.9.15 | numpy 1.21.6
"""

import os
import sys
import logging

import carla
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from config import CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FOV

logger = logging.getLogger(__name__)

# ── VISIBILITY FILTERS ────────────────────────────────────────────────────────
MIN_BBOX_AREA   = 100      # pixels² — discard tiny/distant boxes
MAX_DISTANCE    = 80.0     # metres — discard objects too far to matter
MIN_BBOX_SIZE   = 10       # minimum width AND height in pixels


class LabelGenerator:
    """
    Generates 2D bounding box labels from CARLA world actors.

    Parameters
    ----------
    world           : carla.World
    ego_vehicle     : carla.Vehicle  (the data-collection vehicle)
    camera_sensor   : carla.Actor   (the RGB camera sensor)
    img_w           : int  image width  in pixels
    img_h           : int  image height in pixels
    fov             : float  camera horizontal field of view in degrees
    """

    def __init__(
        self,
        world,
        ego_vehicle,
        camera_sensor,
        img_w=CAMERA_WIDTH,
        img_h=CAMERA_HEIGHT,
        fov=CAMERA_FOV,
    ):
        self._world        = world
        self._ego          = ego_vehicle
        self._camera       = camera_sensor
        self._img_w        = img_w
        self._img_h        = img_h

        # Build camera intrinsic matrix K once
        self._K = self._build_intrinsic(img_w, img_h, fov)

    # ── PUBLIC ────────────────────────────────────────────────────────────────

    def get_labels(self):
        """
        Return a list of label dicts for all visible vehicles and pedestrians.

        Call this every tick AFTER setup.tick() so camera and world are
        on the same simulation step.

        Returns
        -------
        list of dicts (may be empty if no objects in view)
        """
        # Current camera world transform (changes every tick as vehicle moves)
        cam_transform = self._camera.get_transform()
        cam_world_matrix = self._transform_to_matrix(cam_transform)

        ego_loc   = self._ego.get_location()
        labels    = []

        for actor in self._world.get_actors():
            # Skip ego vehicle
            if actor.id == self._ego.id:
                continue

            # Filter actor types
            if "vehicle" in actor.type_id:
                cls = "vehicle"
            elif "walker.pedestrian" in actor.type_id:
                cls = "pedestrian"
            else:
                continue

            # Skip actors that are too far away
            actor_loc = actor.get_location()
            dist = ego_loc.distance(actor_loc)
            if dist > MAX_DISTANCE:
                continue

            # Get 8 corners of the 3D bounding box in world space
            bb      = actor.bounding_box
            corners = self._get_bbox_corners_world(actor, bb)

            # Project all 8 corners to image pixels
            pixels = self._project_to_image(corners, cam_world_matrix)
            if pixels is None:
                continue    # all corners behind camera

            x1, y1, x2, y2 = pixels

            # Clip to image bounds
            x1 = int(np.clip(x1, 0, self._img_w  - 1))
            y1 = int(np.clip(y1, 0, self._img_h - 1))
            x2 = int(np.clip(x2, 0, self._img_w  - 1))
            y2 = int(np.clip(y2, 0, self._img_h - 1))

            # Discard degenerate boxes
            w = x2 - x1
            h = y2 - y1
            if w < MIN_BBOX_SIZE or h < MIN_BBOX_SIZE:
                continue
            if w * h < MIN_BBOX_AREA:
                continue

            t = actor.get_transform()
            labels.append({
                "class"   : cls,
                "actor_id": actor.id,
                "bbox_2d" : [x1, y1, x2, y2],
                "bbox_3d" : [
                    round(actor_loc.x,  3),
                    round(actor_loc.y,  3),
                    round(actor_loc.z,  3),
                    round(bb.extent.x,  3),
                    round(bb.extent.y,  3),
                    round(bb.extent.z,  3),
                    round(t.rotation.yaw, 2),
                ],
                "distance": round(dist, 2),
                "occluded": False,
            })

        return labels

    # ── INTRINSIC MATRIX ─────────────────────────────────────────────────────

    @staticmethod
    def _build_intrinsic(w, h, fov_deg):
        """
        Build 3x3 camera intrinsic matrix from image dimensions and FOV.

        K = [[fx,  0, cx],
             [ 0, fy, cy],
             [ 0,  0,  1]]

        where fx = fy = (W / 2) / tan(FOV/2)
        """
        fov_rad = np.radians(fov_deg)
        fx = (w / 2.0) / np.tan(fov_rad / 2.0)
        fy = fx
        cx = w  / 2.0
        cy = h / 2.0
        K = np.array([
            [fx,  0, cx],
            [ 0, fy, cy],
            [ 0,  0,  1],
        ], dtype=np.float64)
        return K

    # ── 3D BBOX CORNERS ───────────────────────────────────────────────────────

    @staticmethod
    def _get_bbox_corners_world(actor, bb):
        """
        Return the 8 corners of an actor's bounding box in world coordinates.

        CARLA bounding box is defined in the actor's local frame.
        We rotate and translate each corner into world space.

        Returns np.ndarray (8, 3) float64
        """
        ext = bb.extent
        # 8 corners in local bounding box space
        local = np.array([
            [ ext.x,  ext.y,  ext.z],
            [ ext.x,  ext.y, -ext.z],
            [ ext.x, -ext.y,  ext.z],
            [ ext.x, -ext.y, -ext.z],
            [-ext.x,  ext.y,  ext.z],
            [-ext.x,  ext.y, -ext.z],
            [-ext.x, -ext.y,  ext.z],
            [-ext.x, -ext.y, -ext.z],
        ], dtype=np.float64)

        # Actor transform: rotation + translation
        actor_t  = actor.get_transform()
        yaw      = np.radians(actor_t.rotation.yaw)
        pitch    = np.radians(actor_t.rotation.pitch)
        roll     = np.radians(actor_t.rotation.roll)

        # Rotation matrix (CARLA uses left-handed UE4 coords)
        # Yaw around Z, Pitch around Y, Roll around X
        Rz = np.array([
            [ np.cos(yaw), -np.sin(yaw), 0],
            [ np.sin(yaw),  np.cos(yaw), 0],
            [           0,            0, 1],
        ], dtype=np.float64)
        Ry = np.array([
            [ np.cos(pitch), 0, np.sin(pitch)],
            [             0, 1,             0],
            [-np.sin(pitch), 0, np.cos(pitch)],
        ], dtype=np.float64)
        Rx = np.array([
            [1,            0,             0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll),  np.cos(roll)],
        ], dtype=np.float64)

        R   = Rz @ Ry @ Rx
        loc = actor_t.location

        # Bounding box center offset (in local actor frame)
        bb_offset = np.array([bb.location.x, bb.location.y, bb.location.z],
                             dtype=np.float64)

        world_corners = (R @ (local + bb_offset).T).T
        world_corners += np.array([loc.x, loc.y, loc.z], dtype=np.float64)

        return world_corners   # (8, 3)

    # ── PROJECTION ───────────────────────────────────────────────────────────

    @staticmethod
    def _transform_to_matrix(transform):
        """
        Convert carla.Transform → 4x4 world-to-local transformation matrix.
        """
        rot   = transform.rotation
        loc   = transform.location
        yaw   = np.radians(rot.yaw)
        pitch = np.radians(rot.pitch)
        roll  = np.radians(rot.roll)

        Rz = np.array([
            [ np.cos(yaw), -np.sin(yaw), 0, 0],
            [ np.sin(yaw),  np.cos(yaw), 0, 0],
            [           0,            0, 1, 0],
            [           0,            0, 0, 1],
        ], dtype=np.float64)
        Ry = np.array([
            [ np.cos(pitch), 0, np.sin(pitch), 0],
            [             0, 1,             0, 0],
            [-np.sin(pitch), 0, np.cos(pitch), 0],
            [             0, 0,             0, 1],
        ], dtype=np.float64)
        Rx = np.array([
            [1,            0,             0, 0],
            [0, np.cos(roll), -np.sin(roll), 0],
            [0, np.sin(roll),  np.cos(roll), 0],
            [0,            0,             0, 1],
        ], dtype=np.float64)
        T = np.array([
            [1, 0, 0, loc.x],
            [0, 1, 0, loc.y],
            [0, 0, 1, loc.z],
            [0, 0, 0,     1],
        ], dtype=np.float64)

        return T @ Rz @ Ry @ Rx

    def _project_to_image(self, world_corners, cam_world_matrix):
        """
        Project 8 world-space 3D corners onto the 2D image plane.

        Steps:
          1. Transform world coords → camera local coords
          2. Convert from CARLA/UE4 coordinate system to camera coordinate system
          3. Discard points behind the camera (z <= 0 in camera space)
          4. Apply intrinsic matrix K
          5. Return 2D bounding box [x1, y1, x2, y2]

        Returns None if all corners are behind the camera.
        """
        # Step 1: world → camera local
        # cam_world_matrix is camera-in-world transform
        # We need world-to-camera = inverse of that
        cam_inv = np.linalg.inv(cam_world_matrix)

        ones         = np.ones((world_corners.shape[0], 1), dtype=np.float64)
        corners_hom  = np.hstack([world_corners, ones])      # (8, 4)
        cam_coords   = (cam_inv @ corners_hom.T).T            # (8, 4)
        cam_xyz      = cam_coords[:, :3]                      # (8, 3)

        # Step 2: CARLA UE4 camera axes → standard camera axes
        # UE4 camera: X=forward, Y=right, Z=up
        # Standard:   X=right,   Y=down,  Z=forward
        # Transform:  [x,y,z]_std = [y, -z, x]_ue4
        std_coords = np.stack([
            cam_xyz[:, 1],    # right  = UE4 Y
            -cam_xyz[:, 2],   # down   = -UE4 Z
            cam_xyz[:, 0],    # forward= UE4 X
        ], axis=1)            # (8, 3)

        # Step 3: keep only corners in front of camera (z_forward > 0)
        in_front = std_coords[:, 2] > 0.1
        if not np.any(in_front):
            return None

        # Only project corners that are in front of camera
        visible = std_coords[in_front]   # (N, 3) where N <= 8

        # Step 4: project with intrinsic matrix
        # [u, v, 1]^T = K @ [X/Z, Y/Z, 1]^T
        u = (self._K[0, 0] * visible[:, 0] / visible[:, 2]) + self._K[0, 2]
        v = (self._K[1, 1] * visible[:, 1] / visible[:, 2]) + self._K[1, 2]

        # Step 5: 2D bounding box from extremes
        x1, x2 = float(np.min(u)), float(np.max(u))
        y1, y2 = float(np.min(v)), float(np.max(v))

        return x1, y1, x2, y2
