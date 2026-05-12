"""
visualizer.py — Dual-Panel Real-time Visualiser for RA-ASF Demo
================================================================
Renders a 1280 x 480 frame split into two panels:

  LEFT  (640×480) — PERCEPTION VIEW
    • Synthetic camera background (dark blue road-scene gradient)
    • Ground-truth bounding boxes (green = vehicle, orange = pedestrian)
    • HUD: fusion mode, weather, object count, speed, uncertainty bar
    • Sensor health bars (CAM / LiDAR / Radar)
    • Match lines showing Camera ↔ LiDAR corroboration

  RIGHT (640×480) — BIRD'S EYE VIEW (BEV)
    • Dark background with concentric range rings (10/20/40/60/80 m)
    • Cyan LiDAR point cloud (above ground)
    • Orange triangles for Radar returns
    • Green/orange rectangles for detected objects (projected depth)
    • White ego-vehicle rectangle with forward arrow
    • Legend + panel label

Only OpenCV is used — no matplotlib, no Qt, no extra GPU work.
"""

import cv2
import numpy as np

from config import CAMERA_K, CAMERA_WIDTH, CAMERA_HEIGHT

# ── Colour Palette (BGR) ──────────────────────────────────────────────────────
GREEN     = (0, 220, 0)
ORANGE    = (0, 165, 255)
RED       = (0, 0, 220)
BLUE      = (220, 100, 0)
WHITE     = (255, 255, 255)
BLACK     = (0, 0, 0)
CYAN      = (255, 200, 0)
YELLOW    = (0, 220, 220)
DARK_GRAY = (45, 45, 45)
GRAY      = (100, 100, 100)
LIDAR_COL = (200, 200, 0)    # cyan-ish
RADAR_COL = (0, 120, 255)    # bright orange
MATCH_COL = (180, 255, 100)  # lime green for match lines

MODE_COLORS = {
    "GOLD":           GREEN,
    "M1":             CYAN,
    "M2":             ORANGE,
    "DEGRADED":       RED,
    "EMERGENCY_STOP": RED,
}

PANEL_W = 640
PANEL_H = 480


# ═════════════════════════════════════════════════════════════════════════════
# LEFT PANEL — PERCEPTION VIEW
# ═════════════════════════════════════════════════════════════════════════════

def _camera_panel(img, detections, result, health, tick, weather, matched_m1):
    """
    Build the left 640×480 perception panel.
    img is the synthetic (H, W, 3) RGB frame from carla_bridge.
    """
    # ── Scale up from 160×120 → 640×480 ─────────────────────────────────────
    frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    frame = cv2.resize(frame, (PANEL_W, PANEL_H), interpolation=cv2.INTER_NEAREST)

    sx = PANEL_W / CAMERA_WIDTH
    sy = PANEL_H / CAMERA_HEIGHT

    # ── Bounding boxes ───────────────────────────────────────────────────────
    for det in detections:
        x1 = int(det["x1"] * sx);  y1 = int(det["y1"] * sy)
        x2 = int(det["x2"] * sx);  y2 = int(det["y2"] * sy)
        color = GREEN if det["class"] == "vehicle" else ORANGE
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Depth label inside box
        depth_txt = f"{det['depth']:.1f}m"
        cv2.putText(frame, depth_txt,
                    (x1 + 2, min(y2 - 4, PANEL_H - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, color, 1, cv2.LINE_AA)

        label = f"{'VEH' if det['class']=='vehicle' else 'PED'} {det['confidence']:.2f}"
        cv2.putText(frame, label, (x1, max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1, cv2.LINE_AA)

    # ── HUD — semi-transparent left panel ────────────────────────────────────
    overlay = frame.copy()
    cv2.rectangle(overlay, (4, 4), (278, 198), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    mode        = result["mode"]
    unc         = result["uncertainty"]
    speed       = result["target_speed"]
    mode_color  = MODE_COLORS.get(mode, WHITE)

    def txt(text, y, color=WHITE, scale=0.50, bold=1):
        cv2.putText(frame, text, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, bold, cv2.LINE_AA)

    txt("RA-ASF  Live Demo", 26, WHITE,      0.60, 2)
    txt(f"Mode   : {mode}",   50, mode_color, 0.50)
    txt(f"Weather: {weather}", 68, CYAN,       0.50)
    txt(f"Objects: {len(detections)} detected", 86, WHITE, 0.50)
    txt(f"Matched: {len(matched_m1 or [])} cam↔lidar", 104, MATCH_COL, 0.48)
    txt(f"Speed  : {speed:.1f} km/h",          122, YELLOW, 0.50)
    txt(f"Tick   : {tick}",                     140, GRAY,   0.42)

    # Uncertainty progress bar
    txt("Uncertainty:", 164, WHITE, 0.44)
    bar_col = GREEN if unc < 0.40 else (ORANGE if unc < 0.75 else RED)
    bw = int(200 * unc)
    cv2.rectangle(frame, (10, 171), (210, 186), DARK_GRAY, -1)
    if bw > 0:
        cv2.rectangle(frame, (10, 171), (10 + bw, 186), bar_col, -1)
    cv2.rectangle(frame, (10, 171), (210, 186), GRAY, 1)
    txt(f"{unc:.3f}", 198, bar_col, 0.44)

    # ── Sensor health bars (bottom-left) ─────────────────────────────────────
    sensors = [
        ("CAM",   health["cam_score"],   GREEN),
        ("LiDAR", health["lidar_score"], CYAN),
        ("Radar", health["radar_score"], ORANGE),
    ]
    y0 = PANEL_H - 58
    for lbl, score, col in sensors:
        bw2 = int(90 * score)
        cv2.rectangle(frame, (5, y0), (95, y0 + 11), DARK_GRAY, -1)
        if bw2 > 0:
            cv2.rectangle(frame, (5, y0), (5 + bw2, y0 + 11), col, -1)
        cv2.rectangle(frame, (5, y0), (95, y0 + 11), GRAY, 1)
        cv2.putText(frame, f"{lbl} {score:.2f}", (100, y0 + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, col, 1, cv2.LINE_AA)
        y0 += 18

    # Panel label (bottom-right)
    cv2.putText(frame, "PERCEPTION VIEW",
                (PANEL_W - 168, PANEL_H - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (70, 70, 70), 1, cv2.LINE_AA)

    return frame


# ═════════════════════════════════════════════════════════════════════════════
# RIGHT PANEL — BIRD'S EYE VIEW
# ═════════════════════════════════════════════════════════════════════════════

def _bev_panel(lidar_pts, radar_xyz, detections, result):
    """
    Build the right 640×480 BEV panel showing:
      • LiDAR point cloud (cyan dots)
      • Radar returns (orange triangles)
      • Detected objects (coloured rectangles projected at estimated depth)
      • Ego vehicle (white box + forward arrow)
    """
    panel = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
    panel[:] = (14, 14, 20)   # very dark background

    cx  = PANEL_W // 2
    cy  = int(PANEL_H * 0.65)   # ego sits at 65 % down so we see more forward
    PX_PER_M = 5.5              # scale: pixels per metre

    # ── Range rings ──────────────────────────────────────────────────────────
    for r_m in [10, 20, 40, 60, 80]:
        r_px = int(r_m * PX_PER_M)
        cv2.circle(panel, (cx, cy), r_px, (32, 32, 32), 1, cv2.LINE_AA)
        lbl_x = cx + r_px + 3
        if lbl_x < PANEL_W - 20:
            cv2.putText(panel, f"{r_m}m", (lbl_x, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, (55, 55, 55), 1)

    # Cardinal grid lines
    cv2.line(panel, (cx, 0), (cx, PANEL_H), (28, 28, 28), 1)
    cv2.line(panel, (0, cy), (PANEL_W, cy), (28, 28, 28), 1)
    cv2.putText(panel, "FWD", (cx - 14, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (60, 60, 60), 1)

    # ── LiDAR point cloud ─────────────────────────────────────────────────────
    if lidar_pts is not None and lidar_pts.shape[0] > 0:
        # Filter ground plane; keep points above -1.5 m
        above_ground = lidar_pts[lidar_pts[:, 2] > -1.5]
        if above_ground.shape[0] > 0:
            # CARLA LiDAR frame: x=forward, y=right, z=up
            px_arr = (cx + above_ground[:, 1] * PX_PER_M).astype(np.int32)
            py_arr = (cy - above_ground[:, 0] * PX_PER_M).astype(np.int32)
            in_bounds = (
                (px_arr >= 0) & (px_arr < PANEL_W) &
                (py_arr >= 0) & (py_arr < PANEL_H)
            )
            for px_pt, py_pt in zip(px_arr[in_bounds], py_arr[in_bounds]):
                cv2.circle(panel, (int(px_pt), int(py_pt)), 1, LIDAR_COL, -1)

    # ── Radar returns ─────────────────────────────────────────────────────────
    if radar_xyz:
        for (rx, ry, rz) in radar_xyz:
            px_r = int(cx + ry * PX_PER_M)
            py_r = int(cy - rx * PX_PER_M)
            if 0 <= px_r < PANEL_W and 0 <= py_r < PANEL_H:
                tri = np.array([
                    [px_r,     py_r - 7],
                    [px_r - 5, py_r + 5],
                    [px_r + 5, py_r + 5],
                ], dtype=np.int32)
                cv2.fillPoly(panel, [tri], RADAR_COL)

    # ── Detected objects (approximate BEV position from depth + camera cx) ───
    # We back-project using depth and the horizontal pixel offset from centre.
    for det in detections:
        depth   = max(1.0, det.get("depth", 20.0))
        cx_img  = det.get("cx", CAMERA_WIDTH // 2)

        # Lateral offset via thin-lens approximation
        # offset_m = (cx_img - cx_img_centre) / fx * depth
        fx        = CAMERA_K[0, 0]
        lat_m     = (cx_img - CAMERA_WIDTH / 2.0) / fx * depth

        bev_x = int(cx + lat_m * PX_PER_M)
        bev_y = int(cy - depth * PX_PER_M)

        if 0 <= bev_x < PANEL_W and 0 <= bev_y < PANEL_H:
            col = (0, 210, 0) if det["class"] == "vehicle" else (0, 140, 255)
            bw_box, bh_box = 10, 16
            cv2.rectangle(panel,
                          (bev_x - bw_box, bev_y - bh_box),
                          (bev_x + bw_box, bev_y + bh_box),
                          col, 2)
            cv2.putText(panel, det["class"][0].upper(),
                        (bev_x - 4, bev_y + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, col, 1)

    # ── Ego vehicle ───────────────────────────────────────────────────────────
    ew, eh = 10, 18
    cv2.rectangle(panel,
                  (cx - ew, cy - eh), (cx + ew, cy + eh),
                  WHITE, -1)
    cv2.arrowedLine(panel,
                    (cx, cy - eh), (cx, cy - eh - 18),
                    WHITE, 2, tipLength=0.45)
    cv2.putText(panel, "EGO", (cx - 12, cy + eh + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, WHITE, 1)

    # ── Fusion mode badge (top-right) ─────────────────────────────────────────
    mode       = result["mode"]
    mode_color = MODE_COLORS.get(mode, WHITE)
    unc        = result["uncertainty"]

    # Badge background
    cv2.rectangle(panel, (PANEL_W - 200, 6), (PANEL_W - 6, 48), (0, 0, 0), -1)
    cv2.rectangle(panel, (PANEL_W - 200, 6), (PANEL_W - 6, 48), mode_color, 1)
    cv2.putText(panel, f"FUSION: {mode}",
                (PANEL_W - 194, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, mode_color, 1, cv2.LINE_AA)
    cv2.putText(panel, f"U = {unc:.3f}",
                (PANEL_W - 194, 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, mode_color, 1, cv2.LINE_AA)

    # ── Legend (bottom-left) ──────────────────────────────────────────────────
    ly = PANEL_H - 56
    cv2.circle(panel, (12, ly + 4), 3, LIDAR_COL, -1)
    cv2.putText(panel, "LiDAR", (22, ly + 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, LIDAR_COL, 1)

    tri2 = np.array([[12, ly+14], [8, ly+22], [16, ly+22]], dtype=np.int32)
    cv2.fillPoly(panel, [tri2], RADAR_COL)
    cv2.putText(panel, "Radar", (22, ly + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, RADAR_COL, 1)

    cv2.rectangle(panel, (8, ly+28), (16, ly+36), (0, 200, 0), 2)
    cv2.putText(panel, "Vehicle", (22, ly + 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 200, 0), 1)

    # Panel label
    cv2.putText(panel, "BIRD'S EYE VIEW  (BEV)",
                (PANEL_W // 2 - 100, PANEL_H - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (60, 60, 60), 1, cv2.LINE_AA)

    return panel


# ═════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═════════════════════════════════════════════════════════════════════════════

def draw_frame(img: np.ndarray,
               detections: list,
               result: dict,
               health: dict,
               tick: int,
               weather: str,
               lidar_pts: np.ndarray = None,
               radar_xyz: list       = None,
               matched_m1: list      = None) -> np.ndarray:
    """
    Compose the full 1280×480 dual-panel frame.

    Args:
        img        : (H, W, 3) uint8 RGB synthetic camera frame
        detections : list of detection dicts from perception.py
        result     : dict from UncertaintyEngine.compute()
        health     : dict from compute_health()
        tick       : current simulation tick
        weather    : weather state string
        lidar_pts  : (N, 4) float32 LiDAR array [x,y,z,intensity] or None
        radar_xyz  : list of (x,y,z) tuples (Cartesian) or None
        matched_m1 : list of (det_idx, lidar_idx) matched pairs or None

    Returns:
        1280×480 BGR frame suitable for cv2.imshow()
    """
    left  = _camera_panel(img, detections, result, health,
                          tick, weather, matched_m1 or [])
    right = _bev_panel(lidar_pts, radar_xyz or [], detections, result)

    combined = np.hstack([left, right])

    # Divider line
    cv2.line(combined, (PANEL_W, 0), (PANEL_W, PANEL_H), (70, 70, 70), 2)

    return combined
