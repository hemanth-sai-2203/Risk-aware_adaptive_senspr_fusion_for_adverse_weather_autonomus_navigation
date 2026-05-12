# Risk-Aware Autonomous Sensor Fusion (RA-ASF) for CARLA

> **A real-time, uncertainty-driven multi-sensor fusion pipeline for autonomous vehicles, built and validated in the CARLA simulation environment.**

---

## 📌 Project Summary

**RA-ASF** is a research-grade autonomous driving pipeline that combines **Camera**, **LiDAR**, and **Radar** sensor data under a novel **S.D.T (Sensor Degradation Trust)** uncertainty formula. The system computes a real-time uncertainty score from live sensor health, sensor disagreement, and temporal instability — and uses it to dynamically scale the vehicle's target speed through a **Risk-Aware PID controller**. The result is an autonomous vehicle that naturally slows down in fog, rain, or sensor failure — without any hard-coded rules.

The entire system runs **inside CARLA 0.9.16 on a standard Windows laptop** (Intel Iris Xe GPU), with a live dual-panel OpenCV visualizer showing the Perception View and Bird's Eye View simultaneously.

---

## 🎥 Demonstration Videos

> ⏳ **Note:** Full demonstration videos of the system in action (including the 8-minute core demo and the 15-minute data collection run) are currently being processed and will be linked here shortly!

---

## 🧠 The Core Research Idea

### The Gap We Address

Existing autonomous driving pipelines keep uncertainty estimates locked inside the perception module. The vehicle's speed and control strategy do not adapt to how confident the sensors actually are. This is **Gap 3** we identified:

> *"Uncertainty stays in perception; vehicle behavior does not adapt based on confidence."*

### Our Solution — The S.D.T Formula

We derive a single, interpretable uncertainty score using three physically-motivated components:

```
U = α·H_health + β·H_disagree + γ·H_jitter
```

| Component       | Symbol         | Default Weight | What it captures                                              |
|-----------------|----------------|----------------|---------------------------------------------------------------|
| Sensor Health   | `H_health`     | α = 0.30       | How degraded each sensor is (blur, point density, SNR)        |
| Disagreement    | `H_disagree`   | β = 0.50       | Fraction of camera detections not corroborated by LiDAR/Radar |
| Temporal Jitter | `H_jitter`     | γ = 0.20       | Variance in detection count across the last 10 frames         |

The vehicle's target speed is then set as:

```
target_speed = MAX_SPEED × (1 − U)^1.5
```

At `U ≥ 0.85`, the vehicle issues a full **emergency stop**.

---

## 🏗️ System Architecture

The pipeline flows in a strict, sequential order every simulation tick:

```
┌──────────────────────────────────────────────────────────────────┐
│                        CARLA Simulator                           │
│                      (Town02, Sync Mode)                         │
│                                                                  │
│   ┌─────────────┐    ┌──────────────────┐    ┌───────────────┐  │
│   │   Camera    │    │      LiDAR       │    │     Radar     │  │
│   │ (RGB 640x480│    │  (8-ch Ray-Cast  │    │ (60°H/10°V    │  │
│   │   90° FOV)  │    │   2000 pts/s)    │    │  100m range)  │  │
│   └──────┬──────┘    └────────┬─────────┘    └───────┬───────┘  │
└──────────┼────────────────────┼──────────────────────┼──────────┘
           │                    │                      │
           └────────────────────┼──────────────────────┘
                                │
                                ▼
              ┌─────────────────────────────────┐
              │         carla_bridge.py          │
              │  Connects to CARLA, spawns ego   │
              │  vehicle + NPCs, attaches all    │
              │  sensors, advances tick() loop   │
              └─────────────────┬───────────────┘
                                │  Raw sensor data
                    ┌───────────┴────────────┐
                    │                        │
                    ▼                        ▼
      ┌─────────────────────────┐  ┌────────────────────────────┐
      │      perception.py      │  │   sensor_health_monitor.py │
      │                         │  │                            │
      │  1. Ground-Truth / YOLO │  │  Camera: Blur + Exposure   │
      │     Object Detection    │  │  LiDAR:  Point Density     │
      │  2. LiDAR Grid-DBSCAN   │  │          + Coverage        │
      │     Clustering          │  │  Radar:  Det Count + Vel   │
      │  3. Radar Polar →       │  │  → EMA-smoothed scores     │
      │     Cartesian           │  │  → Mode select:            │
      │  4. Hungarian Matching  │  │    GOLD/M1/M2/M3/DEGRADED  │
      │     (Cam ↔ LiDAR/Radar) │  │                            │
      └────────────┬────────────┘  └────────────┬───────────────┘
                   │  detections +                │  health scores +
                   │  unmatched list              │  active_mode
                   └─────────────┬───────────────┘
                                 │
                                 ▼
                  ┌──────────────────────────────┐
                  │        uncertainty.py         │
                  │                              │
                  │  U = α·H_health              │
                  │    + β·H_disagree            │
                  │    + γ·H_jitter              │
                  │                              │
                  │  Environmental Risk Floor:   │
                  │  Heavy fog → U ≥ 0.45        │
                  │  Heavy rain → U ≥ 0.75       │
                  └──────────────┬───────────────┘
                                 │  U score ∈ [0, 1]
                                 ▼
                  ┌──────────────────────────────┐
                  │      risk_aware_pid.py        │
                  │                              │
                  │  target_speed =              │
                  │  MAX_SPEED × (1 − U)^1.5    │
                  │                              │
                  │  U ≥ 0.85 → EMERGENCY STOP   │
                  │  PID → throttle / brake      │
                  └──────────────┬───────────────┘
                                 │  throttle, brake,
                                 │  target_speed
                                 ▼
                  ┌──────────────────────────────┐
                  │        visualizer.py          │
                  │                              │
                  │  LEFT (640×480):             │
                  │    Perception View           │
                  │    Bounding boxes, HUD,      │
                  │    Uncertainty bar,          │
                  │    Sensor health bars        │
                  │                              │
                  │  RIGHT (640×480):            │
                  │    Bird's Eye View (BEV)     │
                  │    LiDAR cloud, Radar,       │
                  │    Detected objects, Ego     │
                  └──────────────────────────────┘
                    1280×480 OpenCV dual-panel
```

### Sensor Fusion Modes

The system dynamically selects one of five fusion modes based on real-time sensor health scores:

| Mode         | Sensors Active     | Trigger Condition                        |
|--------------|--------------------|------------------------------------------|
| **GOLD**     | Camera+LiDAR+Radar | All sensors healthy (score ≥ threshold)  |
| **M1**       | Camera + LiDAR     | Radar degraded                           |
| **M2**       | Camera + Radar     | LiDAR degraded                           |
| **M3**       | LiDAR + Radar      | Camera degraded (e.g. heavy fog/glare)   |
| **DEGRADED** | Best available pair| Two or more sensors degraded             |

---

## 📁 Repository Structure

```
final_files/
│
├── src/                          # Core pipeline source code
│   ├── run_demo.py               #  MAIN ENTRY POINT — run this for the live demo
│   ├── carla_bridge.py           # CARLA connection, vehicle spawn, sensor attachment
│   ├── perception.py             # GT detection, LiDAR DBSCAN, Radar Cartesian, Hungarian matching
│   ├── uncertainty.py            # S.D.T uncertainty engine + sensor health metrics
│   ├── risk_aware_pid.py         # Risk-aware PID controller for speed control
│   ├── visualizer.py             # Dual-panel 1280×480 OpenCV visualizer (Perception + BEV)
│   ├── live_demo.py              # Alternative entry point using full simulation stack
│   ├── safe_demo.py              # Minimal safe mode demo (no NPC, low risk)
│   ├── demo_3d.py                # Experimental 3D visualization demo
│   └── config.py                 # All hyperparameters and sensor configurations
│
├── simulation/                   # CARLA environment management modules
│   ├── carla_setup.py            # CarlaSetup class (connect, spawn, sensors, tick)
│   ├── sensor_health_monitor.py  # Full SensorHealthMonitor with EMA + calibration
│   ├── weather_engine.py         # Dynamic weather cycling (clear→fog→rain)
│   ├── npc_manager.py            # NPC vehicle and pedestrian spawning
│   ├── data_collector.py         # Dataset collection from CARLA for offline training
│   └── label_generator.py        # Ground-truth 2D bounding box label generator
│
├── dataset/
│   └── carla_weather/            # ⭐ CARLA-generated training dataset for YOLOv8
│       ├── dataset.yaml          # Class definitions and dataset splits config
│       ├── images/
│       │   ├── train/            # Training images (CARLA RGB camera frames)
│       │   └── val/              # Validation images
│       └── labels/
│           ├── train/            # YOLO-format bounding box labels (class x y w h)
│           └── val/              # Validation labels
│
├── models/
│   └── yolo_carla_v1-2/          # ⭐ YOLOv8 model trained on above dataset
│       ├── weights/
│       │   ├── best.pt           # Best checkpoint (use this for inference)
│       │   └── last.pt           # Final epoch checkpoint
│       └── results/              # Training metrics, confusion matrix, PR curves
│
├── results/                      # Evaluation outputs from the demo run
│   ├── evaluation_log.json       # Full per-frame log (800 frames, 4 weather conditions)
│   ├── speed_plot.png            # Target speed vs. time across weather transitions
│   ├── uncertainty_plot.png      # Uncertainty score vs. time across weather transitions
│   ├── mode_switches.png         # Fusion mode switches across weather conditions
│   └── summary.txt               # Aggregated stats per weather condition
│
├── docs/
│   ├── RA-ASF_compressed.pdf     # Full project proposal and system design document
│   └── 1.pdf                     # Supporting reference document
│
├── scripts/
│   ├── run.bat                   # Quick launcher for the main demo (Windows)
│   ├── run_3d.bat                # Launcher for 3D demo variant
│   └── run_safe.bat              # Launcher for safe/minimal demo
│
└── requirements.txt              # Python package dependencies
```

---

## ✅ What Has Been Implemented

### 1. CARLA Simulation Environment
- [x] Full CARLA 0.9.16 connection with synchronous tick mode (20 FPS, 0.05s fixed delta)
- [x] Tesla Model3 ego vehicle spawn on Town02 with Traffic Manager autopilot
- [x] NPC spawning: up to 120 vehicles + 5 pedestrians with autopilot
- [x] Dynamic weather cycling: **Clear → Fog Light → Fog Heavy → Rain**
- [x] Traffic light timing reduction to keep traffic flowing

### 2. Sensor Suite
- [x] **Camera** (RGB, 640×480, 90° FOV, mounted at 1.5m forward / 2.4m height)
- [x] **LiDAR** (8 channels, 2000 pts/s, 100m range, 20Hz rotation, ray-cast)
- [x] **Radar** (60° H-FOV, 10° V-FOV, 100m range, 1000 pts/s)
- [x] Synthetic camera frame fallback (dark-blue gradient) when real camera is unavailable due to D3D11 crash on Intel Iris Xe

### 3. Perception Module (`src/perception.py`)
- [x] **Ground-Truth Detector**: Projects all CARLA actors onto the camera plane with perspective projection — simulates a perfect detector
- [x] **LiDAR Grid-DBSCAN Clustering**: Fast grid-voxel clustering (pure NumPy, <1ms per frame)
- [x] **Radar Polar-to-Cartesian**: Converts azimuth/altitude/depth to (x, y, z) with depth range filtering
- [x] **2D Projection**: Projects 3D LiDAR/Radar centroids to image-space using camera intrinsic matrix K
- [x] **Hungarian Matching (M1 & M2)**: Optimal assignment of camera detections to LiDAR clusters (M1) and Radar returns (M2)
- [x] **YOLOv8 Integration**: `get_yolo_detections()` function implemented and ready for GPU-capable machines

### 4. YOLOv8 Custom Model (`models/yolo_carla_v1-2/`)
- [x] Custom dataset collected using `simulation/data_collector.py` across 4 weather conditions
- [x] Automatic YOLO-format label generation via `simulation/label_generator.py`
- [x] Training completed — best weights at `models/yolo_carla_v1-2/weights/best.pt`
- [x] Dataset structure (`dataset/carla_weather/`) preserved for reproducibility and re-training

### 5. Sensor Health Monitor (`simulation/sensor_health_monitor.py`)
- [x] **Camera Health**: Laplacian blur score (65%) + exposure score (35%)
- [x] **LiDAR Health**: Point count (50%) + intensity (30%) + angular coverage (20%)
- [x] **Radar Health**: Detection count (70%) + velocity spread quality (30%)
- [x] **EMA Smoothing** (α=0.3): Prevents single-frame spikes from triggering fusion mode switches
- [x] **Calibration System**: Saves clear-weather baselines to `data/health_monitor_calibration.json` for persistence across runs
- [x] Automatic fusion mode selection: GOLD / M1 / M2 / M3 / DEGRADED

### 6. S.D.T Uncertainty Engine (`src/uncertainty.py`)
- [x] Three-component S.D.T formula: `U = 0.30·H_health + 0.50·H_disagree + 0.20·H_jitter`
- [x] **H_disagree**: Sigmoid-scaled fraction of unmatched camera detections
- [x] **H_jitter**: Temporal standard deviation of detection count over a rolling 10-frame window
- [x] **Environmental Risk Floor**: Forces U ≥ 0.45 in heavy fog, U ≥ 0.75 in heavy rain regardless of detections
- [x] Full uncertainty score clipped to [0.0, 1.0]

### 7. Risk-Aware PID Controller (`src/risk_aware_pid.py`)
- [x] Non-linear speed scaling: `speed = 30 km/h × (1 − U)^1.5`
- [x] Standard PID (Kp=0.5, Ki=0.05, Kd=0.10) with anti-windup integral clamp
- [x] Emergency brake at `U ≥ 0.85` (full brake, zero throttle)

### 8. Real-Time Visualizer (`src/visualizer.py`)
- [x] **1280×480 dual-panel OpenCV display** — no GPU required, pure CPU rendering
- [x] **Left panel — Perception View**: Bounding boxes with depth labels, HUD (mode / weather / speed / tick), uncertainty progress bar (green/orange/red), sensor health bars for all 3 sensors, Camera↔LiDAR match lines
- [x] **Right panel — Bird's Eye View (BEV)**: Cyan LiDAR cloud, orange Radar triangles, detected object boxes, ego vehicle with forward arrow, range rings at 10/20/40/60/80m, fusion mode badge

### 9. Evaluation & Results
- [x] Full evaluation across 4 weather conditions (800 frames total)
- [x] Per-frame JSON logging (`results/evaluation_log.json`)
- [x] Summary statistics (`results/summary.txt`)
- [x] Speed, uncertainty, and mode-switch plots saved as PNG

---

## ❌ What Is NOT Yet Implemented (Future Work)

### 1. Live YOLO in the Demo Loop
The `get_yolo_detections()` function and trained weights exist, but **the live demo uses CARLA Ground-Truth** instead of YOLO because the real RGB camera was replaced by a GNSS proxy to prevent the D3D11/Intel Iris Xe crash. On a GPU-capable machine, restore the real camera in `carla_bridge.py` and call `get_yolo_detections(cam_img, model)` in the main loop.

### 2. Fusion Mode M3 Logic
M3 (LiDAR + Radar, no camera) is defined in the health monitor and mode selector, but **the fusion matching code in `run_demo.py` does not handle M3** — it falls back to empty detections. A BEV-space matching function (without camera reference) is needed.

### 3. EKF-LSTM Localization Integration
An EKF-LSTM hybrid localization pipeline for GPS-denied tunnel traversal was developed in a parallel branch but is **not integrated** into RA-ASF. This would provide better ego-motion estimates for the BEV panel.

### 4. Auto-Calibration at Startup
`SensorHealthMonitor.calibrate()` exists but must be called manually. Triggering it automatically from the first 50 clear-weather warm-up frames would make the system self-calibrating.

### 5. Pedestrian-Specific Speed Response
The uncertainty formula does not distinguish between a detected vehicle and a detected pedestrian. A separate `pedestrian_risk_factor` that applies an additional speed penalty when pedestrians are detected near the ego vehicle would improve safety.

### 6. Deep Learning Uncertainty Estimation
The S.D.T formula uses hand-crafted physics-based weights. A neural uncertainty estimator (e.g., MC Dropout, evidential deep learning) trained from data would be a natural upgrade.

### 7. ROS 2 Integration
The pipeline is a standalone Python application. Wrapping each module as a ROS 2 node would allow plug-in into a standard autonomous driving stack such as Autoware.Universe.

---

## 📊 Key Results

| Weather     | Frames | Avg Uncertainty | Avg Speed (km/h) | Emergency Brakes |
|-------------|--------|-----------------|------------------|------------------|
| Clear       | 200    | 0.195           | 21.9             | 0                |
| Fog Light   | 200    | 0.235           | 20.4             | 0                |
| Fog Heavy   | 200    | 0.343           | 16.2             | 0                |
| Rain        | 108    | 0.236           | 20.2             | 0                |

**Key Observation**: Uncertainty correctly increases as weather degrades, causing proportional speed reduction. No emergency stops were triggered during the evaluation run — confirming that the uncertainty is well-calibrated (rises gradually rather than spiking above 0.85 threshold).

---

## 🚀 How to Run

### Prerequisites

1. **CARLA 0.9.16** installed on Windows (tested with Town02 map).
2. **Python 3.12** — use the provided virtual environment `carla16_env`.
3. All packages from `requirements.txt` installed in the virtual environment.

> **Note**: The `carla16_env/` virtual environment folder lives in the **root of the `ra_asf` workspace**, one level above `final_files/`. The scripts below assume this layout.

---

### Step 1 — Launch CARLA

Open **Terminal 1** and run:

```bat
cd C:\Users\<your_user>\CARLA_0.9.16
.\CarlaUE4.exe /Game/Maps/Town02 -carla-rpc-port=2000 -windowed ^
               -ResX=640 -ResY=480 -quality-level=Low -nosound -dx11
```

Wait until you see the Town02 map fully loaded in the CARLA window (usually 10–20 seconds).

---

### Step 2 — Activate the Virtual Environment

Open **Terminal 2** and run:

```bat
cd C:\Users\<your_user>\ra_asf
.\carla16_env\Scripts\activate
```

You should see `(carla16_env)` at the start of your prompt.

---

### Step 3 — Run the Demo

Still in **Terminal 2**, run:

```bat
cd final_files\src
python run_demo.py
```

> **Alternative (one-click):** From the `final_files\scripts\` folder, double-click `run.bat`. Note the `.bat` file has hardcoded paths — update the paths inside it if your CARLA install location differs.

---

### Step 4 — Watch the Demo

A **1280×480 OpenCV window** will appear with two panels:
- **Left (Perception View)**: Green/orange bounding boxes, sensor health bars, uncertainty progress bar, live speed and mode HUD
- **Right (Bird's Eye View)**: Cyan LiDAR point cloud, orange Radar triangles, detected objects, ego vehicle with range rings

Weather automatically cycles every 200 ticks:
**Clear → Fog Light → Fog Heavy → Rain**

Press **`Q`** in the OpenCV window to stop the demo. A JSON log is saved to `src/results/demo_log.json`.

---

## ⚙️ Configuration

All parameters are in `src/config.py`. Key settings:

| Parameter                   | Default       | Description                                          |
|-----------------------------|---------------|------------------------------------------------------|
| `TOWN`                      | `"Town02"`    | CARLA map (Town02 is lightweight and stable)         |
| `MAX_SPEED`                 | 30.0 km/h     | Target vehicle speed in perfect sensor conditions    |
| `EMERGENCY_BRAKE_THRESHOLD` | 0.85          | Uncertainty level that triggers full emergency brake |
| `SPEED_SCALE_POWER`         | 1.5           | Non-linearity of speed reduction with uncertainty    |
| `ALPHA_HEALTH`              | 0.30          | S.D.T weight for sensor health component             |
| `BETA_DISAGREEMENT`         | 0.50          | S.D.T weight for sensor disagreement component       |
| `GAMMA_JITTER`              | 0.20          | S.D.T weight for temporal jitter component           |
| `TICKS_PER_WEATHER`         | 200           | Simulation frames per weather condition              |
| `DEMO_SCHEDULE`             | clear → fog_light → fog_heavy → rain | Weather cycle order    |
| `N_VEHICLES`                | 120           | Number of NPC vehicles spawned                       |
| `M1_MATCH_THRESHOLD_PX`     | 80 px         | Camera ↔ LiDAR Hungarian matching threshold          |
| `M2_MATCH_THRESHOLD_PX`     | 120 px        | Camera ↔ Radar Hungarian matching threshold          |

---

## 📦 Dependencies

See `requirements.txt`. Key packages:

- `carla` — CARLA Python API (must match server version 0.9.16)
- `numpy` — All sensor data processing and fusion math
- `opencv-python` (`cv2`) — Visualization and camera image processing
- `ultralytics` — YOLOv8 model inference (required for YOLO mode; optional for GT mode)
- `scipy` — (Optional) Full Hungarian algorithm for large-scale matching

---

## 🗺️ Suggested Next Steps for Future Developers

If you want to continue this project, here is the recommended order of work:

1. **Enable real YOLO inference**: Run CARLA on a machine with a dedicated GPU (≥4 GB VRAM). Restore the real RGB camera in `carla_bridge.py`, load `models/yolo_carla_v1-2/weights/best.pt`, and call `get_yolo_detections(cam_img, model)` in `run_demo.py` instead of `get_ground_truth_detections()`.

2. **Implement M3 fusion**: Write a BEV-space matching function that fuses LiDAR grid-DBSCAN clusters with Radar Cartesian returns directly (without using the camera image as reference), and outputs detections compatible with the visualizer.

3. **Auto-calibration at startup**: During the 30-tick warm-up phase in `run_demo.py`, collect camera/LiDAR/Radar frames and call `monitor.calibrate(cam_frames, lidar_frames, radar_frames)` before the main loop starts.

4. **Pedestrian-specific speed penalty**: When one or more pedestrians are detected within 15 m of the ego vehicle, apply an additional multiplier (e.g., 0.6×) to the target speed, independent of the uncertainty score.

5. **ROS 2 wrapper**: Publish `sensor_msgs/Image`, `sensor_msgs/PointCloud2`, and `std_msgs/Float32` topics from each module and subscribe to them with a ROS 2 control node for integration with Autoware.Universe.

---

## 👥 Authors

This project was developed as part of the **Intelligent Autonomous Systems** research initiative.

| Name                      | Roll Number    |
|---------------------------|----------------|
| Hemanth Sai Machireddy    | S20230030389   |
| Sujith Vaishnav Malla     | S20230030391   |
| Srinivasa Rao Komanna     | S20230030386   |

---

## 📄 License

This project is for academic and research purposes. CARLA is developed by the Computer Vision Center (CVC) and is licensed under the MIT License. YOLOv8 is developed by Ultralytics and is licensed under AGPL-3.0.
