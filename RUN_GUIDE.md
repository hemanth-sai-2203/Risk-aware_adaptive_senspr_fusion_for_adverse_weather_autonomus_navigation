# RA-ASF File Execution Guide

This document clarifies which files are meant to be executed directly from the terminal, which are alternative demos, and which are strictly internal modules (do not run them directly).

---

## 🟢 1. Main Entry Points (Run These!)

These are the primary files you will run to launch the project demonstrations.

| File Path | Purpose | How to Run |
|-----------|---------|------------|
| `src/run_demo.py` | **⭐ The Main Live Demo**. This is the full Risk-Aware Sensor Fusion demonstration. It connects to CARLA, spawns the ego vehicle and 120 NPCs, attaches sensors, and opens the dual-panel OpenCV visualizer showing Perception + Bird's Eye View while cycling through weather conditions. | `python src/run_demo.py` |
| `scripts/run.bat` | **Windows Launcher**. A convenient batch script that automatically kills stuck CARLA instances, launches CARLA in Town02, waits for it to load, activates the Python environment, and runs `run_demo.py`. | Double-click or `.\scripts\run.bat` |

---

## 🟡 2. Alternative & Utility Scripts (Run for Specific Tasks)

These files can be executed directly but are meant for specific scenarios like testing, dataset collection, or safe mode.

| File Path | Purpose | How to Run |
|-----------|---------|------------|
| `src/safe_demo.py` | A lightweight version of the demo that runs with **no NPC traffic** and minimal overhead. Great for debugging the uncertainty formula in isolation. | `python src/safe_demo.py` |
| `src/demo_3d.py` | An experimental visualization demo that attempts to plot data in 3D. (Note: Can be very resource-intensive). | `python src/demo_3d.py` |
| `src/live_demo.py` | An older, alternative version of the demo script utilizing the full simulation stack differently. | `python src/live_demo.py` |
| `simulation/data_collector.py`| Runs the vehicle through CARLA to record raw sensor data (Images, LiDAR, Radar). Used to build the training dataset. | `python simulation/data_collector.py` |
| `simulation/label_generator.py`| Processes the raw collected sensor data to generate YOLO-format 2D bounding box labels based on CARLA ground truth. | `python simulation/label_generator.py` |

---

## 🔴 3. Internal Modules (Do NOT Run Directly)

These files contain the core logic, classes, and configurations of the project. They are designed to be **imported** by the main scripts (`run_demo.py`). Running them directly will either do nothing or just execute a quick internal unit test.

### Core Architecture Modules (`src/`)
*   `config.py` - Contains all hyperparameter settings (speeds, thresholds, weights). Edit this to tweak system behavior.
*   `carla_bridge.py` - Manages the CARLA API connection, spawning, and tick loops.
*   `perception.py` - Contains functions for 3D clustering, polar-to-cartesian conversion, and Hungarian matching.
*   `uncertainty.py` - Contains the logic for the S.D.T formula and calculating the final risk score.
*   `risk_aware_pid.py` - The PID speed controller that dynamically adjusts throttle/brake based on uncertainty.
*   `visualizer.py` - Draws the HUD and 1280x480 OpenCV dashboard.

### Simulation Handlers (`simulation/`)
*   `carla_setup.py` - Environment setup logic, designed to handle map changes and server crashes.
*   `sensor_health_monitor.py` - Analyzes sensor clarity, density, and noise to output health scores.
*   `weather_engine.py` - Cycles CARLA weather dynamically.
*   `npc_manager.py` - Handles spawning and autopilot management for background traffic and pedestrians.
