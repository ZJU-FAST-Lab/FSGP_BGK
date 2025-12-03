# FSGP-BGK: Real-time Spatial-temporal Traversability Assessment

Official implementation of our feature-based sparse Gaussian process method for real-time terrain analysis and autonomous navigation (IROS 2025).

![Simulation](src/simulation_env_ros2/doc/pic.png)

## Features

- **Sparse Gaussian Process (SGP)** with inducing points for efficient computation
- **Bayesian Generalized Kernel (BGK)** for uncertainty estimation
- **GPU acceleration** with CUDA/CuPy support
- **ROS1 (Noetic)** and **ROS2 (Humble)** support
- **3D simulation environment** with simulated LiDAR

---

## Dependencies

```bash
# Python
pip install numpy open3d gpytorch torch cupy-cuda11x scikit-learn pyyaml

# ROS1 (Noetic)
sudo apt install ros-noetic-tf2-eigen ros-noetic-pcl-ros ros-noetic-teleop-twist-keyboard

# ROS2 (Humble)
sudo apt install ros-humble-tf2-eigen ros-humble-pcl-ros ros-humble-teleop-twist-keyboard
```

---

## ROS1 (Noetic)

### Build
```bash
cd ~/FSGP-BGK-ROS2

# Ignore ROS2 package (required before build)
touch src/simulation_env_ros2/CATKIN_IGNORE

# Build
catkin_make
source devel/setup.bash
```

### Run Simulation
```bash
# Terminal 1 - Launch simulation
source devel/setup.bash
roslaunch simulation_env_ros1 simulation.launch

# Terminal 2 - Keyboard control
rosrun teleop_twist_keyboard teleop_twist_keyboard.py

# Terminal 3 - Run FSGP-BGK node
cd src/fsgp_bgk/python
python node_ros1.py
```

---

## ROS2 (Humble)

### Build
```bash
cd ~/FSGP-BGK-ROS2

# Ignore ROS1 package (required before build)
touch src/simulation_env_ros1/COLCON_IGNORE

# Build
colcon build
source install/setup.bash
```

### Run Simulation
```bash
# Terminal 1 - Launch simulation
source install/setup.bash
ros2 launch simulation_env simulation.launch.py

# Terminal 2 - Keyboard control
ros2 run teleop_twist_keyboard teleop_twist_keyboard

# Terminal 3 - Run FSGP-BGK node
cd src/fsgp_bgk/python
python node_ros2.py
```

---

## Workspace Structure

| Package | Description | ROS Version |
|---------|-------------|-------------|
| `simulation_env_ros1` | Simulation environment | ROS1 Noetic |
| `simulation_env_ros2` | Simulation environment | ROS2 Humble |
| `fsgp_bgk` | FSGP-BGK algorithm (Python) | Both (run directly) |

> **Note**: `fsgp_bgk` is a pure Python package, run it directly without building.

---

## Configuration

Edit parameters in:
- ROS1: `src/simulation_env_ros1/config/params.yaml`
- ROS2: `src/simulation_env_ros2/config/params.yaml`
- FSGP-BGK: `src/fsgp_bgk/config/params.yaml`

**Important**: Update `pcd_file_path` to your local path before running.

---

## Keyboard Controls

| Key | Action |
|-----|--------|
| `i` | Forward |
| `,` | Backward |
| `j` | Turn left |
| `l` | Turn right |
| `k` | Stop |
| `q/z` | Increase/decrease speed |

---

## License

MIT License