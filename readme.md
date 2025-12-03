# FSGP-BGK: Real-time Spatial-temporal Traversability Assessment

Official implementation of our feature-based sparse Gaussian process method for real-time terrain analysis and autonomous navigation (IROS 2025).

![Simulation](src/simulation_env_ros2/doc/pic.png)

## Features

- **Sparse Gaussian Process (SGP)** with inducing points for efficient computation
- **Bayesian Generalized Kernel (BGK)** for uncertainty estimation
- **GPU acceleration** with CUDA/CuPy support
- **ROS1 (Noetic)** and **ROS2 (Humble/Iron)** support
- **3D simulation environment** with simulated LiDAR

## Quick Start

We provide one-click setup scripts for both ROS1 and ROS2 users:

```bash
# Clone the repository
git clone https://github.com/MarineRock10/FSGP-BGK-ROS2.git
cd FSGP-BGK-ROS2

# For ROS1 (Noetic) users:
./setup_ros1.sh

# For ROS2 (Humble/Iron) users:
./setup_ros2.sh
```

The setup scripts will automatically configure the workspace and build the project.

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

### Build (Manual)
```bash
cd ~/FSGP-BGK-ROS2
./setup_ros1.sh
# Or manually:
# catkin_make && source devel/setup.bash
```

### Run Simulation
```bash
# Terminal 1 - Launch simulation
source devel/setup.bash
roslaunch simulation_env_ros1 simulation.launch

# Terminal 2 - Keyboard control
rosrun teleop_twist_keyboard teleop_twist_keyboard.py

# Terminal 3 - Run FSGP-BGK node
cd src/fsgp_bgk/python && python node_ros1.py
```

---

## ROS2 (Humble/Iron)

### Build (Manual)
```bash
cd ~/FSGP-BGK-ROS2
./setup_ros2.sh
# Or manually:
# colcon build && source install/setup.bash
```

### Run Simulation
```bash
# Terminal 1 - Launch simulation
source install/setup.bash
ros2 launch simulation_env simulation.launch.py

# Terminal 2 - Keyboard control
ros2 run teleop_twist_keyboard teleop_twist_keyboard

# Terminal 3 - Run FSGP-BGK node
cd src/fsgp_bgk/python && python node_ros2.py
```

---

## Switching Between ROS1 and ROS2

This workspace supports both ROS versions using IGNORE marker files. **Use the setup scripts to switch between versions:**

```bash
# Switch to ROS1
./setup_ros1.sh

# Switch to ROS2
./setup_ros2.sh
```

### How it works

| Package | ROS1 (catkin) | ROS2 (colcon) |
|---------|---------------|---------------|
| `simulation_env_ros1` | ✅ Built | ❌ Ignored |
| `simulation_env_ros2` | ❌ Ignored | ✅ Built |
| `fsgp_bgk` | ❌ Ignored (Python only) | ❌ Ignored (Python only) |

> **Note**: `fsgp_bgk` is a pure Python package, run directly without building.

> **Technical Note**: The setup scripts manage `CATKIN_IGNORE` and `COLCON_IGNORE` files.
> Note that newer versions of `catkin_pkg` also recognize `COLCON_IGNORE`, so the ROS1 setup script
> removes any `COLCON_IGNORE` files from ROS1 packages to ensure proper detection.

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