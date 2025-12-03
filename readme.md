# FSGP-BGK: Real-time Spatial-temporal Traversability Assessment

Official implementation of our feature-based sparse Gaussian process method for real-time terrain analysis and autonomous navigation (IROS 2025).

![Simulation](src/simulation_env_ros2/doc/pic.png)

## Features

- **Sparse Gaussian Process (SGP)** with inducing points for efficient computation
- **Bayesian Generalized Kernel (BGK)** for uncertainty estimation
- **GPU acceleration** with CUDA/CuPy support
- **ROS1 (Noetic)** and **ROS2 (Humble/Iron)** support
- **3D simulation environment** with simulated LiDAR

## Dependencies

```bash
# Python
pip install numpy open3d gpytorch torch cupy-cuda11x

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
catkin_make
source devel/setup.bash
```

### Run Simulation
```bash
# Terminal 1 - Launch simulation
roslaunch simulation_env_ros1 simulation.launch

# Terminal 2 - Keyboard control
rosrun teleop_twist_keyboard teleop_twist_keyboard.py

# Terminal 3 - Run FSGP-BGK node
cd src/fsgp_bgk/python && python node_ros1.py
```

---

## ROS2 (Humble/Iron)

### Build
```bash
cd ~/FSGP-BGK-ROS2
colcon build
source install/setup.bash
```

### Run Simulation
```bash
# Terminal 1 - Launch simulation
ros2 launch simulation_env_ros2 simulation.launch.py

# Terminal 2 - Keyboard control
ros2 run teleop_twist_keyboard teleop_twist_keyboard

# Terminal 3 - Run FSGP-BGK node
cd src/fsgp_bgk/python && python node_ros2.py
```

---

## Switching Between ROS1 and ROS2

The workspace supports both ROS versions. Use **CATKIN_IGNORE** files to exclude incompatible packages:

| Package | ROS1 | ROS2 |
|---------|------|------|
| `simulation_env_ros1` | ✅ Built by catkin | ❌ Has CATKIN_IGNORE |
| `simulation_env_ros2` | ❌ Has CATKIN_IGNORE | ✅ Built by colcon |
| `fsgp_bgk` | ❌ Has CATKIN_IGNORE | ❌ Has CATKIN_IGNORE |

> **Note**: `fsgp_bgk` is a pure Python package, run directly without building.

### For ROS1 users:
```bash
catkin_make   # Only builds simulation_env_ros1
```

### For ROS2 users:
```bash
colcon build  # Only builds simulation_env_ros2
```

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