#!/bin/bash
# ============================================================
# FSGP-BGK ROS1 (Noetic) Setup Script
# ============================================================
# This script configures the workspace for ROS1 Noetic builds.
# It sets up the correct IGNORE files and builds the project.
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "  FSGP-BGK ROS1 (Noetic) Setup"
echo "============================================================"

# Check ROS1 environment
if [ -z "$ROS_DISTRO" ]; then
    echo "[WARN] ROS environment not sourced. Trying to source ROS Noetic..."
    if [ -f "/opt/ros/noetic/setup.bash" ]; then
        source /opt/ros/noetic/setup.bash
        echo "[OK] Sourced /opt/ros/noetic/setup.bash"
    else
        echo "[ERROR] ROS Noetic not found. Please install ROS Noetic first."
        exit 1
    fi
elif [ "$ROS_VERSION" != "1" ]; then
    echo "[ERROR] ROS2 environment detected (ROS_DISTRO=$ROS_DISTRO)."
    echo "        Please use a clean terminal or source ROS1 Noetic."
    exit 1
else
    echo "[OK] ROS1 environment detected (ROS_DISTRO=$ROS_DISTRO)"
fi

# Configure IGNORE files for ROS1
echo ""
echo "[1/4] Configuring IGNORE files..."

# ROS2 packages should be ignored by catkin
touch src/simulation_env_ros2/CATKIN_IGNORE
touch src/fsgp_bgk/CATKIN_IGNORE

# IMPORTANT: Remove COLCON_IGNORE from ROS1 package (catkin_pkg also reads it!)
rm -f src/simulation_env_ros1/COLCON_IGNORE

echo "      - src/simulation_env_ros2/CATKIN_IGNORE [created]"
echo "      - src/fsgp_bgk/CATKIN_IGNORE [created]"
echo "      - src/simulation_env_ros1/COLCON_IGNORE [removed]"

# Clean previous build
echo ""
echo "[2/4] Cleaning previous build..."
rm -rf build/ devel/ install/ log/
echo "      Done."

# Build
echo ""
echo "[3/4] Building with catkin_make..."
catkin_make -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

# Source workspace
echo ""
echo "[4/4] Setup complete!"
echo ""
echo "============================================================"
echo "  To use the workspace, run:"
echo "    source devel/setup.bash"
echo ""
echo "  To launch simulation:"
echo "    roslaunch simulation_env_ros1 simulation.launch"
echo ""
echo "  To run FSGP-BGK node:"
echo "    cd src/fsgp_bgk/python && python node_ros1.py"
echo "============================================================"

