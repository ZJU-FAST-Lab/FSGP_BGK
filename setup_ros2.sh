#!/bin/bash
# ============================================================
# FSGP-BGK ROS2 (Humble/Iron) Setup Script
# ============================================================
# This script configures the workspace for ROS2 builds.
# It sets up the correct IGNORE files and builds the project.
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "  FSGP-BGK ROS2 (Humble/Iron) Setup"
echo "============================================================"

# Check ROS2 environment
if [ -z "$ROS_DISTRO" ]; then
    echo "[WARN] ROS environment not sourced. Trying to source ROS2..."
    if [ -f "/opt/ros/humble/setup.bash" ]; then
        source /opt/ros/humble/setup.bash
        echo "[OK] Sourced /opt/ros/humble/setup.bash"
    elif [ -f "/opt/ros/iron/setup.bash" ]; then
        source /opt/ros/iron/setup.bash
        echo "[OK] Sourced /opt/ros/iron/setup.bash"
    else
        echo "[ERROR] ROS2 (Humble/Iron) not found. Please install ROS2 first."
        exit 1
    fi
elif [ "$ROS_VERSION" != "2" ]; then
    echo "[ERROR] ROS1 environment detected (ROS_DISTRO=$ROS_DISTRO)."
    echo "        Please use a clean terminal or source ROS2."
    exit 1
else
    echo "[OK] ROS2 environment detected (ROS_DISTRO=$ROS_DISTRO)"
fi

# Configure IGNORE files for ROS2
echo ""
echo "[1/4] Configuring IGNORE files..."

# ROS1 packages should be ignored by colcon
touch src/simulation_env_ros1/COLCON_IGNORE
touch src/fsgp_bgk/COLCON_IGNORE

# Keep CATKIN_IGNORE for ROS2 package (doesn't affect colcon)
touch src/simulation_env_ros2/CATKIN_IGNORE

echo "      - src/simulation_env_ros1/COLCON_IGNORE [created]"
echo "      - src/fsgp_bgk/COLCON_IGNORE [created]"
echo "      - src/simulation_env_ros2/CATKIN_IGNORE [kept]"

# Clean previous build
echo ""
echo "[2/4] Cleaning previous build..."
rm -rf build/ install/ log/ devel/
echo "      Done."

# Build
echo ""
echo "[3/4] Building with colcon..."
colcon build --symlink-install

# Source workspace
echo ""
echo "[4/4] Setup complete!"
echo ""
echo "============================================================"
echo "  To use the workspace, run:"
echo "    source install/setup.bash"
echo ""
echo "  To launch simulation:"
echo "    ros2 launch simulation_env simulation.launch.py"
echo ""
echo "  To run FSGP-BGK node:"
echo "    cd src/fsgp_bgk/python && python node_ros2.py"
echo "============================================================"

