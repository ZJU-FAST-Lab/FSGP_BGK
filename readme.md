# build
colcon build
# fsgp_bgk 
cd src/fsgp_bgk/python && python node_ros2.py
# simulation_env
source install/setup.zsh   
ros2 launch simulation_env simulation.launch.py 
# keyboard
ros2 run teleop_twist_keyboard teleop_twist_keyboard