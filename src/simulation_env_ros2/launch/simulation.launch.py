from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Get package share directory path
    pkg_share_dir = get_package_share_directory('simulation_env')

    # Define RViz config file path
    rviz_config_file = os.path.join(pkg_share_dir, 'rviz', 'simulation.rviz')

    # Define parameter file path
    params_file = os.path.join(pkg_share_dir, 'config', 'params.yaml')

    # Define nodes
    simulation_node = Node(
        package='simulation_env',
        executable='simulation_node',
        name='simulation_node',
        output='screen',
        parameters=[params_file]
    )

    simulated_lidar_node = Node(
        package='simulation_env',
        executable='simulated_lidar',
        name='simulated_lidar',
        output='screen',
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_file],
        output='screen'
    )

    # Return LaunchDescription
    return LaunchDescription([
        simulation_node,
        simulated_lidar_node,
        rviz_node
    ])