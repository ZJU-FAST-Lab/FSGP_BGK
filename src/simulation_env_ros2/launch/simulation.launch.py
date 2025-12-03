from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # 获取包的共享目录路径
    pkg_share_dir = get_package_share_directory('simulation_env')

    # 定义 RViz 配置文件的路径
    rviz_config_file = os.path.join(pkg_share_dir, 'rviz', 'simulation.rviz')

    # 定义参数文件的路径
    params_file = os.path.join(pkg_share_dir, 'config', 'params.yaml')

    # 定义节点
    simulation_node = Node(
        package='simulation_env',
        executable='simulation_node',
        name='simulation_node',
        output='screen',
        parameters=[params_file]  # 加载参数文件
    )

    simulated_lidar_node = Node(
        package='simulation_env',
        executable='simulated_lidar',
        name='simulated_lidar',
        output='screen',
        # parameters=[params_file]  # 加载参数文件
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_file],
        output='screen'
    )

    # 返回 LaunchDescription
    return LaunchDescription([
        simulation_node,
        simulated_lidar_node,
        rviz_node
    ])