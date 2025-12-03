cmake_minimum_required(VERSION 3.0.2)
project(simulation_env_ros1)

# 使用C++14标准
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 编译选项
add_compile_options(-Wall -Wextra)

# 查找catkin依赖
find_package(catkin REQUIRED COMPONENTS
  roscpp
  visualization_msgs
  geometry_msgs
  sensor_msgs
  nav_msgs
  tf2_ros
  tf2_eigen
  tf2_geometry_msgs
  pcl_ros
  pcl_conversions
)

# 查找第三方库
find_package(PCL REQUIRED COMPONENTS common io filters kdtree)
find_package(Eigen3 REQUIRED)

catkin_package(
  CATKIN_DEPENDS
    roscpp
    visualization_msgs
    geometry_msgs
    sensor_msgs
    nav_msgs
    tf2_ros
    tf2_eigen
    tf2_geometry_msgs
    pcl_ros
    pcl_conversions
)

# 包含目录
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)

# simulation_node
add_executable(simulation_node src/simulation_node.cpp)
target_link_libraries(simulation_node
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
)

# simulated_lidar
add_executable(simulated_lidar src/simulated_lidar.cpp)
target_link_libraries(simulated_lidar
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
)

# 安装可执行文件
install(TARGETS simulation_node simulated_lidar
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# 安装launch文件
install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

# 安装配置文件
install(DIRECTORY config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
)

# 安装rviz配置
install(DIRECTORY rviz/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/rviz
)

# 安装mesh文件
install(DIRECTORY meshes/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/meshes
)

# 安装PCD文件
install(DIRECTORY PCD/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/PCD
)

