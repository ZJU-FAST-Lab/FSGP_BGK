cmake_minimum_required(VERSION 3.0.2)
project(simulation_env_ros1)

# Use C++14 standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compile options
add_compile_options(-Wall -Wextra)

# Find catkin dependencies
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

# Find third-party libraries
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

# Include directories
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

# Install executables
install(TARGETS simulation_node simulated_lidar
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# Install launch files
install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

# Install config files
install(DIRECTORY config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
)

# Install rviz config
install(DIRECTORY rviz/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/rviz
)

# Install mesh files
install(DIRECTORY meshes/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/meshes
)

# Install PCD files
install(DIRECTORY PCD/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/PCD
)

