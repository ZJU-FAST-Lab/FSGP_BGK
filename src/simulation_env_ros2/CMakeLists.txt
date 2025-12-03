cmake_minimum_required(VERSION 3.8)
project(simulation_env)

# Enable ament_auto extension
find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies(
  REQUIRED
  rclcpp
  visualization_msgs
  geometry_msgs
  tf2_ros
  tf2_geometry_msgs
  nav_msgs
  pcl_conversions
)

# Find third-party libraries
find_package(PCL REQUIRED COMPONENTS common)
find_package(Eigen3 REQUIRED)

# Set compile options
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Create executable target
ament_auto_add_executable(simulation_node
  src/simulation_node.cpp
)

# Include third-party libraries
target_include_directories(simulation_node PRIVATE
  ${PCL_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(simulation_node
  ${PCL_LIBRARIES}
  pcl_common
  Eigen3::Eigen
)

# Install resources
install(DIRECTORY
  rviz
  meshes
  DESTINATION share/${PROJECT_NAME}
)

add_executable(simulated_lidar src/simulated_lidar.cpp)
ament_target_dependencies(simulated_lidar rclcpp sensor_msgs nav_msgs pcl_conversions tf2_ros tf2_eigen)

install(TARGETS
  simulated_lidar
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY
  config
  DESTINATION share/${PROJECT_NAME}
)
# Auto-handle installation and export
ament_auto_package()