cmake_minimum_required(VERSION 3.8)
project(simulation_env)

# 启用ament_auto扩展
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

# 手动添加第三方库查找
find_package(PCL REQUIRED COMPONENTS common)
find_package(Eigen3 REQUIRED)

# 设置编译选项
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# 创建可执行目标
ament_auto_add_executable(simulation_node
  src/simulation_node.cpp
)

# 包含第三方库
target_include_directories(simulation_node PRIVATE
  ${PCL_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)

# 链接库
target_link_libraries(simulation_node
  ${PCL_LIBRARIES}
  pcl_common
  Eigen3::Eigen
)

# 安装资源配置
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
# 自动处理安装和导出
ament_auto_package()