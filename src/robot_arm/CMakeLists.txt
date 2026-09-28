cmake_minimum_required(VERSION 3.8)
project(robot_arm)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)

# Install directories
install(DIRECTORY
  launch
  config
  urdf
  meshes
  rviz
  worlds
  DESTINATION share/${PROJECT_NAME}
)

ament_python_install_package(${PROJECT_NAME})

# Install Python executable nodes
install(PROGRAMS
  robot_arm/grasp_detection_node.py
  robot_arm/pick_place_node.py
  robot_arm/anygrasp_service.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
