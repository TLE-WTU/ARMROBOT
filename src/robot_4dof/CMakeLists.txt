cmake_minimum_required(VERSION 3.8)
project(robot_4dof)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

# Install Python package (for ik_solver module)
ament_python_install_package(${PROJECT_NAME})

# Install Python executable nodes
install(PROGRAMS
  robot_4dof/grasp_detection_node.py
  robot_4dof/pick_place_node.py
  robot_4dof/anygrasp_service.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install data directories
install(DIRECTORY
  urdf
  launch
  config
  worlds
  rviz
  meshes
  DESTINATION share/${PROJECT_NAME}/
)

ament_package()
