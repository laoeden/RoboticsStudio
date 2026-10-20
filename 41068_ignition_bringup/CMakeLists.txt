cmake_minimum_required(VERSION 3.8)
project(41068_ignition_bringup)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)

find_package(ros_ign_interfaces REQUIRED)
find_package(ignition-gazebo6 REQUIRED)
find_package(ignition-transport11 REQUIRED)
find_package(ignition-msgs8 REQUIRED)

add_library(planar_velocity_control SHARED
  src/planar_velocity_control.cc
)
target_compile_features(planar_velocity_control PUBLIC cxx_std_17)
target_link_libraries(planar_velocity_control
  ignition-gazebo6::core
  ignition-transport11::core
  ignition-msgs8::core
)
target_include_directories(planar_velocity_control PRIVATE
  ${ignition-gazebo6_INCLUDE_DIRS}
  ${ignition-transport11_INCLUDE_DIRS}
  ${ignition-msgs8_INCLUDE_DIRS}
)
install(TARGETS planar_velocity_control
  LIBRARY DESTINATION lib
)

install(
  DIRECTORY config launch models urdf_husky worlds urdf_parrot
  DESTINATION share/${PROJECT_NAME}
)

install(
  PROGRAMS
    scripts/dynamic_world_demo.py
    scripts/basic_autonomy_demo.py
    ${CMAKE_CURRENT_SOURCE_DIR}/../lidar_3d_plot.py
    ${CMAKE_CURRENT_SOURCE_DIR}/../voxel_nav_explorer.py
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
