cmake_minimum_required(VERSION 3.8)
project(carma_sim)

find_package(ament_cmake REQUIRED)

# worlds/  the row_crop xacro template
# models/  carma_ugv; GZ_SIM_RESOURCE_PATH points at this directory so that
#          model://carma_ugv resolves
# launch/  row_crop.launch.py
# config/  the ros_gz_bridge topic mapping
install(DIRECTORY worlds models launch config DESTINATION share/${PROJECT_NAME})

# bgr_relay.py is a node, not data: it goes in lib/ so that ros2 run and the
# launch file can find it as an executable.
install(PROGRAMS scripts/bgr_relay.py DESTINATION lib/${PROJECT_NAME})

ament_package()
