cmake_minimum_required(VERSION 3.8)
project(carma_sim)

find_package(ament_cmake REQUIRED)

# Worlds and the robot description land with WP1; install them when they exist.
foreach(dir worlds models launch)
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${dir})
    install(DIRECTORY ${dir} DESTINATION share/${PROJECT_NAME})
  endif()
endforeach()

ament_package()
