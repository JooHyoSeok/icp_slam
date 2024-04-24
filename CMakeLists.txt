cmake_minimum_required(VERSION 3.0.2)
project(pyicp_slam)

find_package(catkin REQUIRED COMPONENTS
    pybind11_catkin)
find_package(Eigen3 REQUIRED)
catkin_package(
    LIBRARIES
    CATKIN_DEPENDS pybind11_catkin
)

include_directories(
    include
    ${catkin_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
)

find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

pybind_add_module(calc_distance_sc MODULE src/calc_distance_sc.cpp)

catkin_install_python(PROGRAMS scripts/utils/ScanContextManager.py
                      DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

install(TARGETS calc_distance_sc
  LIBRARY DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

