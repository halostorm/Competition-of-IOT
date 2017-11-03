cmake_minimum_required(VERSION 2.6)
project(object_detect_tracking)


find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(detect_tracking main.cpp)

target_link_libraries(detect_tracking ${OpenCV_LIBS})

install(TARGETS detect_tracking RUNTIME DESTINATION bin)
