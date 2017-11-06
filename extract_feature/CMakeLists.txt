cmake_minimum_required(VERSION 2.6)

project(extract_feature_from_two_pic)
set( CMAKE_CXX_FLAGS "-std=c++11" )


find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(extract_feature_from_two_pic main.cpp)
target_link_libraries(extract_feature_from_two_pic ${OpenCV_LIBS})

install(TARGETS extract_feature_from_two_pic RUNTIME DESTINATION bin)
