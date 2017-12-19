cmake_minimum_required(VERSION 2.8)
project(aka)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread -g -O0")
link_directories(/home/ubuntu/Flight2/lib)
link_directories(/usr/local/lib)
link_directories(/home/ubuntu/Onboard-sdk/build/libs)
include_directories(/home/ubuntu/Flight2)
include_directories(/usr/local/include/eigen3)
include_directories(/home/ubuntu/Downloads/aruco-aruco-git/src)
include_directories(/home/ubuntu/Onboard-sdk/osdk-core/inc)
include_directories(/home/ubuntu/Onboard-sdk/osdk-wrapper/inc)
include_directories(/home/ubuntu/Onboard-sdk/platform/linux/inc)


find_package(OpenCV REQUIRED)


add_executable(${PROJECT_NAME} main.cpp)
add_executable(test main2.cpp)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} pthread libaruco.so libusb-1.0.so libdjiosdk-core.a libdjiosdk-platform.a libdjiosdk-wrapper.a)
#target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} pthread  libusb-1.0.so )
target_link_libraries(test ${OpenCV_LIBS} pthread libaruco.so libusb-1.0.so libdjiosdk-core.a libdjiosdk-platform.a libdjiosdk-wrapper.a)
#target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} pthread  libusb-1.0.so )
