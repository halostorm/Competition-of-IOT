cmake_minimum_required(VERSION 2.6)
set( CMAKE_BUILD_TYPE "Release" )
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )

project(trangulate_show)

find_package( OpenCV 3.1 REQUIRED )

include_directories( 
    ${OpenCV_INCLUDE_DIRS} 
 #   "/usr/include/eigen3/"
)

add_executable(trangulate_show main.cpp)
target_link_libraries( trangulate_show ${OpenCV_LIBS} )

install(TARGETS trangulate_show RUNTIME DESTINATION bin)
