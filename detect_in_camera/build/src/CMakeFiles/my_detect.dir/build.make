# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.7

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/pi/src/Competition-of-IOT/detect_in_camera

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/pi/src/Competition-of-IOT/detect_in_camera/build

# Include any dependencies generated for this target.
include src/CMakeFiles/my_detect.dir/depend.make

# Include the progress variables for this target.
include src/CMakeFiles/my_detect.dir/progress.make

# Include the compile flags for this target's objects.
include src/CMakeFiles/my_detect.dir/flags.make

src/CMakeFiles/my_detect.dir/config.cpp.o: src/CMakeFiles/my_detect.dir/flags.make
src/CMakeFiles/my_detect.dir/config.cpp.o: ../src/config.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/pi/src/Competition-of-IOT/detect_in_camera/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/CMakeFiles/my_detect.dir/config.cpp.o"
	cd /home/pi/src/Competition-of-IOT/detect_in_camera/build/src && g++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/my_detect.dir/config.cpp.o -c /home/pi/src/Competition-of-IOT/detect_in_camera/src/config.cpp

src/CMakeFiles/my_detect.dir/config.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/my_detect.dir/config.cpp.i"
	cd /home/pi/src/Competition-of-IOT/detect_in_camera/build/src && g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/pi/src/Competition-of-IOT/detect_in_camera/src/config.cpp > CMakeFiles/my_detect.dir/config.cpp.i

src/CMakeFiles/my_detect.dir/config.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/my_detect.dir/config.cpp.s"
	cd /home/pi/src/Competition-of-IOT/detect_in_camera/build/src && g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/pi/src/Competition-of-IOT/detect_in_camera/src/config.cpp -o CMakeFiles/my_detect.dir/config.cpp.s

src/CMakeFiles/my_detect.dir/config.cpp.o.requires:

.PHONY : src/CMakeFiles/my_detect.dir/config.cpp.o.requires

src/CMakeFiles/my_detect.dir/config.cpp.o.provides: src/CMakeFiles/my_detect.dir/config.cpp.o.requires
	$(MAKE) -f src/CMakeFiles/my_detect.dir/build.make src/CMakeFiles/my_detect.dir/config.cpp.o.provides.build
.PHONY : src/CMakeFiles/my_detect.dir/config.cpp.o.provides

src/CMakeFiles/my_detect.dir/config.cpp.o.provides.build: src/CMakeFiles/my_detect.dir/config.cpp.o


# Object files for target my_detect
my_detect_OBJECTS = \
"CMakeFiles/my_detect.dir/config.cpp.o"

# External object files for target my_detect
my_detect_EXTERNAL_OBJECTS =

../lib/libmy_detect.so: src/CMakeFiles/my_detect.dir/config.cpp.o
../lib/libmy_detect.so: src/CMakeFiles/my_detect.dir/build.make
../lib/libmy_detect.so: /usr/local/lib/libopencv_stitching.so.3.2.0
../lib/libmy_detect.so: /usr/local/lib/libopencv_superres.so.3.2.0
../lib/libmy_detect.so: /usr/local/lib/libopencv_videostab.so.3.2.0
../lib/libmy_detect.so: /usr/local/lib/libopencv_aruco.so.3.2.0
../lib/libmy_detect.so: /usr/local/lib/libopencv_bgsegm.so.3.2.0
../lib/libmy_detect.so: /usr/local/lib/libopencv_bioinspired.so.3.2.0
../lib/libmy_detect.so: /usr/local/lib/libopencv_ccalib.so.3.2.0
../lib/libmy_detect.so: /usr/local/lib/libopencv_dpm.so.3.2.0
../lib/libmy_detect.so: /usr/local/lib/libopencv_freetype.so.3.2.0
../lib/libmy_detect.so: /usr/local/lib/libopencv_fuzzy.so.3.2.0
../lib/libmy_detect.so: /usr/local/lib/libopencv_line_descriptor.so.3.2.0
../lib/libmy_detect.so: /usr/local/lib/libopencv_optflow.so.3.2.0
../lib/libmy_detect.so: /usr/local/lib/libopencv_reg.so.3.2.0
../lib/libmy_detect.so: /usr/local/lib/libopencv_saliency.so.3.2.0
../lib/libmy_detect.so: /usr/local/lib/libopencv_stereo.so.3.2.0
../lib/libmy_detect.so: /usr/local/lib/libopencv_structured_light.so.3.2.0
../lib/libmy_detect.so: /usr/local/lib/libopencv_surface_matching.so.3.2.0
../lib/libmy_detect.so: /usr/local/lib/libopencv_tracking.so.3.2.0
../lib/libmy_detect.so: /usr/local/lib/libopencv_xfeatures2d.so.3.2.0
../lib/libmy_detect.so: /usr/local/lib/libopencv_ximgproc.so.3.2.0
../lib/libmy_detect.so: /usr/local/lib/libopencv_xobjdetect.so.3.2.0
../lib/libmy_detect.so: /usr/local/lib/libopencv_xphoto.so.3.2.0
../lib/libmy_detect.so: /usr/local/lib/libopencv_shape.so.3.2.0
../lib/libmy_detect.so: /usr/local/lib/libopencv_phase_unwrapping.so.3.2.0
../lib/libmy_detect.so: /usr/local/lib/libopencv_rgbd.so.3.2.0
../lib/libmy_detect.so: /usr/local/lib/libopencv_calib3d.so.3.2.0
../lib/libmy_detect.so: /usr/local/lib/libopencv_video.so.3.2.0
../lib/libmy_detect.so: /usr/local/lib/libopencv_datasets.so.3.2.0
../lib/libmy_detect.so: /usr/local/lib/libopencv_dnn.so.3.2.0
../lib/libmy_detect.so: /usr/local/lib/libopencv_face.so.3.2.0
../lib/libmy_detect.so: /usr/local/lib/libopencv_plot.so.3.2.0
../lib/libmy_detect.so: /usr/local/lib/libopencv_text.so.3.2.0
../lib/libmy_detect.so: /usr/local/lib/libopencv_features2d.so.3.2.0
../lib/libmy_detect.so: /usr/local/lib/libopencv_flann.so.3.2.0
../lib/libmy_detect.so: /usr/local/lib/libopencv_objdetect.so.3.2.0
../lib/libmy_detect.so: /usr/local/lib/libopencv_ml.so.3.2.0
../lib/libmy_detect.so: /usr/local/lib/libopencv_highgui.so.3.2.0
../lib/libmy_detect.so: /usr/local/lib/libopencv_photo.so.3.2.0
../lib/libmy_detect.so: /usr/local/lib/libopencv_videoio.so.3.2.0
../lib/libmy_detect.so: /usr/local/lib/libopencv_imgcodecs.so.3.2.0
../lib/libmy_detect.so: /usr/local/lib/libopencv_imgproc.so.3.2.0
../lib/libmy_detect.so: /usr/local/lib/libopencv_core.so.3.2.0
../lib/libmy_detect.so: src/CMakeFiles/my_detect.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/pi/src/Competition-of-IOT/detect_in_camera/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared library ../../lib/libmy_detect.so"
	cd /home/pi/src/Competition-of-IOT/detect_in_camera/build/src && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/my_detect.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/CMakeFiles/my_detect.dir/build: ../lib/libmy_detect.so

.PHONY : src/CMakeFiles/my_detect.dir/build

src/CMakeFiles/my_detect.dir/requires: src/CMakeFiles/my_detect.dir/config.cpp.o.requires

.PHONY : src/CMakeFiles/my_detect.dir/requires

src/CMakeFiles/my_detect.dir/clean:
	cd /home/pi/src/Competition-of-IOT/detect_in_camera/build/src && $(CMAKE_COMMAND) -P CMakeFiles/my_detect.dir/cmake_clean.cmake
.PHONY : src/CMakeFiles/my_detect.dir/clean

src/CMakeFiles/my_detect.dir/depend:
	cd /home/pi/src/Competition-of-IOT/detect_in_camera/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/pi/src/Competition-of-IOT/detect_in_camera /home/pi/src/Competition-of-IOT/detect_in_camera/src /home/pi/src/Competition-of-IOT/detect_in_camera/build /home/pi/src/Competition-of-IOT/detect_in_camera/build/src /home/pi/src/Competition-of-IOT/detect_in_camera/build/src/CMakeFiles/my_detect.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/CMakeFiles/my_detect.dir/depend

