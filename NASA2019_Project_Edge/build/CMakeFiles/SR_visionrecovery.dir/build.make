# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.13

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
CMAKE_SOURCE_DIR = /mnt/TOSHIBA/NeuroPilot/NASA2019_Project

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /mnt/TOSHIBA/NeuroPilot/NASA2019_Project/build

# Include any dependencies generated for this target.
include CMakeFiles/SR_visionrecovery.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/SR_visionrecovery.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/SR_visionrecovery.dir/flags.make

CMakeFiles/SR_visionrecovery.dir/src/TFLiteSR.cpp.o: CMakeFiles/SR_visionrecovery.dir/flags.make
CMakeFiles/SR_visionrecovery.dir/src/TFLiteSR.cpp.o: ../src/TFLiteSR.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/TOSHIBA/NeuroPilot/NASA2019_Project/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/SR_visionrecovery.dir/src/TFLiteSR.cpp.o"
	../android-ndk-toolchain/bin/aarch64-linux-android-g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/SR_visionrecovery.dir/src/TFLiteSR.cpp.o -c /mnt/TOSHIBA/NeuroPilot/NASA2019_Project/src/TFLiteSR.cpp

CMakeFiles/SR_visionrecovery.dir/src/TFLiteSR.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/SR_visionrecovery.dir/src/TFLiteSR.cpp.i"
	../android-ndk-toolchain/bin/aarch64-linux-android-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/TOSHIBA/NeuroPilot/NASA2019_Project/src/TFLiteSR.cpp > CMakeFiles/SR_visionrecovery.dir/src/TFLiteSR.cpp.i

CMakeFiles/SR_visionrecovery.dir/src/TFLiteSR.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/SR_visionrecovery.dir/src/TFLiteSR.cpp.s"
	../android-ndk-toolchain/bin/aarch64-linux-android-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/TOSHIBA/NeuroPilot/NASA2019_Project/src/TFLiteSR.cpp -o CMakeFiles/SR_visionrecovery.dir/src/TFLiteSR.cpp.s

# Object files for target SR_visionrecovery
SR_visionrecovery_OBJECTS = \
"CMakeFiles/SR_visionrecovery.dir/src/TFLiteSR.cpp.o"

# External object files for target SR_visionrecovery
SR_visionrecovery_EXTERNAL_OBJECTS =

SR_visionrecovery: CMakeFiles/SR_visionrecovery.dir/src/TFLiteSR.cpp.o
SR_visionrecovery: CMakeFiles/SR_visionrecovery.dir/build.make
SR_visionrecovery: CMakeFiles/SR_visionrecovery.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/mnt/TOSHIBA/NeuroPilot/NASA2019_Project/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable SR_visionrecovery"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/SR_visionrecovery.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/SR_visionrecovery.dir/build: SR_visionrecovery

.PHONY : CMakeFiles/SR_visionrecovery.dir/build

CMakeFiles/SR_visionrecovery.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/SR_visionrecovery.dir/cmake_clean.cmake
.PHONY : CMakeFiles/SR_visionrecovery.dir/clean

CMakeFiles/SR_visionrecovery.dir/depend:
	cd /mnt/TOSHIBA/NeuroPilot/NASA2019_Project/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /mnt/TOSHIBA/NeuroPilot/NASA2019_Project /mnt/TOSHIBA/NeuroPilot/NASA2019_Project /mnt/TOSHIBA/NeuroPilot/NASA2019_Project/build /mnt/TOSHIBA/NeuroPilot/NASA2019_Project/build /mnt/TOSHIBA/NeuroPilot/NASA2019_Project/build/CMakeFiles/SR_visionrecovery.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/SR_visionrecovery.dir/depend

