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
include CMakeFiles/RNN_dataimputation.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/RNN_dataimputation.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/RNN_dataimputation.dir/flags.make

CMakeFiles/RNN_dataimputation.dir/src/TFLiteRNN.cpp.o: CMakeFiles/RNN_dataimputation.dir/flags.make
CMakeFiles/RNN_dataimputation.dir/src/TFLiteRNN.cpp.o: ../src/TFLiteRNN.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/TOSHIBA/NeuroPilot/NASA2019_Project/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/RNN_dataimputation.dir/src/TFLiteRNN.cpp.o"
	../android-ndk-toolchain/bin/aarch64-linux-android-g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/RNN_dataimputation.dir/src/TFLiteRNN.cpp.o -c /mnt/TOSHIBA/NeuroPilot/NASA2019_Project/src/TFLiteRNN.cpp

CMakeFiles/RNN_dataimputation.dir/src/TFLiteRNN.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/RNN_dataimputation.dir/src/TFLiteRNN.cpp.i"
	../android-ndk-toolchain/bin/aarch64-linux-android-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/TOSHIBA/NeuroPilot/NASA2019_Project/src/TFLiteRNN.cpp > CMakeFiles/RNN_dataimputation.dir/src/TFLiteRNN.cpp.i

CMakeFiles/RNN_dataimputation.dir/src/TFLiteRNN.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/RNN_dataimputation.dir/src/TFLiteRNN.cpp.s"
	../android-ndk-toolchain/bin/aarch64-linux-android-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/TOSHIBA/NeuroPilot/NASA2019_Project/src/TFLiteRNN.cpp -o CMakeFiles/RNN_dataimputation.dir/src/TFLiteRNN.cpp.s

# Object files for target RNN_dataimputation
RNN_dataimputation_OBJECTS = \
"CMakeFiles/RNN_dataimputation.dir/src/TFLiteRNN.cpp.o"

# External object files for target RNN_dataimputation
RNN_dataimputation_EXTERNAL_OBJECTS =

RNN_dataimputation: CMakeFiles/RNN_dataimputation.dir/src/TFLiteRNN.cpp.o
RNN_dataimputation: CMakeFiles/RNN_dataimputation.dir/build.make
RNN_dataimputation: CMakeFiles/RNN_dataimputation.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/mnt/TOSHIBA/NeuroPilot/NASA2019_Project/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable RNN_dataimputation"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/RNN_dataimputation.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/RNN_dataimputation.dir/build: RNN_dataimputation

.PHONY : CMakeFiles/RNN_dataimputation.dir/build

CMakeFiles/RNN_dataimputation.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/RNN_dataimputation.dir/cmake_clean.cmake
.PHONY : CMakeFiles/RNN_dataimputation.dir/clean

CMakeFiles/RNN_dataimputation.dir/depend:
	cd /mnt/TOSHIBA/NeuroPilot/NASA2019_Project/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /mnt/TOSHIBA/NeuroPilot/NASA2019_Project /mnt/TOSHIBA/NeuroPilot/NASA2019_Project /mnt/TOSHIBA/NeuroPilot/NASA2019_Project/build /mnt/TOSHIBA/NeuroPilot/NASA2019_Project/build /mnt/TOSHIBA/NeuroPilot/NASA2019_Project/build/CMakeFiles/RNN_dataimputation.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/RNN_dataimputation.dir/depend
