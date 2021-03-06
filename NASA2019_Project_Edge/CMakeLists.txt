CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

PROJECT("deadline-driven_NASA2019")

SET(NDK_STANDALONE_TOOLCHAIN ../android-ndk-toolchain)

SET(CMAKE_C_COMPILER ${NDK_STANDALONE_TOOLCHAIN}/bin/aarch64-linux-android-gcc)
SET(CMAKE_CXX_COMPILER ${NDK_STANDALONE_TOOLCHAIN}/bin/aarch64-linux-android-g++)
SET(SYSROOT ${NDK_STANDALONE_TOOLCHAIN}/sysroot)
SET(CMAKE_FIND_ROOT_PATH ${NDK_STANDALONE_TOOLCHAIN})
SET(CMAKE_C_FLAGS "${LINUX_FLAGS} -D__ANDROID_LINUX__ -Wno-attributes --sysroot=${SYSROOT}")
SET(CMAKE_CXX_FLAGS "${LINUX_FLAGS} -D__ANDROID_LINUX__ -Wno-attributes --sysroot=${SYSROOT}")
LINK_DIRECTORIES(${CMAKE_CURRENT_LIST_DIR}/libs/arm64)
SET(CMAKE_ANDROID_STL_TYPE c++_static)
SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fPIE -pie")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")

ADD_EXECUTABLE(SR_visionrecovery src/TFLiteSR.cpp)
TARGET_LINK_LIBRARIES(SR_visionrecovery log)
TARGET_LINK_LIBRARIES(SR_visionrecovery neuralnetworks)

ADD_EXECUTABLE(RNN_dataimputation src/TFLiteRNN.cpp)
TARGET_LINK_LIBRARIES(RNN_dataimputation log)
TARGET_LINK_LIBRARIES(RNN_dataimputation neuralnetworks)
