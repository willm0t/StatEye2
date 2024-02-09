cmake_minimum_required(VERSION 3.10)
project(StatEye2_0)

set(CMAKE_CXX_STANDARD 11)

# Set OpenCV_DIR to the path provided by pkg-config
set(OpenCV_DIR /opt/homebrew/opt/opencv/lib/cmake/opencv4)

# Find the OpenCV package
find_package(OpenCV REQUIRED)

# Include OpenCV directories
include_directories(${OpenCV_INCLUDE_DIRS})

# Define your executable and link OpenCV libraries
add_executable(StatEye2_0 main.cpp)
target_link_libraries(StatEye2_0 ${OpenCV_LIBS})
