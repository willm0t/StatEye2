cmake_minimum_required(VERSION 3.24)
project(StatEye2)

set(CMAKE_CXX_STANDARD 17)

set(OpenCV_DIR /opt/homebrew/opt/opencv/lib/cmake/opencv4)

find_package(OpenCV 4 REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})


add_executable(StatEye2 main.cpp VideoTracker.cpp VideoTracker.h Setup.cpp Setup.h)
target_link_libraries(StatEye2 ${OpenCV_LIBS})
