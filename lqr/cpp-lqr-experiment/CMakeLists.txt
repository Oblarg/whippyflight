cmake_minimum_required(VERSION 3.16)
project(cpp_lqr_experiment)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(Eigen3 REQUIRED)
find_package(mp-units REQUIRED)

# Create the library
add_library(cpp_lqr_experiment STATIC
    src/types.cpp
    src/units.cpp
    src/gain_generation.cpp
    src/gain_interpolation.cpp
    src/attitude_control.cpp
)

# Link libraries
target_link_libraries(cpp_lqr_experiment
    Eigen3::Eigen
    mp-units::mp-units
)

# Include directories
target_include_directories(cpp_lqr_experiment PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Compiler flags
target_compile_options(cpp_lqr_experiment PRIVATE
    -O3
    -DNDEBUG
    -Wall
    -Wextra
    -DMP_UNITS_HOSTED=0
)

# Create test executable
add_executable(test_cpp_lqr
    test_cpp_lqr.cpp
)

target_link_libraries(test_cpp_lqr
    cpp_lqr_experiment
) 