cmake_minimum_required(VERSION 3.1)
project(QuEST VERSION 0.1 LANGUAGES CXX)

# Location of custom detection scripts
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/modules")

# for debugging un comment
set(CMAKE_BUILD_TYPE Debug)

# Set RELEASE as default build type
if (NOT CMAKE_BUILD_TYPE)
  set (CMAKE_BUILD_TYPE RELEASE CACHE STRING
      "Choose the type of build, options are: None Debug Release."
      FORCE)
endif (NOT CMAKE_BUILD_TYPE)

# Enable warning flags if available
include("${PROJECT_SOURCE_DIR}/cmake/flag_check.cmake")
enable_cxx_compiler_flag_if_supported("-Wall")
enable_cxx_compiler_flag_if_supported("-Wextra")
enable_cxx_compiler_flag_if_supported("-fdiagnostics-color=auto")
enable_cxx_compiler_flag_if_supported("-pedantic")

# Require a C++14 standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(quest "")
target_compile_features(quest PRIVATE ${CPP_FEATURES})

find_package(Eigen3 3.2.2 REQUIRED)
target_include_directories(quest PUBLIC ${EIGEN3_INCLUDE_DIR})

find_package(Boost 1.55 COMPONENTS program_options REQUIRED)
target_include_directories(quest PUBLIC ${Boost_INCLUDE_DIR})

include("${PROJECT_SOURCE_DIR}/cmake/set_git_hash.cmake")
set_git_macros(quest)

include(src/CMakeLists.txt)
include(test/CMakeLists.txt)
include(tools/CMakeLists.txt)
