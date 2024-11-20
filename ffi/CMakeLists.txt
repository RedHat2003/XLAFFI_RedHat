# ffi/CMakeLists.txt

cmake_minimum_required(VERSION 3.10)
project(rms_norm)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python (for JAX's include paths)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Locate JAX's XLA FFI include directory
set(XLA_FFI_INCLUDE_DIR "/home/redhat_pc/.pyenv/versions/redhat3.13.0/lib/python3.13/site-packages/jaxlib/include")

include_directories(${XLA_FFI_INCLUDE_DIR})

# Define the shared library
add_library(rms_norm SHARED rms_norm.cc)

# Set the library name and version
set_target_properties(rms_norm PROPERTIES PREFIX "" SUFFIX ".so")

# (Optional) Suppress specific compiler warnings
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(rms_norm PRIVATE -Wno-attributes -Wno-return-type)
endif()
# Define the shared library
add_library(row_sum SHARED row_sum.cc)

# Set the library name and version
set_target_properties(row_sum PROPERTIES PREFIX "" SUFFIX ".so")

# (Optional) Suppress specific compiler warnings
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(row_sum PRIVATE -Wno-attributes -Wno-return-type)
endif()
#_________________________________________
add_library(identity SHARED identity.cc)
set_target_properties(identity PROPERTIES PREFIX "" SUFFIX ".so")
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(identity PRIVATE -Wno-attributes -Wno-return-type)
endif()
