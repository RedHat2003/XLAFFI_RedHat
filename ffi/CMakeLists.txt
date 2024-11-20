# ffi/CMakeLists.txt

cmake_minimum_required(VERSION 3.10)
project(identity)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Locate JAX's XLA FFI include directory
set(XLA_FFI_INCLUDE_DIR "/home/redhat_pc/.pyenv/versions/redhat3.13.0/lib/python3.13/site-packages/jaxlib/include")

include_directories(${XLA_FFI_INCLUDE_DIR})

# Define the shared library
add_library(identity SHARED identity.cc)

# Set the library name and version
set_target_properties(identity PROPERTIES PREFIX "" SUFFIX ".so")

# Suppress specific warnings
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(identity PRIVATE -Wno-attributes -Wno-return-type)
endif()

