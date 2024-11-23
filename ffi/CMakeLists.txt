cmake_minimum_required(VERSION 3.10)
project(rms_norm)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Automatically include system-wide directories
include_directories(/usr/local/include)

# Define the shared library rms_norm
add_library(rms_norm SHARED rms_norm.cc)
set_target_properties(rms_norm PROPERTIES PREFIX "" SUFFIX ".so")
target_link_libraries(rms_norm xla)  # Link with JAX's XLA library
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(rms_norm PRIVATE -Wno-attributes -Wno-return-type)
endif()

# Define the shared library row_sum
add_library(row_sum SHARED row_sum.cc)
set_target_properties(row_sum PROPERTIES PREFIX "" SUFFIX ".so")
target_link_libraries(row_sum xla)  # Link with JAX's XLA library
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(row_sum PRIVATE -Wno-attributes -Wno-return-type)
endif()

# Define the shared library identity
add_library(identity SHARED identity.cc)
set_target_properties(identity PROPERTIES PREFIX "" SUFFIX ".so")
target_link_libraries(identity xla)  # Link with JAX's XLA library
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(identity PRIVATE -Wno-attributes -Wno-return-type)
endif()
# Define the shared library identity
add_library(sayhi SHARED new.cc)
set_target_properties(sayhi PROPERTIES PREFIX "" SUFFIX ".so")
target_link_libraries(sayhi xla)  # Link with JAX's XLA library
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(sayhi PRIVATE -Wno-attributes -Wno-return-type)
endif()

