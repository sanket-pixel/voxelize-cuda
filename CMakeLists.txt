cmake_minimum_required(VERSION 1.10)
project(voxelize_cuda)
set(arch ${CMAKE_HOST_SYSTEM_PROCESSOR})

if(${arch}     STREQUAL  "aarch64")
elseif(${arch} STREQUAL "x86_64")
  set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)
endif() 

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS_RELEASE "-std=c++14 -Wextra -Wall -Wno-deprecated-declarations -O3")
# set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -gencode arch=compute_87,code=compute_87)


find_package(CUDA REQUIRED)
if(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
  message("Using Debug Mode")
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -g -G --ptxas-options=-v)
endif()

file(GLOB_RECURSE SOURCE_FILES 
  src/*.cu 
  src/*.cpp
)

# Check if PCL is available
find_package(PCL 1.8.1 QUIET)
if(PCL_FOUND)
    find_package(Boost REQUIRED)
    include_directories(${Boost_INCLUDE_DIRS})
    link_directories(${Boost_LIBRARIES})
    message(STATUS "PCL found. Including PCL-related components.")
    include_directories(${PCL_INCLUDE_DIRS})
    link_directories(${PCL_LIBRARY_DIRS})
    add_definitions(${PCL_DEFINITIONS})
    # Define USE_PCL preprocessor directive
    add_definitions(-DUSE_PCL)
else()
    # Filter out visualizer.cpp if PCL is not found
    list(FILTER SOURCE_FILES EXCLUDE REGEX "src/visualizer.cpp")
endif()


include_directories(
  ${CUDA_INCLUDE_DIRS}
  include
)

cuda_add_executable(${PROJECT_NAME} main.cpp ${SOURCE_FILES} )
# Add the header files directly to the target
if(PCL_FOUND)
    # Add the include directory for PCL
    target_include_directories(${PROJECT_NAME} PRIVATE ${PCL_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})
    # Link PCL libraries only if PCL is found
    target_link_libraries(${PROJECT_NAME} ${PCL_LIBRARIES}  ${Boost_LIBRARIES})
endif()
