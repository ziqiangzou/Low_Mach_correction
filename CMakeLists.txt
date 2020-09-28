cmake_minimum_required(VERSION 3.1)

project(euler_kokkos CXX C)

# options
option (USE_DOUBLE "build with double precision" ON)
option (USE_MPI "Activate / want MPI build" OFF)
option (USE_MPI_CUDA_AWARE_ENFORCED "Some MPI cuda-aware implementation are not well detected; use this to enforce" OFF)
option (USE_HDF5 "build HDF5 input/output support" OFF)
option (USE_PNETCDF "build PNETCDF input/output support (MPI required)" OFF)
option (USE_FPE_DEBUG "build with floating point Nan tracing (signal handler)" OFF)

# C++11 is for Kokkos
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)


#
# default local cmake macro repository
#
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

#
# Prevent from build in source tree
#
include(preventBuildInSource)

#
# Init build type: Release, Debug, ...
#
include(initBuildType)

#
# Write a small header with build data, git version, etc...
#
include(write_version)

#####################################################################
# External packages: MPI, ...
#####################################################################


#####################################################################
# Kokkos
#####################################################################
add_subdirectory(externals/kokkos)

# pass Kokkos include directories to our target application
include_directories(${Kokkos_INCLUDE_DIRS_RET})

#####################################################################
# MPI
#####################################################################
if(USE_MPI)
  find_package(MPI REQUIRED)

  add_definitions(-DUSE_MPI)

  message(STATUS "MPI support found")
  message(STATUS "MPI compile flags: " ${MPI_CXX_COMPILE_FLAGS})
  message(STATUS "MPI include path: " ${MPI_CXX_INCLUDE_PATH})
  message(STATUS "MPI link flags path: " ${MPI_CXX_LINK_FLAGS})
  message(STATUS "MPI libraries: " ${MPI_CXX_LIBRARIES})

  if(${KOKKOS_ENABLE_CUDA})
    find_program(OMPI_INFO
      NAMES ompi_info
      HINTS ${MPI_CXX_LIBRARIES}/../bin)

    # Full command line to probe if cuda support in MPI implementation is enabled
    # ompi_info --parsable --all | grep mpi_built_with_cuda_support:value
    if(OMPI_INFO)
      execute_process(COMMAND ${OMPI_INFO} OUTPUT_VARIABLE _output)
      if((_output MATCHES "smcuda") OR (USE_MPI_CUDA_AWARE_ENFORCED))
        message(STATUS "Found OpenMPI with CUDA support built.")
      else((_output MATCHES "smcuda") OR (USE_MPI_CUDA_AWARE_ENFORCED))
        message(WARNING "OpenMPI found, but it is not built with CUDA support.")
        add_definitions(-DMPI_CUDA_AWARE_OFF)
      endif((_output MATCHES "smcuda") OR (USE_MPI_CUDA_AWARE_ENFORCED))
    endif(OMPI_INFO)
  endif(${KOKKOS_ENABLE_CUDA})
endif(USE_MPI)

#####################################################################
# HDF5
#####################################################################
if(USE_HDF5)
  # prefer using parallel HDF5 when build with mpi
  if(USE_MPI)
    set(HDF5_PREFER_PARALLEL TRUE)
  endif(USE_MPI)

  find_package(HDF5 REQUIRED)

  add_definitions(-DUSE_HDF5)
  include_directories(${HDF5_INCLUDE_DIR})
  set(my_hdf5_libs hdf5 hdf5_cpp)
  if(HDF5_IS_PARALLEL)
    add_definitions(-DUSE_HDF5_PARALLEL)
  endif(HDF5_IS_PARALLEL)
endif(USE_HDF5)

#####################################################################
# PNETCDF
#####################################################################
if(USE_PNETCDF AND USE_MPI)
  find_package(PNETCDF REQUIRED)

  add_definitions(-DUSE_PNETCDF)
  include_directories(${PNETCDF_INCLUDE_DIRS})
endif(USE_PNETCDF AND USE_MPI)

#
# common flags
#
if(USE_DOUBLE)
  add_definitions(-DUSE_DOUBLE)
endif(USE_DOUBLE)

if(USE_FPE_DEBUG)
  add_definitions(-DUSE_FPE_DEBUG)
endif(USE_FPE_DEBUG)

##
## Using flags -Wextra, it's too strong for Kokkos, too many warnings
## But -Wall is really a minimum
##
# add_compile_options(-Wall -Wextra -Wconversion -Wshadow -Wnon-virtual-dtor -Wold-style-cast -Wcast-align -Wcast-qual -Wunused -Woverloaded-virtual -pedantic)
# add_compile_options(-Wall -Wextra -Wshadow -Wcast-align -Wunused -Woverloaded-virtual -pedantic -Wsign-compare -Wtype-limits -Wuninitialized)
# add_compile_options(-Wall -Wextra)
add_compile_options(-Wall)

option(BUILD_TESTS "Build tests" OFF)
if(BUILD_TESTS)
  enable_testing()
  add_subdirectory(test)
endif(BUILD_TESTS)

add_subdirectory(src)

############################ PRINT CONFIGURE STATUS ############################
message(STATUS "")
 
#message(STATUS "                    _oo8oo_                                          _oo8oo_                                         _oo8oo_")
#message(STATUS "                   o8888888o                                        o8888888o                                       o8888888o")
#message(STATUS "                   88' . '88                                        88' . '88                                       88' . '88")
#message(STATUS "                   (| -_- |)                                        (| -_- |)                                       (| -_- |)")
#message(STATUS "                   0\\  =  /0                                        0\\  =  /0                                       0\\  =  /0")
#message(STATUS "                 ___/'==='\\___                                    ___/'==='\\___                                   ___/'==='\\___")
#message(STATUS "               .' \\\\|     |// '.                                .' \\\\|     |// '.                               .' \\\\|     |// '.")
#message(STATUS "              / \\\\|||  :  |||// \\                              / \\\\|||  :  |||// \\                             / \\\\|||  :  |||// \\  ")
#message(STATUS "             / _||||| -:- |||||_ \\                            / _||||| -:- |||||_ \\                           / _||||| -:- |||||_ \\ ")
#message(STATUS "            |   | \\\\\\  -  /// |   |                          |   | \\\\\\  -  /// |   |                         |   | \\\\\\  -  /// |   |")
#message(STATUS "            | \\_|  ''\\---/''  |_/ |                          | \\_|  ''\\---/''  |_/ |                         | \\_|  ''\\---/''  |_/ |")
#message(STATUS "            \\  .-\\__  '-'  __/-.  /                          \\  .-\\__  '-'  __/-.  /                         \\  .-\\__  '-'  __/-.  / ")
#message(STATUS "          ___'. .'  /--.--\\  '. .'___                      ___'. .'  /--.--\\  '. .'___                      ___'. .'  /--.--\\  '. .'___")
#message(STATUS "       .'' '<  '.___\\_<|>_/___.'  >' ''.                .'' '<  '.___\\_<|>_/___.'  >' ''.                .'' '<  '.___\\_<|>_/___.'  >' ''.")
#message(STATUS "      | | :  `- \\`.:`\\ _ /`:.`/ -`  : | |              | | :  `- \\`.:`\\ _ /`:.`/ -`  : | |              | | :  `- \\`.:`\\ _ /`:.`/ -`  : | |")
#message(STATUS "      \\  \\ `-.   \\_ __\\ /__ _/   .-` /  /              \\  \\ `-.   \\_ __\\ /__ _/   .-` /  /              \\  \\ `-.   \\_ __\\ /__ _/   .-` /  /")
#message(STATUS "  =====`-.____`.___ \\_____/ ___.`____.-`=====      =====`-.____`.___ \\_____/ ___.`____.-`=====      =====`-.____`.___ \\_____/ ___.`____.-`=====")
#message(STATUS "                      ")  
#message(STATUS "                      ")  
#message(STATUS "                      ")  
#message(STATUS "                      ")  
#message(STATUS "  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
#message(STATUS "                      ")  
#message(STATUS "                佛祖保佑                                             永不宕机                                         永无bug ")



message(STATUS "")
message(STATUS "****************** ${PROJECT_NAME} Settings ******************")

message(STATUS "DOUBLE precision  : ${USE_DOUBLE}")
message(STATUS "MPI       enabled : ${USE_MPI}")
message(STATUS "CUDA      enabled : ${KOKKOS_ENABLE_CUDA}")
message(STATUS "HDF5      enabled : ${USE_HDF5}")
message(STATUS "PNETCDF   enabled : ${USE_PNETCDF}")
message(STATUS "HWLOC     enabled : ${KOKKOS_ENABLE_HWLOC}")
message(STATUS "FPE_DEBUG enabled : ${USE_FPE_DEBUG}")

message(STATUS "****************** *************** ******************")
