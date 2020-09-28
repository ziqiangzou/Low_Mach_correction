#
# Kokkos test example : query device information
#

# config file reader / ConfigMap / monitoring timer
add_subdirectory(utils)

# common solver code
add_subdirectory(shared)

# Hydro solver using 2nd order Muscl-Hancock solver
add_subdirectory(muscl)

# Hydro solver using 1st order All-Regime solver
add_subdirectory(all_regime)

#
# euler_kokkos executable
#
set(EXE_NAME "${PROJECT_NAME}")
add_executable(${EXE_NAME}
  main.cpp euler_kokkos.cpp
  )

target_include_directories(${EXE_NAME}
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
  )

target_link_libraries(${EXE_NAME} solver_factory config kokkos dl all_regime muscl shared io monitoring)

if (USE_MPI)
  target_link_libraries(${EXE_NAME} mpiUtils)
endif()
