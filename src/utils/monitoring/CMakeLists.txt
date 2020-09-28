list(APPEND monitoring_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/SimpleTimer.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/SimpleTimer.h
  )

if (KOKKOS_ENABLE_OPENMP)
  list(APPEND monitoring_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/OpenMPTimer.cpp ${CMAKE_CURRENT_SOURCE_DIR}/OpenMPTimer.h)
endif (KOKKOS_ENABLE_OPENMP)

if (KOKKOS_ENABLE_CUDA)
  list(APPEND monitoring_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/CudaTimer.h)
endif (KOKKOS_ENABLE_CUDA)

add_library(monitoring
  ${monitoring_SRCS}
  )
