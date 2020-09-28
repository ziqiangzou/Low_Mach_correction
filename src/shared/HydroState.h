#ifndef HYDRO_STATE_H_
#define HYDRO_STATE_H_

#include "real_type.h"
#include <Kokkos_Core.hpp>

constexpr int HYDRO_2D_NBVAR=2+2+1;
constexpr int HYDRO_3D_NBVAR=2+3+1;

using HydroState2d = Kokkos::Array<real_t,HYDRO_2D_NBVAR>;
using HydroState3d = Kokkos::Array<real_t,HYDRO_3D_NBVAR>;

#endif // HYDRO_STATE_H_
