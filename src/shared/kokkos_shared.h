#ifndef KOKKOS_SHARED_H_
#define KOKKOS_SHARED_H_

#include <Kokkos_Core.hpp>
#include <Kokkos_Parallel.hpp>
#include <Kokkos_View.hpp>
#include <Kokkos_Macros.hpp>
#include <impl/Kokkos_Error.hpp>

#include "shared/real_type.h"
#include "shared/utils.h"


#ifdef KOKKOS_ENABLE_CUDA
#include <cuda.h>
#endif

using device = Kokkos::DefaultExecutionSpace;

enum KokkosLayout
{
    KOKKOS_LAYOUT_LEFT,
    KOKKOS_LAYOUT_RIGHT
};

// last index is hydro variable
// n-1 first indexes are space (i,j,k,....)
using DataArray2d      = Kokkos::View<real_t***, device>;
using DataArray2dConst = DataArray2d::const_type;
using DataArray2dHost  = DataArray2d::HostMirror;

using DataArray3d      = Kokkos::View<real_t****, device>;
using DataArray3dConst = DataArray3d::const_type;
using DataArray3dHost  = DataArray3d::HostMirror;

template <int dim>
struct DataArrays
{
};

template <>
struct DataArrays<2>
{
    using DataArray      = Kokkos::View<real_t***, device>;
    using DataArrayConst = DataArray::const_type;
    using DataArrayHost  = DataArray::HostMirror;
};

template <>
struct DataArrays<3>
{
    using DataArray      = Kokkos::View<real_t****, device>;
    using DataArrayConst = DataArray::const_type;
    using DataArrayHost  = DataArray::HostMirror;
};

/**
 * Retrieve cartesian coordinate from index, using memory layout information.
 *
 * for each execution space define a prefered layout.
 * Prefer left layout  for CUDA execution space.
 * Prefer right layout for OpenMP execution space.
 *
 * These function will eventually disappear.
 * We still need then as long as parallel_reduce does not accept MDRange policy.
 */

/* 2D */

KOKKOS_INLINE_FUNCTION
void index2coord(int index, int &i, int &j, int Nx, int Ny)
{
    UNUSED(Nx);
    UNUSED(Ny);

#ifdef KOKKOS_ENABLE_CUDA
    j = index / Nx;
    i = index - j*Nx;
#else
    i = index / Ny;
    j = index - i*Ny;
#endif
}

KOKKOS_INLINE_FUNCTION
int coord2index(int i, int j, int Nx, int Ny)
{
    UNUSED(Nx);
    UNUSED(Ny);
#ifdef KOKKOS_ENABLE_CUDA
    return i + Nx*j; // left layout
#else
    return j + Ny*i; // right layout
#endif
}

/* 3D */

KOKKOS_INLINE_FUNCTION
void index2coord(int index,
                 int &i, int &j, int &k,
                 int Nx, int Ny, int Nz)
{
    UNUSED(Nx);
    UNUSED(Nz);
#ifdef KOKKOS_ENABLE_CUDA
    int NxNy = Nx*Ny;
    k = index / NxNy;
    j = (index - k*NxNy) / Nx;
    i = index - j*Nx - k*NxNy;
#else
    int NyNz = Ny*Nz;
    i = index / NyNz;
    j = (index - i*NyNz) / Nz;
    k = index - j*Nz - i*NyNz;
#endif
}

KOKKOS_INLINE_FUNCTION
int coord2index(int i,  int j,  int k,
                int Nx, int Ny, int Nz)
{
    UNUSED(Nx);
    UNUSED(Nz);
#ifdef KOKKOS_ENABLE_CUDA
    return i + Nx*j + Nx*Ny*k; // left layout
#else
    return k + Nz*j + Nz*Ny*i; // right layout
#endif
}


#endif // KOKKOS_SHARED_H_
