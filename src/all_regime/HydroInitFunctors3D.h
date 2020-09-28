#pragma once

#include <limits> // for std::numeric_limits
#ifdef __CUDA_ARCH__
#include <math_constants.h> // for cuda math constants, e.g. CUDART_INF
#endif // __CUDA_ARCH__

#include "shared/kokkos_shared.h"
#include "HydroBaseFunctor3D.h"

// init conditions
#include "shared/BlastParams.h"
#include "shared/PoiseuilleParams.h"

namespace euler_kokkos { namespace all_regime
{

class InitPoiseuilleFunctor3D : HydroBaseFunctor3D
{
public:
    InitPoiseuilleFunctor3D(HydroParams params_,
                            PoiseuilleParams poiseuilleParams_,
                            DataArray Udata_):
        HydroBaseFunctor3D(params_),
        poiseuilleParams(poiseuilleParams_),
        Udata(Udata_) {};

    static void apply(HydroParams params, PoiseuilleParams poiseuilleParams,
                      DataArray Udata, int nbCells)
    {
        InitPoiseuilleFunctor3D functor(params, poiseuilleParams, Udata);
        Kokkos::parallel_for(nbCells, functor);
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const int& index) const
    {
        const real_t ghostWidth = params.ghostWidth;
        const real_t xmin = params.xmin;
        const real_t ymin = params.ymin;
        const real_t zmin = params.zmin;
        const real_t nx = params.nx;
        const real_t ny = params.ny;
        const real_t nz = params.nz;
        const real_t dx = params.dx;
        const real_t dy = params.dy;
        const real_t dz = params.dz;

        int i,j,k;
        index2coord(index, i, j, k, params.isize, params.jsize, params.ksize);
#ifdef USE_MPI
        const int i_mpi = params.myMpiPos[IX];
        const int j_mpi = params.myMpiPos[IY];
        const int k_mpi = params.myMpiPos[IZ];
#else
        const int i_mpi = 0;
        const int j_mpi = 0;
        const int k_mpi = 0;
#endif
        const real_t pressure_gradient = poiseuilleParams.poiseuille_pressure_gradient;
        const real_t p0 = poiseuilleParams.poiseuille_pressure0;
        const real_t gamma0 = params.settings.gamma0;
        const real_t x = xmin + (HALF_F + i + nx*i_mpi-ghostWidth)*dx;
        const real_t y = ymin + (HALF_F + j + ny*j_mpi-ghostWidth)*dy;
        const real_t z = zmin + (HALF_F + k + nz*k_mpi-ghostWidth)*dz;

        const real_t d = poiseuilleParams.poiseuille_density;
        const real_t u = ZERO_F;
        const real_t v = ZERO_F;
        const real_t w = ZERO_F;
        real_t p;
        if (poiseuilleParams.poiseuille_flow_direction == IX)
        {
            p = p0 + (x - xmin) * pressure_gradient;
        }
        else if (poiseuilleParams.poiseuille_flow_direction == IY)
        {
            p = p0 + (y - ymin) * pressure_gradient;
        }
        else
        {
            p = p0 + (z - zmin) * pressure_gradient;
        }

        Udata(i, j, k, ID) = d;
        Udata(i, j, k, IU) = d * u;
        Udata(i, j, k, IV) = d * v;
        Udata(i, j, k, IW) = d * w;
        Udata(i, j, k, IP) = p / (gamma0-ONE_F) + HALF_F * d * (u*u+v*v+w*w);
    }

    PoiseuilleParams poiseuilleParams;
    DataArray Udata;

}; // InitPoiseuilleFunctor3D

class InitRayleighTaylorFunctor3D : HydroBaseFunctor3D
{

public:
    InitRayleighTaylorFunctor3D(HydroParams params_, DataArray Udata_) :
        HydroBaseFunctor3D(params_), Udata(Udata_) {};

    static void apply(HydroParams params,
                      DataArray Udata, int nbCells)
    {
        InitRayleighTaylorFunctor3D functor(params, Udata);
        Kokkos::parallel_for(nbCells, functor);
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(int index) const
    {
        const real_t ghostWidth = params.ghostWidth;
        const real_t xmin = params.xmin;
        const real_t ymin = params.ymin;
        const real_t zmin = params.zmin;
        const real_t xmax = params.xmax;
        const real_t ymax = params.ymax;
        const real_t zmax = params.zmax;
        const real_t dx = params.dx;
        const real_t dy = params.dy;
        const real_t dz = params.dz;
        const real_t nx = params.nx;
        const real_t ny = params.ny;
        const real_t nz = params.nz;

        int i, j, k;
        index2coord(index, i, j, k, params.isize, params.jsize, params.ksize);
#ifdef USE_MPI
        const int i_mpi = params.myMpiPos[IX];
        const int j_mpi = params.myMpiPos[IY];
        const int k_mpi = params.myMpiPos[IZ];
#else
        const int i_mpi = 0;
        const int j_mpi = 0;
        const int k_mpi = 0;
#endif
        const real_t x = xmin + dx/2 + (i+nx*i_mpi-ghostWidth)*dx;
        const real_t y = ymin + dy/2 + (j+ny*j_mpi-ghostWidth)*dy;
        const real_t z = zmin + dz/2 + (k+nz*k_mpi-ghostWidth)*dz;
        const real_t Lx = xmax - xmin;
        const real_t Ly = ymax - ymin;
        const real_t Lz = zmax - zmin;
        const real_t Pi = std::acos(-ONE_F);

        const real_t A = 0.01;
        const real_t p0 = 2.5;
        const real_t rho = (z<=ZERO_F) ? ONE_F : TWO_F;
        const real_t v = A * (1.0+std::cos(TWO_F*Pi*x/Lx))*(1.0+std::cos(TWO_F*Pi*y/Ly))*(1.0+std::cos(TWO_F*Pi*z/Lz))/8.0;

        Udata(i, j, k, ID) = rho;
        Udata(i, j, k, IP) = (p0 + rho*params.settings.g_z*z)/(params.settings.gamma0-ONE_F) + HALF_F*rho*v*v;
        Udata(i, j, k, IU) = ZERO_F;
        Udata(i, j, k, IV) = ZERO_F;
        Udata(i, j, k, IW) = rho * v;
    }

    DataArray Udata;

}; // InitRayleighTaylorFunctor3D


class InitFakeFunctor3D : public HydroBaseFunctor3D
{

public:
    InitFakeFunctor3D(HydroParams params,
                      DataArray Udata) :
        HydroBaseFunctor3D(params), Udata(Udata) {};

    KOKKOS_INLINE_FUNCTION
    void operator()(const int& index) const
    {

        const int isize = params.isize;
        const int jsize = params.jsize;
        const int ksize = params.ksize;

        int i,j,k;
        index2coord(index,i,j,k,isize,jsize,ksize);

        Udata(i  ,j  ,k  , ID) = 0.0;
        Udata(i  ,j  ,k  , IP) = 0.0;
        Udata(i  ,j  ,k  , IU) = 0.0;
        Udata(i  ,j  ,k  , IV) = 0.0;
        Udata(i  ,j  ,k  , IW) = 0.0;

    } // end operator ()

    DataArray Udata;

}; // InitFakeFunctor3D


class InitImplodeFunctor3D : public HydroBaseFunctor3D
{

public:
    InitImplodeFunctor3D(HydroParams params,
                         DataArray Udata) :
        HydroBaseFunctor3D(params), Udata(Udata) {};

    static void apply(HydroParams params,
                      DataArray Udata, int nbCells)
    {
        InitImplodeFunctor3D functor(params, Udata);
        Kokkos::parallel_for(nbCells, functor);
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const int& index) const
    {

        const int isize = params.isize;
        const int jsize = params.jsize;
        const int ksize = params.ksize;
        const int ghostWidth = params.ghostWidth;

#ifdef USE_MPI
        const int i_mpi = params.myMpiPos[IX];
        const int j_mpi = params.myMpiPos[IY];
        const int k_mpi = params.myMpiPos[IZ];
#else
        const int i_mpi = 0;
        const int j_mpi = 0;
        const int k_mpi = 0;
#endif

        const int nx = params.nx;
        const int ny = params.ny;
        const int nz = params.nz;

        const real_t xmin = params.xmin;
        const real_t ymin = params.ymin;
        const real_t zmin = params.zmin;

        const real_t dx = params.dx;
        const real_t dy = params.dy;
        const real_t dz = params.dz;

        const real_t gamma0 = params.settings.gamma0;

        int i,j,k;
        index2coord(index,i,j,k,isize,jsize,ksize);

        real_t x = xmin + dx/2 + (i+nx*i_mpi-ghostWidth)*dx;
        real_t y = ymin + dy/2 + (j+ny*j_mpi-ghostWidth)*dy;
        real_t z = zmin + dz/2 + (k+nz*k_mpi-ghostWidth)*dz;

        real_t tmp = x + y + z;
        if (tmp > 0.5 && tmp < 2.5)
        {
            Udata(i  ,j  ,k  , ID) = 1.0;
            Udata(i  ,j  ,k  , IP) = 1.0/(gamma0-1.0);
            Udata(i  ,j  ,k  , IU) = 0.0;
            Udata(i  ,j  ,k  , IV) = 0.0;
            Udata(i  ,j  ,k  , IW) = 0.0;
        }
        else
        {
            Udata(i  ,j  ,k  , ID) = 0.125;
            Udata(i  ,j  ,k  , IP) = 0.14/(gamma0-1.0);
            Udata(i  ,j  ,k  , IU) = 0.0;
            Udata(i  ,j  ,k  , IV) = 0.0;
            Udata(i  ,j  ,k  , IW) = 0.0;
        }

    } // end operator ()

    DataArray Udata;

}; // InitImplodeFunctor3D


class InitBlastFunctor3D : public HydroBaseFunctor3D
{

public:
    InitBlastFunctor3D(HydroParams params,
                       BlastParams bParams,
                       DataArray Udata) :
        HydroBaseFunctor3D(params), bParams(bParams), Udata(Udata) {};

    static void apply(HydroParams params, BlastParams blastParams,
                      DataArray Udata, int nbCells)
    {
        InitBlastFunctor3D functor(params, blastParams, Udata);
        Kokkos::parallel_for(nbCells, functor);
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const int& index) const
    {

        const int isize = params.isize;
        const int jsize = params.jsize;
        const int ksize = params.ksize;
        const int ghostWidth = params.ghostWidth;

#ifdef USE_MPI
        const int i_mpi = params.myMpiPos[IX];
        const int j_mpi = params.myMpiPos[IY];
        const int k_mpi = params.myMpiPos[IZ];
#else
        const int i_mpi = 0;
        const int j_mpi = 0;
        const int k_mpi = 0;
#endif

        const int nx = params.nx;
        const int ny = params.ny;
        const int nz = params.nz;

        const real_t xmin = params.xmin;
        const real_t ymin = params.ymin;
        const real_t zmin = params.zmin;

        const real_t dx = params.dx;
        const real_t dy = params.dy;
        const real_t dz = params.dz;

        const real_t gamma0 = params.settings.gamma0;

        // blast problem parameters
        const real_t blast_radius      = bParams.blast_radius;
        const real_t radius2           = blast_radius*blast_radius;
        const real_t blast_center_x    = bParams.blast_center_x;
        const real_t blast_center_y    = bParams.blast_center_y;
        const real_t blast_center_z    = bParams.blast_center_z;
        const real_t blast_density_in  = bParams.blast_density_in;
        const real_t blast_density_out = bParams.blast_density_out;
        const real_t blast_pressure_in = bParams.blast_pressure_in;
        const real_t blast_pressure_out= bParams.blast_pressure_out;


        int i,j,k;
        index2coord(index,i,j,k,isize,jsize,ksize);

        real_t x = xmin + dx/2 + (i+nx*i_mpi-ghostWidth)*dx;
        real_t y = ymin + dy/2 + (j+ny*j_mpi-ghostWidth)*dy;
        real_t z = zmin + dz/2 + (k+nz*k_mpi-ghostWidth)*dz;

        real_t d2 =
            (x-blast_center_x)*(x-blast_center_x)+
            (y-blast_center_y)*(y-blast_center_y)+
            (z-blast_center_z)*(z-blast_center_z);

        if (d2 < radius2)
        {
            Udata(i  ,j  ,k  , ID) = blast_density_in;
            Udata(i  ,j  ,k  , IP) = blast_pressure_in/(gamma0-1.0);
            Udata(i  ,j  ,k  , IU) = 0.0;
            Udata(i  ,j  ,k  , IV) = 0.0;
            Udata(i  ,j  ,k  , IW) = 0.0;
        }
        else
        {
            Udata(i  ,j  ,k  , ID) = blast_density_out;
            Udata(i  ,j  ,k  , IP) = blast_pressure_out/(gamma0-1.0);
            Udata(i  ,j  ,k  , IU) = 0.0;
            Udata(i  ,j  ,k  , IV) = 0.0;
            Udata(i  ,j  ,k  , IW) = 0.0;
        }

    } // end operator ()

    BlastParams bParams;
    DataArray Udata;
}; // InitBlastFunctor3D

} // namespace all_regime

} // namespace euler_kokkos
