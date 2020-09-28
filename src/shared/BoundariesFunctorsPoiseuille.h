#pragma once

#include "shared/kokkos_shared.h"
#include "shared/HydroParams.h"
#include "shared/PoiseuilleParams.h"

template <FaceIdType faceId>
class MakeBoundariesFunctor2D_Poiseuille
{
public:
    MakeBoundariesFunctor2D_Poiseuille(HydroParams params_,
                                       PoiseuilleParams poiseuilleParams_,
                                       DataArray2d Udata_)
        : params(params_)
        , poiseuilleParams(poiseuilleParams_)
        , Udata(Udata_) {};

    static void apply(HydroParams params,
                      PoiseuilleParams poiseuilleParams,
                      DataArray2d Udata,
                      int nbCells)
    {
        MakeBoundariesFunctor2D_Poiseuille<faceId> functor(params, poiseuilleParams, Udata);
        Kokkos::parallel_for(nbCells, functor);
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const int& index) const
    {
        const int nx = params.nx;
        const int ny = params.ny;

        const int ghostWidth = params.ghostWidth;

        const int imin = params.imin;
        const int imax = params.imax;

        const int jmin = params.jmin;
        const int jmax = params.jmax;

        const real_t dx = params.dx;
        const real_t dy = params.dy;
        const real_t xmin = params.xmin;
        const real_t ymin = params.ymin;

#ifdef USE_MPI
        const int i_mpi = params.myMpiPos[IX];
        const int j_mpi = params.myMpiPos[IY];
#else
        const int i_mpi = 0;
        const int j_mpi = 0;
#endif
        const real_t gamma0 = params.settings.gamma0;
        const real_t pressure_gradient = poiseuilleParams.poiseuille_pressure_gradient;
        const real_t p0 = poiseuilleParams.poiseuille_pressure0;

        if (faceId == FACE_XMIN)
        {
            const int j = index / ghostWidth;
            const int i = index - j*ghostWidth;

            if(j >= jmin && j <= jmax &&
               i >= 0    && i <ghostWidth)
            {
                if (poiseuilleParams.poiseuille_flow_direction == IX)
                {
                    const real_t x = xmin + (HALF_F + i + nx*i_mpi-ghostWidth)*dx;
                    const int i0 = ghostWidth;

                    const real_t d = Udata(i0, j, ID);
                    const real_t u = Udata(i0, j, IU) / d;
                    const real_t v = Udata(i0, j, IV) / d;
                    const real_t p = p0 + (x - xmin) * pressure_gradient;

                    Udata(i, j, ID) = d;
                    Udata(i, j, IU) = d * u;
                    Udata(i, j, IV) = d * v;
                    Udata(i, j, IP) = p / (gamma0 - ONE_F) + HALF_F * d * (u*u + v*v);
                }
                else
                {
                    const int i0 = 2*ghostWidth-1-i;
                    Udata(i, j, ID) =   Udata(i0, j, ID);
                    Udata(i, j, IU) = - Udata(i0, j, IU);
                    Udata(i, j, IV) = - Udata(i0, j, IV);
                    Udata(i, j, IP) =   Udata(i0, j, IP);
                }
            }
        }

        if (faceId == FACE_XMAX)
        {
            const int j = index / ghostWidth;
            const int i = index - j*ghostWidth + nx + ghostWidth;

            if(j >= jmin          && j <= jmax             &&
               i >= nx+ghostWidth && i <= nx+2*ghostWidth-1)
            {
                if (poiseuilleParams.poiseuille_flow_direction == IX)
                {
                    const real_t x = xmin + (HALF_F + i + nx*i_mpi-ghostWidth)*dx;
                    const int i0 = nx+ghostWidth-1;

                    const real_t d = Udata(i0, j, ID);
                    const real_t u = Udata(i0, j, IU) / d;
                    const real_t v = Udata(i0, j, IV) / d;
                    const real_t p = p0 + (x - xmin) * pressure_gradient;

                    Udata(i, j, ID) = d;
                    Udata(i, j, IU) = d * u;
                    Udata(i, j, IV) = d * v;
                    Udata(i, j, IP) = p / (gamma0 - ONE_F) + HALF_F * d * (u*u + v*v);
                }
                else
                {
                    const int i0 = 2*nx+2*ghostWidth-1-i;
                    Udata(i, j, ID) =   Udata(i0, j, ID);
                    Udata(i, j, IU) = - Udata(i0, j, IU);
                    Udata(i, j, IV) = - Udata(i0, j, IV);
                    Udata(i, j, IP) =   Udata(i0, j, IP);
                }
            }
        }

        if (faceId == FACE_YMIN)
        {
            const int i = index / ghostWidth;
            const int j = index - i*ghostWidth;

            if(i >= imin && i <= imax    &&
               j >= 0    && j <ghostWidth)
            {
                if (poiseuilleParams.poiseuille_flow_direction == IY)
                {
                    const real_t y = ymin + (HALF_F + j + ny*j_mpi-ghostWidth)*dy;
                    const int j0 = ghostWidth;

                    const real_t d = Udata(i, j0, ID);
                    const real_t u = Udata(i, j0, IU) / d;
                    const real_t v = Udata(i, j0, IV) / d;
                    const real_t p = p0 + (y - ymin) * pressure_gradient;

                    Udata(i, j, ID) = d;
                    Udata(i, j, IU) = d * u;
                    Udata(i, j, IV) = d * v;
                    Udata(i, j, IP) = p / (gamma0 - ONE_F) + HALF_F * d * (u*u + v*v);
                }
                else
                {
                    const int j0 = 2*ghostWidth-1-j;
                    Udata(i, j, ID) =   Udata(i, j0, ID);
                    Udata(i, j, IU) = - Udata(i, j0, IU);
                    Udata(i, j, IV) = - Udata(i, j0, IV);
                    Udata(i, j, IP) =   Udata(i, j0, IP);
                }
            }
        } // end FACE_YMIN

        if (faceId == FACE_YMAX)
        {
            const int i = index / ghostWidth;
            const int j = index - i*ghostWidth + ny+ghostWidth;

            if(i >= imin          && i <= imax              &&
               j >= ny+ghostWidth && j <= ny+2*ghostWidth-1)
            {
                if (poiseuilleParams.poiseuille_flow_direction == IY)
                {
                    const real_t y = ymin + (HALF_F + j + ny*j_mpi-ghostWidth)*dy;
                    const int j0 = ny+ghostWidth-1;

                    const real_t d = Udata(i, j0, ID);
                    const real_t u = Udata(i, j0, IU) / d;
                    const real_t v = Udata(i, j0, IV) / d;
                    const real_t p = p0 + (y - ymin) * pressure_gradient;

                    Udata(i, j, ID) = d;
                    Udata(i, j, IU) = d * u;
                    Udata(i, j, IV) = d * v;
                    Udata(i, j, IP) = p / (gamma0 - ONE_F) + HALF_F * d * (u*u + v*v);
                }
                else
                {
                    const int j0 = 2*ny+2*ghostWidth-1-j;
                    Udata(i, j, ID) =   Udata(i, j0, ID);
                    Udata(i, j, IU) = - Udata(i, j0, IU);
                    Udata(i, j, IV) = - Udata(i, j0, IV);
                    Udata(i, j, IP) =   Udata(i, j0, IP);
                }
            }
        } // end FACE_YMAX
    }

    HydroParams params;
    PoiseuilleParams poiseuilleParams;
    DataArray2d Udata;
};



template <FaceIdType faceId>
class MakeBoundariesFunctor3D_Poiseuille
{
public:
    MakeBoundariesFunctor3D_Poiseuille(HydroParams params_,
                                       PoiseuilleParams poiseuilleParams_,
                                       DataArray3d Udata_) :
        params(params_), poiseuilleParams(poiseuilleParams_), Udata(Udata_) {};

    static void apply(HydroParams params,
                      PoiseuilleParams poiseuilleParams,
                      DataArray3d Udata,
                      int nbCells)
    {
        MakeBoundariesFunctor3D_Poiseuille<faceId> functor(params, poiseuilleParams, Udata);
        Kokkos::parallel_for(nbCells, functor);
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const int& index) const
    {
        const int nx = params.nx;
        const int ny = params.ny;
        const int nz = params.nz;

        const int nbvar = params.nbvar;
        const int ghostWidth = params.ghostWidth;

        const int imin = params.imin;
        const int jmin = params.jmin;
        const int kmin = params.kmin;

        const int imax = params.imax;
        const int jmax = params.jmax;
        const int kmax = params.kmax;

        const int isize = params.isize;
        const int jsize = params.jsize;

        const real_t dx = params.dx;
        const real_t dy = params.dy;
        const real_t dz = params.dz;
        const real_t xmin = params.xmin;
        const real_t ymin = params.ymin;
        const real_t zmin = params.zmin;

#ifdef USE_MPI
        const int i_mpi = params.myMpiPos[IX];
        const int j_mpi = params.myMpiPos[IY];
        const int k_mpi = params.myMpiPos[IZ];
#else
        const int i_mpi = 0;
        const int j_mpi = 0;
        const int k_mpi = 0;
#endif
        const real_t gamma0 = params.settings.gamma0;
        const real_t pressure_gradient = poiseuilleParams.poiseuille_pressure_gradient;
        const real_t p0 = poiseuilleParams.poiseuille_pressure0;

        if (faceId == FACE_XMIN)
        {
            const int k = index / (ghostWidth*jsize);
            const int j = (index - k*ghostWidth*jsize) / ghostWidth;
            const int i = index - j*ghostWidth - k*ghostWidth*jsize;

            if(k >= kmin && k <= kmax &&
               j >= jmin && j <= jmax &&
               i >= 0    && i <ghostWidth)
            {
                if (poiseuilleParams.poiseuille_flow_direction == IX)
                {
                    const real_t x = xmin + (HALF_F + i + nx*i_mpi-ghostWidth)*dx;
                    const int i0 = ghostWidth;

                    const real_t d = Udata(i0, j, k, ID);
                    const real_t u = Udata(i0, j, k, IU) / d;
                    const real_t v = Udata(i0, j, k, IV) / d;
                    const real_t w = Udata(i0, j, k, IW) / d;
                    const real_t p = p0 + (x - xmin) * pressure_gradient;

                    Udata(i, j, k, ID) = d;
                    Udata(i, j, k, IU) = d * u;
                    Udata(i, j, k, IV) = d * v;
                    Udata(i, j, k, IW) = d * w;
                    Udata(i, j, k, IP) = p / (gamma0 - ONE_F) + HALF_F * d * (u*u + v*v + w*w);
                }
                else if (poiseuilleParams.poiseuille_normal_direction == IX)
                {
                    const int i0 = 2*ghostWidth-1-i;
                    Udata(i, j, k, ID) =   Udata(i0, j, k, ID);
                    Udata(i, j, k, IU) = - Udata(i0, j, k, IU);
                    Udata(i, j, k, IV) = - Udata(i0, j, k, IV);
                    Udata(i, j, k, IW) = - Udata(i0, j, k, IW);
                    Udata(i, j, k, IP) =   Udata(i0, j, k, IP);
                }
                else
                {
                    const int i0 = i+nx;
                    for (int iVar=0; iVar<nbvar; ++iVar)
                    {
                        Udata(i, j, k, iVar) =  Udata(i0, j, k, iVar);
                    }
                }
            }
        }

        if (faceId == FACE_XMAX)
        {
            const int k = index / (ghostWidth*jsize);
            const int j = (index - k*ghostWidth*jsize) / ghostWidth;
            const int i = index - j*ghostWidth - k*ghostWidth*jsize + nx + ghostWidth;

            if(k >= kmin          && k <= kmax &&
               j >= jmin          && j <= jmax &&
               i >= nx+ghostWidth && i <= nx+2*ghostWidth-1)
            {
                if (poiseuilleParams.poiseuille_flow_direction == IX)
                {
                    const real_t x = xmin + (HALF_F + i + nx*i_mpi-ghostWidth)*dx;
                    const int i0 = nx+ghostWidth-1;

                    const real_t d = Udata(i0, j, k, ID);
                    const real_t u = Udata(i0, j, k, IU) / d;
                    const real_t v = Udata(i0, j, k, IV) / d;
                    const real_t w = Udata(i0, j, k, IW) / d;
                    const real_t p = p0 + (x - xmin) * pressure_gradient;

                    Udata(i, j, k, ID) = d;
                    Udata(i, j, k, IU) = d * u;
                    Udata(i, j, k, IV) = d * v;
                    Udata(i, j, k, IW) = d * w;
                    Udata(i, j, k, IP) = p / (gamma0 - ONE_F) + HALF_F * d * (u*u + v*v + w*w);
                }
                else if (poiseuilleParams.poiseuille_normal_direction == IX)
                {
                    const int i0 = 2*nx+2*ghostWidth-1-i;
                    Udata(i, j, k, ID) =   Udata(i0, j, k, ID);
                    Udata(i, j, k, IU) = - Udata(i0, j, k, IU);
                    Udata(i, j, k, IV) = - Udata(i0, j, k, IV);
                    Udata(i, j, k, IW) = - Udata(i0, j, k, IW);
                    Udata(i, j, k, IP) =   Udata(i0, j, k, IP);
                }
                else
                {
                    const int i0 = i-nx;
                    for (int iVar=0; iVar<nbvar; ++iVar)
                    {
                        Udata(i, j, k, iVar) =  Udata(i0, j, k, iVar);
                    }
                }
            }
        }

        if (faceId == FACE_YMIN)
        {
            const int k = index / (isize*ghostWidth);
            const int j = (index - k*isize*ghostWidth) / isize;
            const int i = index - j*isize - k*isize*ghostWidth;

            if(k >= kmin && k <= kmax       &&
               j >= 0    && j <  ghostWidth &&
               i >= imin && i <= imax)
            {
                if (poiseuilleParams.poiseuille_flow_direction == IY)
                {
                    const real_t y = ymin + (HALF_F + j + ny*j_mpi-ghostWidth)*dy;
                    const int j0 = ghostWidth;

                    const real_t d = Udata(i, j0, k, ID);
                    const real_t u = Udata(i, j0, k, IU) / d;
                    const real_t v = Udata(i, j0, k, IV) / d;
                    const real_t w = Udata(i, j0, k, IW) / d;
                    const real_t p = p0 + (y - ymin) * pressure_gradient;

                    Udata(i, j, k, ID) = d;
                    Udata(i, j, k, IU) = d * u;
                    Udata(i, j, k, IV) = d * v;
                    Udata(i, j, k, IW) = d * w;
                    Udata(i, j, k, IP) = p / (gamma0 - ONE_F) + HALF_F * d * (u*u + v*v + w*w);
                }
                else if (poiseuilleParams.poiseuille_normal_direction == IY)
                {
                    const int j0 = 2*ghostWidth-1-j;
                    Udata(i, j, k, ID) =   Udata(i, j0, k, ID);
                    Udata(i, j, k, IU) = - Udata(i, j0, k, IU);
                    Udata(i, j, k, IV) = - Udata(i, j0, k, IV);
                    Udata(i, j, k, IW) = - Udata(i, j0, k, IW);
                    Udata(i, j, k, IP) =   Udata(i, j0, k, IP);
                }
                else
                {
                    const int j0 = j+ny;
                    for (int iVar=0; iVar<nbvar; ++iVar)
                    {
                        Udata(i, j, k, iVar) = Udata(i, j0, k, iVar);
                    }
                }
            }
        } // end FACE_YMIN

        if (faceId == FACE_YMAX)
        {
            const int k = index / (isize*ghostWidth);
            int j = (index - k*isize*ghostWidth) / isize;
            const int i = index - j*isize - k*isize*ghostWidth;
            j += ny+ghostWidth;

            if(k >= kmin           && k <= kmax              &&
               j >= ny+ghostWidth  && j <= ny+2*ghostWidth-1 &&
               i >= imin           && i <= imax)
            {
                if (poiseuilleParams.poiseuille_flow_direction == IY)
                {
                    const real_t y = ymin + (HALF_F + j + ny*j_mpi-ghostWidth)*dy;
                    const int j0 = ny+ghostWidth-1;

                    const real_t d = Udata(i, j0, k, ID);
                    const real_t u = Udata(i, j0, k, IU) / d;
                    const real_t v = Udata(i, j0, k, IV) / d;
                    const real_t w = Udata(i, j0, k, IW) / d;
                    const real_t p = p0 + (y - ymin) * pressure_gradient;

                    Udata(i, j, k, ID) = d;
                    Udata(i, j, k, IU) = d * u;
                    Udata(i, j, k, IV) = d * v;
                    Udata(i, j, k, IW) = d * w;
                    Udata(i, j, k, IP) = p / (gamma0 - ONE_F) + HALF_F * d * (u*u + v*v + w*w);
                }
                else if (poiseuilleParams.poiseuille_normal_direction == IY)
                {
                    const int j0 = 2*ny+2*ghostWidth-1-j;
                    Udata(i, j, k, ID) =   Udata(i, j0, k, ID);
                    Udata(i, j, k, IU) = - Udata(i, j0, k, IU);
                    Udata(i, j, k, IV) = - Udata(i, j0, k, IV);
                    Udata(i, j, k, IW) = - Udata(i, j0, k, IW);
                    Udata(i, j, k, IP) =   Udata(i, j0, k, IP);
                }
                else
                {
                    const int j0 = j-ny;
                    for (int iVar=0; iVar<nbvar; ++iVar)
                    {
                        Udata(i, j, k, iVar) = Udata(i, j0, k, iVar);
                    }
                }
            }
        } // end FACE_YMAX

        if (faceId == FACE_ZMIN)
        {
            const int k = index / (isize*jsize);
            const int j = (index - k*isize*jsize) / isize;
            const int i = index - j*isize - k*isize*jsize;

            if(k >= 0    && k <  ghostWidth &&
               j >= jmin && j <= jmax       &&
               i >= imin && i <= imax)
            {
                if (poiseuilleParams.poiseuille_flow_direction == IZ)
                {
                    const real_t z = zmin + (HALF_F + k + nz*k_mpi-ghostWidth)*dz;
                    const int k0 = ghostWidth;

                    const real_t d = Udata(i, j, k0, ID);
                    const real_t u = Udata(i, j, k0, IU) / d;
                    const real_t v = Udata(i, j, k0, IV) / d;
                    const real_t w = Udata(i, j, k0, IW) / d;
                    const real_t p = p0 + (z - zmin) * pressure_gradient;

                    Udata(i, j, k, ID) = d;
                    Udata(i, j, k, IU) = d * u;
                    Udata(i, j, k, IV) = d * v;
                    Udata(i, j, k, IW) = d * w;
                    Udata(i, j, k, IP) = p / (gamma0 - ONE_F) + HALF_F * d * (u*u + v*v + w*w);
                }
                else if (poiseuilleParams.poiseuille_normal_direction == IZ)
                {
                    const int k0 = 2*ghostWidth-1-k;
                    Udata(i, j, k, ID) =   Udata(i, j, k0, ID);
                    Udata(i, j, k, IU) = - Udata(i, j, k0, IU);
                    Udata(i, j, k, IV) = - Udata(i, j, k0, IV);
                    Udata(i, j, k, IW) = - Udata(i, j, k0, IW);
                    Udata(i, j, k, IP) =   Udata(i, j, k0, IP);
                }
                else
                {
                    for (int iVar=0; iVar<nbvar; iVar++)
                    {
                        const int k0 = k+nz;
                        Udata(i, j, k, iVar) = Udata(i, j, k0, iVar);
                    }
                }
            } // end zmin
        }

        if (faceId == FACE_ZMAX)
        {
            int k = index / (isize*jsize);
            const int j = (index - k*isize*jsize) / isize;
            const int i = index - j*isize - k*isize*jsize;
            k += nz + ghostWidth;

            if(k >= nz+ghostWidth && k <= nz+2*ghostWidth-1 &&
               j >= jmin          && j <= jmax              &&
               i >= imin          && i <= imax)
            {
                if (poiseuilleParams.poiseuille_flow_direction == IZ)
                {
                    const real_t z = zmin + (HALF_F + k + nz*k_mpi-ghostWidth)*dz;
                    const int k0 = nz+ghostWidth-1;

                    const real_t d = Udata(i, j, k0, ID);
                    const real_t u = Udata(i, j, k0, IU) / d;
                    const real_t v = Udata(i, j, k0, IV) / d;
                    const real_t w = Udata(i, j, k0, IW) / d;
                    const real_t p = p0 + (z - zmin) * pressure_gradient;

                    Udata(i, j, k, ID) = d;
                    Udata(i, j, k, IU) = d * u;
                    Udata(i, j, k, IV) = d * v;
                    Udata(i, j, k, IW) = d * w;
                    Udata(i, j, k, IP) = p / (gamma0 - ONE_F) + HALF_F * d * (u*u + v*v + w*w);
                }
                else if (poiseuilleParams.poiseuille_normal_direction == IZ)
                {
                    const int k0 = 2*nz+2*ghostWidth-1-k;
                    Udata(i, j, k, ID) =   Udata(i, j, k0, ID);
                    Udata(i, j, k, IU) = - Udata(i, j, k0, IU);
                    Udata(i, j, k, IV) = - Udata(i, j, k0, IV);
                    Udata(i, j, k, IW) = - Udata(i, j, k0, IW);
                    Udata(i, j, k, IP) =   Udata(i, j, k0, IP);
                }
                else
                {
                    for (int iVar=0; iVar<nbvar; iVar++)
                    {
                        const int k0 = k-nz;
                        Udata(i, j, k, iVar) = Udata(i, j, k0, iVar);
                    }
                }
            } // end zmax
        }
    }

    HydroParams params;
    PoiseuilleParams poiseuilleParams;
    DataArray3d Udata;
};
