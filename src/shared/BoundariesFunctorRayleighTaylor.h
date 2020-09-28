#pragma once

#include "all_regime/HydroBaseFunctor2D.h"
#include "all_regime/HydroBaseFunctor3D.h"
#include "shared/HydroParams.h"    // for HydroParams
#include "shared/kokkos_shared.h"  // for Data arrays

namespace euler_kokkos
{

template <FaceIdType faceId>
class MakeBoundariesFunctor2D_RayleighTaylor : all_regime::HydroBaseFunctor2D
{
public:
    MakeBoundariesFunctor2D_RayleighTaylor(HydroParams params_, DataArray2d Udata_) :
        all_regime::HydroBaseFunctor2D(params_), Udata(Udata_) {};

    static void apply(HydroParams params,
                      DataArray2d Udata,
                      int nbCells)
    {
        MakeBoundariesFunctor2D_RayleighTaylor<faceId> functor(params, Udata);
        Kokkos::parallel_for(nbCells, functor);
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const int& index) const
    {
        const int ny = params.ny;

        const int ghostWidth = params.ghostWidth;

        const int imin = params.imin;
        const int imax = params.imax;

        const int jmin = params.jmin;

        const real_t gamma0 = params.settings.gamma0;

        if (faceId == FACE_YMIN)
        {
            const int i = index / ghostWidth;

            if(i >= imin && i <= imax)
            {
                for (int j=jmin+ghostWidth-1; j>=jmin; --j)
                {
                    const int j0 = 2*ghostWidth-1-j;

                    const real_t rho_R = Udata(i, j0, ID);
                    const real_t u_R = Udata(i, j0, IU) / rho_R;
                    const real_t v_R = Udata(i, j0, IV) / rho_R;
                    const real_t p_R = ((gamma0 - ONE_F) *
                                        (Udata(i, j0, IP) - HALF_F * rho_R * (u_R * u_R + v_R * v_R)));
                    const real_t psi_R = psi(i, j+1);

                    const real_t psi_L = psi(i, j);
                    const real_t rho_L = rho_R;
                    const real_t u_L = + u_R;
                    const real_t v_L = - v_R;
                    const real_t p_L = p_R - HALF_F * (rho_R + rho_L) * (psi_L - psi_R);

                    Udata(i, j, ID) = rho_L;
                    Udata(i, j, IU) = rho_L * u_L;
                    Udata(i, j, IV) = rho_L * v_L;
                    Udata(i, j, IP) = p_L / (params.settings.gamma0-ONE_F) + HALF_F * rho_L * (u_L * u_L + v_L * v_L);
                }
            }
        }

        if (faceId == FACE_YMAX)
        {
            const int i = index / ghostWidth;

            if(i >= imin && i <= imax)
            {
                for (int j=ny+ghostWidth; j<=ny+2*ghostWidth-1; ++j)
                {
                    const int j0 = 2*ny+2*ghostWidth-1-j;

                    const real_t rho_L = Udata(i, j0, ID);
                    const real_t u_L = Udata(i, j0, IU) / rho_L;
                    const real_t v_L = Udata(i, j0, IV) / rho_L;
                    const real_t p_L = ((gamma0 - ONE_F)*
                                        (Udata(i, j0, IP) - HALF_F * rho_L * (u_L * u_L + v_L * v_L)));
                    const real_t psi_L = psi(i, j-1);

                    const real_t psi_R = psi(i, j);
                    const real_t rho_R = rho_L;
                    const real_t u_R = + u_L;
                    const real_t v_R = - v_L;
                    const real_t p_R = p_L + HALF_F * (rho_R + rho_L) * (psi_L - psi_R);


                    Udata(i, j, ID) = rho_R;
                    Udata(i, j, IU) = rho_R * u_R;
                    Udata(i, j, IV) = rho_R * v_R;
                    Udata(i, j, IP) = p_R / (params.settings.gamma0-ONE_F) + HALF_F * rho_R * (u_R * u_R + v_R * v_R);
                }
            }
        }
    }

    DataArray2d Udata;
}; // MakeBoundariesFunctor2D_RayleighTaylor

template <FaceIdType faceId>
class MakeBoundariesFunctor3D_RayleighTaylor : all_regime::HydroBaseFunctor3D
{
public:
    MakeBoundariesFunctor3D_RayleighTaylor(HydroParams params_, DataArray Udata_) :
        all_regime::HydroBaseFunctor3D(params_), Udata(Udata_) {};

    static void apply(HydroParams params,
                      DataArray3d Udata,
                      int nbCells)
    {
        MakeBoundariesFunctor3D_RayleighTaylor<faceId> functor(params, Udata);
        Kokkos::parallel_for(nbCells, functor);
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const int& index) const
    {
        const int nz = params.nz;

        const int ghostWidth = params.ghostWidth;

        const int imin = params.imin;
        const int jmin = params.jmin;
        const int kmin = params.kmin;

        const int imax = params.imax;
        const int jmax = params.jmax;

        const int isize = params.isize;
        const int jsize = params.jsize;

        const real_t gamma0 = params.settings.gamma0;

        if (faceId == FACE_ZMIN)
        {
            const int k_ = index / (isize*jsize);
            const int j = (index - k_*isize*jsize) / isize;
            const int i = index - j*isize - k_*isize*jsize;

            if(j >= jmin && j <= jmax &&
               i >= imin && i <= imax &&
               k_ == kmin+ghostWidth-1)
            {
                for (int k=kmin+ghostWidth-1; k>=kmin; --k)
                {
                    const int k0 = 2*ghostWidth-1-k;

                    const real_t rho_R = Udata(i, j, k0, ID);
                    const real_t u_R = Udata(i, j, k0, IU) / rho_R;
                    const real_t v_R = Udata(i, j, k0, IV) / rho_R;
                    const real_t w_R = Udata(i, j, k0, IW) / rho_R;
                    const real_t p_R = ((gamma0 - ONE_F) *
                                        (Udata(i, j, k0, IP) - HALF_F * rho_R * (u_R*u_R + v_R*v_R + w_R*w_R)));
                    const real_t phi_R = phi(i, j, k+1);

                    const real_t phi_L = phi(i, j, k);
                    const real_t rho_L = rho_R;
                    const real_t u_L = + u_R;
                    const real_t v_L = + v_R;
                    const real_t w_L = - w_R;
                    const real_t p_L = p_R - HALF_F * (rho_R + rho_L) * (phi_L - phi_R);

                    Udata(i, j, k, ID) = rho_L;
                    Udata(i, j, k, IU) = rho_L * u_L;
                    Udata(i, j, k, IV) = rho_L * v_L;
                    Udata(i, j, k, IW) = rho_L * w_L;
                    Udata(i, j, k, IP) = p_L / (params.settings.gamma0-ONE_F) + HALF_F * rho_L * (u_L*u_L + v_L*v_L + w_L*w_L);
                }
            }
        }

        if (faceId == FACE_ZMAX)
        {
            int k_ = index / (isize*jsize);
            const int j = (index - k_*isize*jsize) / isize;
            const int i = index - j*isize - k_*isize*jsize;
            k_ += nz + ghostWidth;

            if(j >= jmin && j <= jmax &&
               i >= imin && i <= imax &&
               k_ == nz+ghostWidth)
            {
                for (int k=nz+ghostWidth; k<=nz+2*ghostWidth-1; ++k)
                {
                    const int k0 = 2*nz+2*ghostWidth-1-k;

                    const real_t rho_L = Udata(i, j, k0, ID);
                    const real_t u_L = Udata(i, j, k0, IU) / rho_L;
                    const real_t v_L = Udata(i, j, k0, IV) / rho_L;
                    const real_t w_L = Udata(i, j, k0, IW) / rho_L;
                    const real_t p_L = ((gamma0 - ONE_F)*
                                        (Udata(i, j, k0, IP) - HALF_F * rho_L * (u_L*u_L + v_L*v_L + w_L*w_L)));
                    const real_t phi_L = phi(i, j, k-1);

                    const real_t phi_R = phi(i, j, k);
                    const real_t rho_R = rho_L;
                    const real_t u_R = + u_L;
                    const real_t v_R = + v_L;
                    const real_t w_R = - w_L;
                    const real_t p_R = p_L + HALF_F * (rho_R + rho_L) * (phi_L - phi_R);

                    Udata(i, j, k, ID) = rho_R;
                    Udata(i, j, k, IU) = rho_R * u_R;
                    Udata(i, j, k, IV) = rho_R * v_R;
                    Udata(i, j, k, IW) = rho_R * w_R;
                    Udata(i, j, k, IP) = p_R / (params.settings.gamma0-ONE_F) + HALF_F * rho_R * (u_R*u_R + v_R*v_R + w_R*w_R);
                }
            }
        }
    }

    DataArray Udata;
}; // MakeBoundariesFunctor3D_RayleighTaylor

} // namespace euler_kokkos
