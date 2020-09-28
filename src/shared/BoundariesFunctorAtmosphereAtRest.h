#pragma once

#include "shared/HydroParams.h"    // for HydroParams
#include "shared/kokkos_shared.h"  // for Data arrays
#include <iostream>

namespace euler_kokkos
{

/**
 * Functors to update ghost cells (Hydro 2D) for the test case
 */
template <FaceIdType faceId>
class MakeBoundariesFunctor2D_AtmosphereAtRest
{
public:
    MakeBoundariesFunctor2D_AtmosphereAtRest(HydroParams params_, DataArray2d Udata_) :
        params(params_), Udata(Udata_) {};

    static void apply(HydroParams params, DataArray2d Udata, int nbCells)
    {
        MakeBoundariesFunctor2D_AtmosphereAtRest<faceId> functor(params, Udata);
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

        const real_t dy = params.dy;
        const real_t gamma0 = params.settings.gamma0;

        if (faceId == FACE_YMIN)
        {
            const int i = index / ghostWidth;

            if(i >= imin && i <= imax)
            {
                for (int j=jmin+ghostWidth-1; j>=jmin; j--)
                {
                    /* std::cout << j << std::endl; */
                    // u[index + ID*ijSize] = exp((j-ghostWidth) * log((TWO_F - dy)/(TWO_F + dy)));
                    Udata(i, j, ID) = (TWO_F + dy) / (TWO_F - dy) * Udata(i, j+1, ID);
                    Udata(i, j, IP) = Udata(i, j, ID) / (gamma0-ONE_F);
                    Udata(i, j, IU) = ZERO_F;
                    Udata(i, j, IV) = ZERO_F;
                }
            }
        }

        if (faceId == FACE_YMAX)
        {
            const int i = index / ghostWidth;

            if(i >= imin && i <= imax)
            {
                for (int j= ny+ghostWidth; j <= ny+2*ghostWidth-1; j++)
                {
                    // u[index + ID*ijSize] = exp((j-ghostWidth) * log((TWO_F - dy)/(TWO_F + dy)));
                    Udata(i, j, ID) = (TWO_F - dy) / (TWO_F + dy) * Udata(i, j-1, ID);
                    Udata(i, j, IP) = Udata(i, j, ID)/(gamma0-ONE_F);
                    Udata(i, j, IU) = ZERO_F;
                    Udata(i, j, IV) = ZERO_F;
                }
            }
        }
    }

    const HydroParams params;
    DataArray2d Udata;
}; // MakeBoundariesFunctor2D_AtmosphereAtRest

} // namespace euler_kokkos
