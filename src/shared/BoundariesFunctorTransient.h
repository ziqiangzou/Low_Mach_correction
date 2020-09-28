#pragma once

#include "all_regime/HydroBaseFunctor2D.h"
#include "shared/TransientProcessParams.h"
#include "shared/HydroParams.h"    // for HydroParams
#include "shared/kokkos_shared.h"  // for Data arrays
#include "shared/units.h"

#include <iostream>

namespace euler_kokkos
{

/**
 * Functors to update ghost cells (Hydro 2D) for the test case
 */
template <FaceIdType faceId>
class MakeBoundariesFunctor2D_Transient : all_regime::HydroBaseFunctor2D
{
public:
    MakeBoundariesFunctor2D_Transient(HydroParams params_,
                                           TransientParams transientParams_,
                                           DataArray2d Udata_)
        : all_regime::HydroBaseFunctor2D(params_)
        , transientParams(transientParams_)
        , Udata(Udata_) {};

    static void apply(HydroParams params,
                      TransientParams transientParams,
                      DataArray2d Udata,
                      int nbCells)
    {
        MakeBoundariesFunctor2D_Transient<faceId> functor(params, transientParams, Udata);
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


        const real_t T_top = transientParams.transient_temp_top;
        const real_t T_bot = transientParams.transient_temp_bot;

        if (faceId == FACE_YMIN)
        {
            const int i = index / ghostWidth;

            if(i >= imin && i <= imax)
            {
                for (int j=jmin+ghostWidth-1; j>=jmin; j--)
                {
                    const int j0 = j+1;
                    const HydroState q_R  = computePrimitives(getHydroState(Udata, i, j0));
                    const real_t Rstar =q_R[IH]>ZERO_F?  params.settings.Rstar0:params.settings.Rstar1;
                    const real_t Bstate =q_R[IH]>ZERO_F?  params.Bstate0:params.Bstate1;
                    const real_t T0=computeTemperature(q_R);
                    const real_t  T_L = TWO_F * T_bot - T0;

                    HydroState q_L;
                    q_L[ID] = (q_R[IP] + Bstate)/ T_L /Rstar;
                    q_L[IP] = q_R[IP];
                    q_L[IH] = q_R[IH];
                    q_L[IU] = + q_R[IU];
                    q_L[IV] = - q_R[IV];

                    setHydroState(Udata, computeConservatives(q_L), i, j);
                }
            }
        }

        if (faceId == FACE_YMAX)
        {
            const int i = index / ghostWidth;

            if(i >= imin && i <= imax)
            {
                for (int j=ny+ghostWidth; j<=ny+2*ghostWidth-1; j++)
                {
                    const int j0 = j-1;
                    const HydroState q_L = computePrimitives(getHydroState(Udata, i, j0));
                    const real_t T0=computeTemperature(q_L);
                    real_t T_R;
                     T_R = TWO_F * T_top - T0;
                    const real_t Rstar =q_L[IH]>ZERO_F?  params.settings.Rstar0:params.settings.Rstar1;
                    const real_t Bstate =q_L[IH]>ZERO_F?  params.Bstate0:params.Bstate1;

                    HydroState q_R;
                    q_R[ID] = (q_L[IP] + Bstate)/ T_R /Rstar;
                    q_R[IP] = q_L[IP];
                    q_R[IH] = q_L[IH];
                    q_R[IU] = + q_L[IU];
                    q_R[IV] = - q_L[IV];

                    setHydroState(Udata, computeConservatives(q_R), i, j);
                }
            }
        }
    }

    // HydroParams params;
    TransientParams transientParams;
    DataArray2d Udata;
}; // MakeBoundariesFunctor2D_RayleighBenard

} // namespace euler_kokkos
