#pragma once

#include "all_regime/HydroBaseFunctor2D.h"
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
		class MakeBoundariesFunctor2D_DamBreak : all_regime::HydroBaseFunctor2D
	{
		public:
			MakeBoundariesFunctor2D_DamBreak(HydroParams params_,
					DataArray2d Udata_)
				: all_regime::HydroBaseFunctor2D(params_)
				  , Udata(Udata_) {};

			static void apply(HydroParams params,
					DataArray2d Udata,
					int nbCells)
			{
				MakeBoundariesFunctor2D_DamBreak<faceId> functor(params, Udata);
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

					const real_t Rstar = params.settings.Rstar0;

					if (faceId == FACE_YMIN)
					{
						const int i = index / ghostWidth;

						if(i >= imin && i <= imax)
						{
							for (int j=jmin+ghostWidth-1; j>=jmin; j--)
							{
								const int j0 = 2*ghostWidth-1-j;
								const HydroState q_R = computePrimitives(getHydroState(Udata, i, j0));
								const real_t T_R = computeTemperature(q_R);

								const real_t T_L = T_R ;
								const real_t Dphi = psi(i, j0) - psi(i, j);
								HydroState q_L;
								q_L[ID] = (q_R[IP] + HALF_F*q_R[ID]*Dphi) / (Rstar * T_L - HALF_F*Dphi);
								q_L[IP] = q_L[ID] * Rstar * T_L;
								q_L[IS] = q_R[IS];
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
								const int j0 = 2*ny+2*ghostWidth-1-j;
								const HydroState q_L = computePrimitives(getHydroState(Udata, i, j0));
								const real_t T_L = computeTemperature(q_L);

								const real_t Dphi = psi(i, j) - psi(i, j0);
								const real_t T_R = T_L;
								HydroState q_R;
								q_R[ID] = (q_L[IP] - HALF_F*q_L[ID]*Dphi) / (Rstar * T_R + HALF_F*Dphi);
								q_R[IP] = q_R[ID] * Rstar * T_R;
								q_R[IS] = q_L[IS];
								q_R[IU] = + q_L[IU];
								q_R[IV] = - q_L[IV];

								setHydroState(Udata, computeConservatives(q_R), i, j);
							}
						}
					}
				}

			// HydroParams params;
			DataArray2d Udata;
	}; // MakeBoundariesFunctor2D_DamBreak

} // namespace euler_kokkos
