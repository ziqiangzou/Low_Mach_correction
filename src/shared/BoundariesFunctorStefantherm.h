#pragma once

#include "all_regime/HydroBaseFunctor2D.h"
#include "shared/StefanthermParams.h"
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
		class MakeBoundariesFunctor2D_Stefantherm : all_regime::HydroBaseFunctor2D
	{
		public:
			MakeBoundariesFunctor2D_Stefantherm(HydroParams params_,
					StefanthermParams stefanthermParams_,
					DataArray2d Udata_)
				: all_regime::HydroBaseFunctor2D(params_)
				  , stefanthermParams(stefanthermParams_)
				  , Udata(Udata_) {};

			static void apply(HydroParams params,
					StefanthermParams stefanthermParams,
					DataArray2d Udata,
					int nbCells)
			{
				MakeBoundariesFunctor2D_Stefantherm<faceId> functor(params, stefanthermParams, Udata);
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


					const real_t T_top = stefanthermParams.wall_temp;
					//const real_t density_vapor = stefanthermParams.vapor_density;

					if (faceId == FACE_YMIN)
					{
						const int i = index / ghostWidth;

						if(i >= imin && i <= imax)
						{
							for (int j=jmin+ghostWidth-1; j>=jmin; j--)
							{
								const int j0 = j+1;
								const HydroState q_R  = computePrimitives(getHydroState(Udata, i, j0));

								HydroState q_L;
								q_L[ID] = q_R[ID];
								q_L[IP] = q_R[IP];
								q_L[IH] = q_R[IH];
								q_L[IU] = + q_R[IU];
								q_L[IV] = + q_R[IV];

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
								const int j0 = ny +ghostWidth -1;
								const HydroState q_L = computePrimitives(getHydroState(Udata, i, j0));
								const real_t T0=computeTemperature(q_L);
								real_t T_R;
								T_R = T0 + (T_top - T0) * ((j - j0)*TWO_F -ONE_F);
								const real_t Rstar =q_L[IH]>ZERO_F?  params.settings.Rstar0:params.settings.Rstar1;

								HydroState q_R;
								const bool barotropic= q_L[IH]>ZERO_F? params.settings.barotropic0 : params.settings.barotropic1;

								if (!barotropic)
								{
									q_R[ID] = (q_L[IP])/ T_R /Rstar;
									q_R[IP] = q_L[IP];
								}
								else
								{
									q_R[ID] = q_L[ID];
									q_R[IP] = T_R;
								}
								q_R[IH] = q_L[IH];
								q_R[IU] = + q_L[IU];
								q_R[IV] = - q_L[IV];

								setHydroState(Udata, computeConservatives(q_R), i, j);
							}
						}
					}
				}

			// HydroParams params;
			StefanthermParams stefanthermParams;
			DataArray2d Udata;
	}; // MakeBoundariesFunctor2D_RayleighBenard

} // namespace euler_kokkos
