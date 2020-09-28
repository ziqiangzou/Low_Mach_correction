
#pragma once

#include <limits> // for std::numeric_limits
#ifdef __CUDA_ARCH__
#include <math_constants.h> // for cuda math constants, e.g. CUDART_INF
#include <iostream>
#include <limits>
#include <iomanip>
#include <sstream>
#endif // __CUDA_ARCH__

#include "shared/kokkos_shared.h"
#include "HydroBaseFunctor2D.h"
#include "shared/RiemannSolvers.h"
#include "fstream"
#include<ctime>

namespace euler_kokkos { namespace all_regime
	{
		class CopyToGhostFunctor2D : HydroBaseFunctor2D
		{
			public:
				CopyToGhostFunctor2D(HydroParams params_,
						DataArray Udata_, DataArray U0data_) :
					HydroBaseFunctor2D(params_),
					Udata(Udata_), U0data(U0data_){};

				static void apply(HydroParams params,
						DataArray Udata, DataArray U0data, 
						int nbCells)
				{
					CopyToGhostFunctor2D functor(params, Udata, U0data);
					Kokkos::parallel_for(nbCells, functor);
				}
				KOKKOS_INLINE_FUNCTION
					void operator()(int index) const
					{
						int i,j;
						index2coord(index, i, j, params.isize, params.jsize);

						const int ghostWidth = params.ghostWidth;

						if (j>=ghostWidth && j<=params.jmax-ghostWidth &&
								i>=ghostWidth && i<=params.imax-ghostWidth)
						{
							const real_t phiLoc = Udata(i, j, IH);
							if (phiLoc > ZERO_F)
							{
								U0data(i, j, ID0) = Udata(i, j, ID);
								U0data(i, j, IE0) = Udata(i, j, IE);

							}
							else
							{
								U0data(i, j, ID1) = Udata(i, j, ID);
								U0data(i, j, IE1) = Udata(i, j, IE);

							}
						}
					}

				const DataArray Udata;
				const DataArray U0data;
		}; // CopyToGhostFunctor2D
		class ComputeFirstDerivativeFunctor2D : HydroBaseFunctor2D
		{
			public:
				ComputeFirstDerivativeFunctor2D(HydroParams params_,
						DataArray Qdata_, DataArray U0data_, DataArray gradphi_) :
					HydroBaseFunctor2D(params_),
					Qdata(Qdata_), U0data(U0data_), gradphi(gradphi_) {};

				static void apply(HydroParams params,
						DataArray Qdata, DataArray U0data, DataArray gradphi,
						int nbCells)
				{
					ComputeFirstDerivativeFunctor2D functor(params, Qdata, U0data, gradphi);
					Kokkos::parallel_for(nbCells, functor);
				}
				KOKKOS_INLINE_FUNCTION
					void operator()(int index) const
					{
						int i,j;
						index2coord(index, i, j, params.isize, params.jsize);

						const int ghostWidth = params.ghostWidth;
						const real_t dx = params.dx;


						if (j>=ghostWidth && j<=params.jmax-ghostWidth &&
								i>=ghostWidth && i<=params.imax-ghostWidth)
						{
							const real_t phiLoc = Qdata(i, j, IH);
							if (fabs(phiLoc) > dx && fabs(phiLoc) < TWO_F * THREE_F  * dx)
							{

								const real_t drhodx   = (Qdata(i+1, j  , ID) - Qdata(i-1, j  , ID));
								const real_t drhody   = (Qdata(i  , j+1, ID) - Qdata(i  , j-1, ID));

								const real_t dEdx     = (Qdata(i+1, j  , IE) - Qdata(i-1, j  , IE));
								const real_t dEdy     = (Qdata(i  , j+1, IE) - Qdata(i  , j-1, IE));

								real_t gradphix = gradphi(i, j, IPX);
								real_t gradphiy = gradphi(i, j, IPY);

								const real_t gradphimod = sqrt(gradphix*gradphix + gradphiy  * gradphiy);
								gradphix /= gradphimod;
								gradphiy /= gradphimod;

								if (phiLoc>ZERO_F)
								{
									U0data(i, j, FD0) =  drhodx * gradphix + drhody * gradphiy;
									U0data(i, j, FE0) =  dEdx   * gradphix + dEdy   * gradphiy;
								}
								else
								{
									U0data(i, j, FD1) =  drhodx * gradphix + drhody * gradphiy;
									U0data(i, j, FE1) =  dEdx   * gradphix + dEdy   * gradphiy;
								}
							}

						}
					}

				const DataArray Qdata;
				const DataArray U0data;
				const DataArray gradphi;
		}; // ComputeFirstDerivetiveFunctor2D
		class ExtrapolateStep2Functor2D : HydroBaseFunctor2D
		{
			public:
				ExtrapolateStep2Functor2D(HydroParams params_,
						DataArray Qdata_, DataArray U0data_, DataArray gradphi_) :
					HydroBaseFunctor2D(params_),
					Qdata(Qdata_), U0data(U0data_), gradphi(gradphi_) {};

				static void apply(HydroParams params,
						DataArray Qdata, DataArray U0data, DataArray gradphi,
						int nbCells)
				{
					ExtrapolateStep2Functor2D functor(params, Qdata, U0data, gradphi);
					Kokkos::parallel_for(nbCells, functor);
				}

				KOKKOS_INLINE_FUNCTION
					void operator()(int index) const
					{
						int i,j;
						index2coord(index, i, j, params.isize, params.jsize);

						const int ghostWidth = params.ghostWidth;

						if (j>=ghostWidth && j<=params.jmax-ghostWidth &&
								i>=ghostWidth && i<=params.imax-ghostWidth)
						{
							const real_t phiLoc = Qdata(i, j, IH);

							real_t gradphix = gradphi(i, j, IPX);
							real_t gradphiy = gradphi(i, j, IPY);

							const real_t gradphimod = sqrt(gradphix*gradphix + gradphiy  * gradphiy);

							gradphix /= gradphimod;
							gradphiy /= gradphimod;

							if (phiLoc>ZERO_F)
							{
								const real_t drhodx = gradphix >ZERO_F? THREE_F * U0data(i, j, RD1) - FOUR_F * U0data(i-1, j, RD1) + U0data(i-2, j, RD1) : -U0data(i+2, j, RD1) + FOUR_F * U0data(i+1, j, RD1) - THREE_F * U0data(i, j, RD1);

								const real_t drhody = gradphiy >ZERO_F? THREE_F * U0data(i, j, RD1) - FOUR_F * U0data(i, j-1, RD1) + U0data(i, j-2, RD1) : -U0data(i, j+2, RD1) + FOUR_F * U0data(i, j+1, RD1) - THREE_F * U0data(i, j, RD1);

								const real_t dpdx = gradphix >ZERO_F? THREE_F * U0data(i, j, RE1) - FOUR_F * U0data(i-1, j, RE1) + U0data(i-2, j, RE1) : -U0data(i+2, j, RE1) + FOUR_F * U0data(i+1, j, RE1) - THREE_F * U0data(i, j, RE1);

								const real_t dpdy = gradphiy >ZERO_F? THREE_F * U0data(i, j, RE1) - FOUR_F * U0data(i, j-1, RE1) + U0data(i, j-2, RE1) : -U0data(i, j+2, RE1) + FOUR_F * U0data(i, j+1, RE1) - THREE_F * U0data(i, j, RE1);

								U0data(i, j, ID1) -= HALF_F * HALF_F * HALF_F *(gradphix * drhodx   + gradphiy * drhody   - U0data(i, j, FD1) );
								U0data(i, j, IE1) -= HALF_F * HALF_F * HALF_F *(gradphix * dpdx     + gradphiy * dpdy     - U0data(i, j, FE1) );

							}
							if (phiLoc<ZERO_F)
							{
								const real_t drhodx = gradphix <ZERO_F? THREE_F * U0data(i, j, RD0) - FOUR_F * U0data(i-1, j, RD0) + U0data(i-2, j, RD0) : -U0data(i+2, j, RD0) + FOUR_F * U0data(i+1, j, RD0) - THREE_F * U0data(i, j, RD0);

								const real_t drhody = gradphiy <ZERO_F? THREE_F * U0data(i, j, RD0) - FOUR_F * U0data(i, j-1, RD0) + U0data(i, j-2, RD0) : -U0data(i, j+2, RD0) + FOUR_F * U0data(i, j+1, RD0) - THREE_F * U0data(i, j, RD0);

								const real_t dpdx = gradphix <ZERO_F? THREE_F * U0data(i, j, RE0) - FOUR_F * U0data(i-1, j, RE0) + U0data(i-2, j, RE0) : -U0data(i+2, j, RE0) + FOUR_F * U0data(i+1, j, RE0) - THREE_F * U0data(i, j, RE0);

								const real_t dpdy = gradphiy <ZERO_F? THREE_F * U0data(i, j, RE0) - FOUR_F * U0data(i, j-1, RE0) + U0data(i, j-2, RE0) : -U0data(i, j+2, RE0) + FOUR_F * U0data(i, j+1, RE0) - THREE_F * U0data(i, j, RE0);

								U0data(i, j, ID0) += HALF_F * HALF_F * HALF_F *(gradphix * drhodx  + gradphiy * drhody - U0data(i, j, FD0) );
								U0data(i, j, IE0) += HALF_F * HALF_F * HALF_F *(gradphix * dpdx    + gradphiy * dpdy   - U0data(i, j, FE0) );
							}

						}
					}
				const DataArray Qdata;
				const DataArray U0data;
				const DataArray gradphi;
		};// ExtrapolateDensityPressureFunctor2D
		class ExtrapolateFunctor2D : HydroBaseFunctor2D
		{
			public:
				ExtrapolateFunctor2D(HydroParams params_,
						DataArray Qdata_, DataArray U0data_, DataArray gradphi_) :
					HydroBaseFunctor2D(params_),
					Qdata(Qdata_), U0data(U0data_), gradphi(gradphi_) {};

				static void apply(HydroParams params,
						DataArray Qdata, DataArray U0data, DataArray gradphi,
						int nbCells)
				{
					ExtrapolateFunctor2D functor(params, Qdata, U0data, gradphi);
					Kokkos::parallel_for(nbCells, functor);
				}

				KOKKOS_INLINE_FUNCTION
					void operator()(int index) const
					{
						int i,j;
						index2coord(index, i, j, params.isize, params.jsize);

						const int ghostWidth = params.ghostWidth;

						if (j>=ghostWidth && j<=params.jmax-ghostWidth &&
								i>=ghostWidth && i<=params.imax-ghostWidth)
						{
							const real_t phiLoc = Qdata(i, j, IH);

							real_t gradphix = gradphi(i, j, IPX);
							real_t gradphiy = gradphi(i, j, IPY);

							const real_t gradphimod = sqrt(gradphix*gradphix + gradphiy  * gradphiy);

							gradphix /= gradphimod;
							gradphiy /= gradphimod;

							if (phiLoc>ZERO_F)
							{
								const real_t drhodx = gradphix >ZERO_F? THREE_F * U0data(i, j, ID1) - FOUR_F * U0data(i-1, j, ID1) + U0data(i-2, j, ID1) : -U0data(i+2, j, ID1) + FOUR_F * U0data(i+1, j, ID1) - THREE_F * U0data(i, j, ID1);

								const real_t drhody = gradphiy >ZERO_F? THREE_F * U0data(i, j, ID1) - FOUR_F * U0data(i, j-1, ID1) + U0data(i, j-2, ID1) : -U0data(i, j+2, ID1) + FOUR_F * U0data(i, j+1, ID1) - THREE_F * U0data(i, j, ID1);

								const real_t dpdx = gradphix >ZERO_F? THREE_F * U0data(i, j, IE1) - FOUR_F * U0data(i-1, j, IE1) + U0data(i-2, j, IE1) : -U0data(i+2, j, IE1) + FOUR_F * U0data(i+1, j, IE1) - THREE_F * U0data(i, j, IE1);

								const real_t dpdy = gradphiy >ZERO_F? THREE_F * U0data(i, j, IE1) - FOUR_F * U0data(i, j-1, IE1) + U0data(i, j-2, IE1) : -U0data(i, j+2, IE1) + FOUR_F * U0data(i, j+1, IE1) - THREE_F * U0data(i, j, IE1);

								U0data(i, j, RD1) -= HALF_F * HALF_F * HALF_F * HALF_F *(gradphix * drhodx   + gradphiy * drhody   - U0data(i, j, FD1) );
								U0data(i, j, RE1) -= HALF_F * HALF_F * HALF_F * HALF_F *(gradphix * dpdx     + gradphiy * dpdy     - U0data(i, j, FE1) );


							}
							if (phiLoc<ZERO_F)
							{
								const real_t drhodx = gradphix <ZERO_F? THREE_F * U0data(i, j, ID0) - FOUR_F * U0data(i-1, j, ID0) + U0data(i-2, j, ID0) : -U0data(i+2, j, ID0) + FOUR_F * U0data(i+1, j, ID0) - THREE_F * U0data(i, j, ID0);

								const real_t drhody = gradphiy <ZERO_F? THREE_F * U0data(i, j, ID0) - FOUR_F * U0data(i, j-1, ID0) + U0data(i, j-2, ID0) : -U0data(i, j+2, ID0) + FOUR_F * U0data(i, j+1, ID0) - THREE_F * U0data(i, j, ID0);

								const real_t dpdx = gradphix <ZERO_F? THREE_F * U0data(i, j, IE0) - FOUR_F * U0data(i-1, j, IE0) + U0data(i-2, j, IE0) : -U0data(i+2, j, IE0) + FOUR_F * U0data(i+1, j, IE0) - THREE_F * U0data(i, j, IE0);

								const real_t dpdy = gradphiy <ZERO_F? THREE_F * U0data(i, j, IE0) - FOUR_F * U0data(i, j-1, IE0) + U0data(i, j-2, IE0) : -U0data(i, j+2, IE0) + FOUR_F * U0data(i, j+1, IE0) - THREE_F * U0data(i, j, IE0);

								U0data(i, j, RD0) += HALF_F * HALF_F * HALF_F * HALF_F *(gradphix * drhodx  + gradphiy * drhody - U0data(i, j, FD0) );
								U0data(i, j, RE0) += HALF_F * HALF_F * HALF_F * HALF_F *(gradphix * dpdx    + gradphiy * dpdy   - U0data(i, j, FE0) );
							}

						}
					}
				const DataArray Qdata;
				const DataArray U0data;
				const DataArray gradphi;
		};// ExtrapolateDensityPressureFunctor2D
		class ExtrapolateFirstDerivativeFunctor2D : HydroBaseFunctor2D
		{
			public:
				ExtrapolateFirstDerivativeFunctor2D(HydroParams params_,
						DataArray Qdata_, DataArray U0data_, DataArray gradphi_) :
					HydroBaseFunctor2D(params_),
					Qdata(Qdata_), U0data(U0data_), gradphi(gradphi_) {};

				static void apply(HydroParams params,
						DataArray Qdata, DataArray U0data, DataArray gradphi,
						int nbCells)
				{
					ExtrapolateFirstDerivativeFunctor2D functor(params, Qdata, U0data, gradphi);
					Kokkos::parallel_for(nbCells, functor);
				}

				KOKKOS_INLINE_FUNCTION
					void operator()(int index) const
					{
						int i,j;
						index2coord(index, i, j, params.isize, params.jsize);

						const int ghostWidth = params.ghostWidth;
						const real_t dx = params.dx;

						if (j>=ghostWidth && j<=params.jmax-ghostWidth &&
								i>=ghostWidth && i<=params.imax-ghostWidth)
						{
							const real_t phiLoc = Qdata(i, j, IH);
							real_t gradphix = gradphi(i, j, IPX);
							real_t gradphiy = gradphi(i, j, IPY);
							const real_t gradphimod = sqrt(gradphix*gradphix + gradphiy  * gradphiy);
							gradphix /= gradphimod;
							gradphiy /= gradphimod;

							if ((phiLoc + dx)>ZERO_F)
							{
								const real_t d_drhodx = gradphix >ZERO_F? THREE_F * U0data(i, j, FD1) - FOUR_F * U0data(i-1, j, FD1) + U0data(i-2, j, FD1) : -U0data(i+2, j, FD1) + FOUR_F * U0data(i+1, j, FD1) - THREE_F * U0data(i, j, FD1);

								const real_t d_drhody = gradphiy >ZERO_F? THREE_F * U0data(i, j, FD1) - FOUR_F * U0data(i, j-1, FD1) + U0data(i, j-2, FD1) : -U0data(i, j+2, FD1) + FOUR_F * U0data(i, j+1, FD1) - THREE_F * U0data(i, j, FD1);

								const real_t d_dEdx = gradphix >ZERO_F? THREE_F * U0data(i, j, FE1) - FOUR_F * U0data(i-1, j, FE1) + U0data(i-2, j, FE1) : -U0data(i+2, j, FE1) + FOUR_F * U0data(i+1, j, FE1) - THREE_F * U0data(i, j, FE1);

								const real_t d_dEdy = gradphiy >ZERO_F? THREE_F * U0data(i, j, FE1) - FOUR_F * U0data(i, j-1, FE1) + U0data(i, j-2, FE1) : -U0data(i, j+2, FE1) + FOUR_F * U0data(i, j+1, FE1) - THREE_F * U0data(i, j, FE1);
								U0data(i, j, RD1) -= HALF_F  *HALF_F * HALF_F *HALF_F * (gradphix *  d_drhodx + gradphiy * d_drhody );
								U0data(i, j, RE1) -= HALF_F  *HALF_F * HALF_F *HALF_F * (gradphix *  d_dEdx   + gradphiy * d_dEdy   );


							}
							if ((phiLoc - dx)<ZERO_F)
							{
								const real_t d_drhodx = gradphix <ZERO_F? THREE_F * U0data(i, j, FD0) - FOUR_F * U0data(i-1, j, FD0) + U0data(i-2, j, FD0) : -U0data(i+2, j, FD0) + FOUR_F * U0data(i+1, j, FD0) - THREE_F * U0data(i, j, FD0);

								const real_t d_drhody = gradphiy <ZERO_F? THREE_F * U0data(i, j, FD0) - FOUR_F * U0data(i, j-1, FD0) + U0data(i, j-2, FD0) : -U0data(i, j+2, FD0) + FOUR_F * U0data(i, j+1, FD0) - THREE_F * U0data(i, j, FD0);

								const real_t d_dEdx = gradphix <ZERO_F? THREE_F * U0data(i, j, FE0) - FOUR_F * U0data(i-1, j, FE0) + U0data(i-2, j, FE0) : -U0data(i+2, j, FE0) + FOUR_F * U0data(i+1, j, FE0) - THREE_F * U0data(i, j, FE0);

								const real_t d_dEdy = gradphiy <ZERO_F? THREE_F * U0data(i, j, FE0) - FOUR_F * U0data(i, j-1, FE0) + U0data(i, j-2, FE0) : -U0data(i, j+2, FE0) + FOUR_F * U0data(i, j+1, FE0) - THREE_F * U0data(i, j, FE0);
								U0data(i, j, RD0) += HALF_F  *HALF_F * HALF_F *HALF_F * (gradphix * d_drhodx + gradphiy * d_drhody );
								U0data(i, j, RE0) += HALF_F  *HALF_F * HALF_F *HALF_F * (gradphix * d_dEdx   + gradphiy * d_dEdy   );
							}

						}
					}
				const DataArray Qdata;
				const DataArray U0data;
				const DataArray gradphi;
		};// ExtrapolateFirstDerivetiveFunctor2D
		class ExtrapolateFirstDerivativeStep2Functor2D : HydroBaseFunctor2D
		{
			public:
				ExtrapolateFirstDerivativeStep2Functor2D(HydroParams params_,
						DataArray Qdata_, DataArray U0data_, DataArray gradphi_) :
					HydroBaseFunctor2D(params_),
					Qdata(Qdata_), U0data(U0data_), gradphi(gradphi_) {};

				static void apply(HydroParams params,
						DataArray Qdata, DataArray U0data, DataArray gradphi,
						int nbCells)
				{
					ExtrapolateFirstDerivativeStep2Functor2D functor(params, Qdata, U0data, gradphi);
					Kokkos::parallel_for(nbCells, functor);
				}

				KOKKOS_INLINE_FUNCTION
					void operator()(int index) const
					{
						int i,j;
						index2coord(index, i, j, params.isize, params.jsize);

						const int ghostWidth = params.ghostWidth;
						const real_t dx = params.dx;

						if (j>=ghostWidth && j<=params.jmax-ghostWidth &&
								i>=ghostWidth && i<=params.imax-ghostWidth)
						{
							const real_t phiLoc = Qdata(i, j, IH);
							real_t gradphix = gradphi(i, j, IPX);
							real_t gradphiy = gradphi(i, j, IPY);
							const real_t gradphimod = sqrt(gradphix*gradphix + gradphiy  * gradphiy);
							gradphix /= gradphimod;
							gradphiy /= gradphimod;

							if ((phiLoc + dx)>ZERO_F)
							{
								const real_t d_drhodx = gradphix >ZERO_F? THREE_F * U0data(i, j, RD1) - FOUR_F * U0data(i-1, j, RD1) + U0data(i-2, j, RD1) : -U0data(i+2, j, RD1) + FOUR_F * U0data(i+1, j, RD1) - THREE_F * U0data(i, j, RD1);

								const real_t d_drhody = gradphiy >ZERO_F? THREE_F * U0data(i, j, RD1) - FOUR_F * U0data(i, j-1, RD1) + U0data(i, j-2, RD1) : -U0data(i, j+2, RD1) + FOUR_F * U0data(i, j+1, RD1) - THREE_F * U0data(i, j, RD1);

								const real_t d_dEdx = gradphix >ZERO_F? THREE_F * U0data(i, j, RE1) - FOUR_F * U0data(i-1, j, RE1) + U0data(i-2, j, RE1) : -U0data(i+2, j, RE1) + FOUR_F * U0data(i+1, j, RE1) - THREE_F * U0data(i, j, RE1);

								const real_t d_dEdy = gradphiy >ZERO_F? THREE_F * U0data(i, j, RE1) - FOUR_F * U0data(i, j-1, RE1) + U0data(i, j-2, RE1) : -U0data(i, j+2, RE1) + FOUR_F * U0data(i, j+1, RE1) - THREE_F * U0data(i, j, RE1);
								U0data(i, j, FD1) -= HALF_F  *HALF_F *HALF_F *  (gradphix *  d_drhodx + gradphiy * d_drhody );
								U0data(i, j, FE1) -= HALF_F  *HALF_F *HALF_F *  (gradphix *  d_dEdx   + gradphiy * d_dEdy   );


							}
							if ((phiLoc - dx)<ZERO_F)
							{
								const real_t d_drhodx = gradphix <ZERO_F? THREE_F * U0data(i, j, RD0) - FOUR_F * U0data(i-1, j, RD0) + U0data(i-2, j, RD0) : -U0data(i+2, j, RD0) + FOUR_F * U0data(i+1, j, RD0) - THREE_F * U0data(i, j, RD0);

								const real_t d_drhody = gradphiy <ZERO_F? THREE_F * U0data(i, j, RD0) - FOUR_F * U0data(i, j-1, RD0) + U0data(i, j-2, RD0) : -U0data(i, j+2, RD0) + FOUR_F * U0data(i, j+1, RD0) - THREE_F * U0data(i, j, RD0);

								const real_t d_dEdx = gradphix <ZERO_F? THREE_F * U0data(i, j, RE0) - FOUR_F * U0data(i-1, j, RE0) + U0data(i-2, j, RE0) : -U0data(i+2, j, RE0) + FOUR_F * U0data(i+1, j, RE0) - THREE_F * U0data(i, j, RE0);

								const real_t d_dEdy = gradphiy <ZERO_F? THREE_F * U0data(i, j, RE0) - FOUR_F * U0data(i, j-1, RE0) + U0data(i, j-2, RE0) : -U0data(i, j+2, RE0) + FOUR_F * U0data(i, j+1, RE0) - THREE_F * U0data(i, j, RE0);
								U0data(i, j, FD0) += HALF_F  *HALF_F *HALF_F *  (gradphix * d_drhodx + gradphiy * d_drhody );
								U0data(i, j, FE0) += HALF_F  *HALF_F *HALF_F *  (gradphix * d_dEdx   + gradphiy * d_dEdy   );
							}

						}
					}
				const DataArray Qdata;
				const DataArray U0data;
				const DataArray gradphi;
		};// ExtrapolateFirstDerivetiveFunctor2D
	}
}
