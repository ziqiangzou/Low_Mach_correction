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

namespace euler_kokkos { namespace all_regime

	{
		class CopyLSFunctor2D : HydroBaseFunctor2D
		{
			public:
				CopyLSFunctor2D(HydroParams params_, DataArray Udata_, DataArray U2data_):
					HydroBaseFunctor2D(params_), Udata(Udata_),U2data(U2data_) {};
				static void apply(HydroParams params,
						DataArray Udata, DataArray U2data,
						int nbCells)
				{
					CopyLSFunctor2D functor(params, Udata, U2data);
					Kokkos::parallel_for(nbCells, functor);
				}
				KOKKOS_INLINE_FUNCTION
					void operator()(int index) const
					{
						int i, j;
						index2coord(index, i, j, params.isize, params.jsize);

						if (j>=params.jmin && j<=params.jmax &&
								i>=params.imin && i<=params.imax)
						{
							Udata(i, j, IH)=U2data(i, j, IH);


						}
					}
				const DataArray Udata;
				const DataArray U2data;
		};//Copy LS function
		class ComputeCurvatureFunctor2D : HydroBaseFunctor2D
		{
			public:
				ComputeCurvatureFunctor2D(HydroParams params_,  DataArray U2data_, DataArray gradphi_):
					HydroBaseFunctor2D(params_), U2data(U2data_),  gradphi(gradphi_) {};
				static void apply(HydroParams params,
						DataArray U2data, DataArray gradphi,
						int nbCells)
				{
					ComputeCurvatureFunctor2D functor(params, U2data, gradphi);
					Kokkos::parallel_for(nbCells, functor);
				}
				KOKKOS_INLINE_FUNCTION
					void operator()(int index) const
					{

						const int ghostWidth = params.ghostWidth;
						const real_t onesurdx=params.onesurdx;
						const real_t onesurdy=params.onesurdy;
						int i, j;
						index2coord(index, i, j, params.isize, params.jsize);

						if (j>=ghostWidth && j<=params.jmax-ghostWidth &&
								i>=ghostWidth && i<=params.imax-ghostWidth)
						{
							const real_t phi0    =U2data(i,   j,   IH);
							if (fabs(phi0)<2.*params.dx)
							{
								const real_t phi0Px  =U2data(i+1, j,   IH);
								const real_t phi0Mx  =U2data(i-1, j,   IH);
								const real_t phi0Py  =U2data(i,   j+1, IH);
								const real_t phi0My  =U2data(i,   j-1, IH);
								const real_t phi0PxPy=U2data(i+1, j+1, IH);
								const real_t phi0MxPy=U2data(i-1, j+1, IH);
								const real_t phi0PxMy=U2data(i+1, j-1, IH);
								const real_t phi0MxMy=U2data(i-1, j-1, IH);
								const real_t dphix =(phi0Px-phi0Mx)*onesurdx*HALF_F;
								const real_t dphiy =(phi0Py-phi0My)*onesurdy*HALF_F;
								const real_t dphixx=(phi0Px+phi0Mx-TWO_F*phi0)*onesurdx*onesurdx;
								const real_t dphiyy=(phi0Py+phi0My-TWO_F*phi0)*onesurdy*onesurdy;
								const real_t dphixy=(phi0PxPy+phi0MxMy-phi0PxMy-phi0MxPy)*ONE_FOURTH_F*onesurdx*onesurdy;

								gradphi(i, j, IPK) = (TWO_F*dphix*dphiy*dphixy - dphix*dphix*dphiyy - dphiy*dphiy*dphixx)/pow((dphix*dphix+dphiy*dphiy),1.5);
							}
							else
								gradphi(i, j, IPK) = ZERO_F;

						}
					}
				const DataArray U2data;
				const DataArray gradphi;
		};// Compute_Curvature_LS

		class ComputeRedistancingPhi1Functor2D : HydroBaseFunctor2D
		{
			public:
				ComputeRedistancingPhi1Functor2D(HydroParams params_, DataArray Udata_, DataArray U2data_, DataArray gradphi_):
					HydroBaseFunctor2D(params_), Udata(Udata_),U2data(U2data_),  gradphi(gradphi_) {};
				static void apply(HydroParams params,
						DataArray Udata, DataArray U2data, DataArray gradphi,
						int nbCells)
				{
					ComputeRedistancingPhi1Functor2D functor(params, Udata, U2data, gradphi);
					Kokkos::parallel_for(nbCells, functor);
				}
				KOKKOS_INLINE_FUNCTION
					void operator()(int index) const
					{
						const int ghostWidth = params.ghostWidth;
						int i, j;
						index2coord(index, i, j, params.isize, params.jsize);

						if (j>=ghostWidth && j<=params.jmax-ghostWidth &&
								i>=ghostWidth && i<=params.imax-ghostWidth)
						{
							const real_t gradphix=gradphi(i, j, IPX);
							const real_t gradphiy=gradphi(i, j, IPY);
							real_t phi =U2data(i, j, IH);
							const real_t sign    =gradphi(i, j, IPS);
							phi-=sign*params.dx*0.3*(sqrt(gradphix*gradphix+gradphiy*gradphiy)-ONE_F);

							U2data(i, j, IH)= phi;

						}
					}
				const DataArray Udata;
				const DataArray U2data;
				const DataArray gradphi;
		};// Redistancing phi WENO5
		class ComputeRedistancingPhi2Functor2D : HydroBaseFunctor2D
		{
			public:
				ComputeRedistancingPhi2Functor2D(HydroParams params_, DataArray Udata_, DataArray U2data_, DataArray gradphi_):
					HydroBaseFunctor2D(params_), Udata(Udata_),U2data(U2data_),  gradphi(gradphi_) {};
				static void apply(HydroParams params,
						DataArray Udata, DataArray U2data, DataArray gradphi,
						int nbCells)
				{
					ComputeRedistancingPhi2Functor2D functor(params, Udata, U2data, gradphi);
					Kokkos::parallel_for(nbCells, functor);
				}
				KOKKOS_INLINE_FUNCTION
					void operator()(int index) const
					{
						const int ghostWidth = params.ghostWidth;
						int i, j;
						index2coord(index, i, j, params.isize, params.jsize);

						if (j>=ghostWidth && j<=params.jmax-ghostWidth &&
								i>=ghostWidth && i<=params.imax-ghostWidth)
						{
							const real_t gradphix=gradphi(i, j, IPX);
							const real_t gradphiy=gradphi(i, j, IPY);
							const real_t sign    =gradphi(i, j, IPS);
							real_t phi =U2data(i, j, IH);
							real_t phi0 =Udata(i, j, IH);
							phi =THREE_FOURTH_F*phi0+ONE_FOURTH_F*(phi-sign*params.dx*0.3*(sqrt(gradphix*gradphix+gradphiy*gradphiy)-ONE_F));
							U2data(i, j, IH)= phi;

						}
					}
				const DataArray Udata;
				const DataArray U2data;
				const DataArray gradphi;
		};// Redistancing phi WENO5

		class ComputeRedistancingPhi3Functor2D : HydroBaseFunctor2D
		{
			public:
				ComputeRedistancingPhi3Functor2D(HydroParams params_, DataArray Udata_, DataArray U2data_, DataArray gradphi_):
					HydroBaseFunctor2D(params_), Udata(Udata_),U2data(U2data_),  gradphi(gradphi_) {};
				static void apply(HydroParams params,
						DataArray Udata, DataArray U2data, DataArray gradphi,
						int nbCells)
				{
					ComputeRedistancingPhi3Functor2D functor(params, Udata, U2data, gradphi);
					Kokkos::parallel_for(nbCells, functor);
				}
				KOKKOS_INLINE_FUNCTION
					void operator()(int index) const
					{
						const int ghostWidth = params.ghostWidth;
						int i, j;
						index2coord(index, i, j, params.isize, params.jsize);

						if (j>=ghostWidth && j<=params.jmax-ghostWidth &&
								i>=ghostWidth && i<=params.imax-ghostWidth)
						{
							const real_t gradphix=gradphi(i, j, IPX);
							const real_t gradphiy=gradphi(i, j, IPY);
							const real_t sign    =gradphi(i, j, IPS);
							real_t phi0= Udata(i, j, IH);
							real_t phi =U2data(i, j, IH);
							phi =ONE_THIRD_F*phi0+TWO_THIRD_F*(phi-sign*params.dx*0.3*(sqrt(gradphix*gradphix+gradphiy*gradphiy)-ONE_F));
							U2data(i, j, IH)= phi;

						}
					}
				const DataArray Udata;
				const DataArray U2data;
				const DataArray gradphi;
		};// Redistancing phi WENO5
		class ComputePhiTransport2Functor2D : HydroBaseFunctor2D
		{
			public:
				ComputePhiTransport2Functor2D(HydroParams params_, DataArray Udata_,DataArray U2data_, 
						DataArrayConst Qdata_, DataArray gradphi_, real_t dt_):
					HydroBaseFunctor2D(params_), Udata(Udata_), U2data(U2data_),Qdata(Qdata_),gradphi(gradphi_), dt(dt_) {};
				static void apply(HydroParams params,
						DataArray Udata, DataArray U2data, DataArrayConst Qdata, DataArray gradphi, real_t dt,
						int nbCells)
				{
					ComputePhiTransport2Functor2D functor(params, Udata, U2data, Qdata, gradphi, dt);
					Kokkos::parallel_for(nbCells, functor);
				}
				KOKKOS_INLINE_FUNCTION
					void operator()(int index) const
					{
						const int ghostWidth = params.ghostWidth;
						int i, j;
						index2coord(index, i, j, params.isize, params.jsize);

						if (j>=ghostWidth && j<=params.jmax-ghostWidth &&
								i>=ghostWidth && i<=params.imax-ghostWidth)
						{
							const real_t gradphix=gradphi(i, j, IPX);
							const real_t gradphiy=gradphi(i, j, IPY);
							real_t ux=Qdata(i, j, IU);
							real_t uy=Qdata(i, j, IV);
							real_t phi=U2data(i, j, IH);
							real_t phi0=Udata(i, j, IH);

							phi=THREE_FOURTH_F*phi0+(phi-(ux*gradphix+uy*gradphiy)*dt)*ONE_FOURTH_F;
							U2data(i, j, IH)=phi;
						}
					}

				const DataArray Udata;
				const DataArray U2data;
				const DataArrayConst Qdata;
				const DataArray gradphi;
				real_t dt;
		};//ComputeTransportPhi RK3 step_2
		class ComputePhiTransport3Functor2D : HydroBaseFunctor2D
		{
			public:
				ComputePhiTransport3Functor2D(HydroParams params_, DataArray Udata_,DataArray U2data_, 
						DataArrayConst Qdata_, DataArray gradphi_, real_t dt_):
					HydroBaseFunctor2D(params_), Udata(Udata_), U2data(U2data_),Qdata(Qdata_),gradphi(gradphi_), dt(dt_) {};
				static void apply(HydroParams params,
						DataArray Udata, DataArray U2data, DataArrayConst Qdata, DataArray gradphi, real_t dt,
						int nbCells)
				{
					ComputePhiTransport3Functor2D functor(params, Udata, U2data, Qdata, gradphi, dt);
					Kokkos::parallel_for(nbCells, functor);
				}
				KOKKOS_INLINE_FUNCTION
					void operator()(int index) const
					{
						const int ghostWidth = params.ghostWidth;
						int i, j;
						index2coord(index, i, j, params.isize, params.jsize);

						if (j>=ghostWidth && j<=params.jmax-ghostWidth &&
								i>=ghostWidth && i<=params.imax-ghostWidth)
						{
							const real_t gradphix=gradphi(i, j, IPX);
							const real_t gradphiy=gradphi(i, j, IPY);
							real_t ux=Qdata(i, j, IU);
							real_t uy=Qdata(i, j, IV);
							real_t phi=U2data(i, j, IH);
							real_t phi0=Udata(i, j, IH);

							phi=ONE_THIRD_F*phi0+(phi-(ux*gradphix+uy*gradphiy)*dt)*TWO_THIRD_F;
							U2data(i, j, IH)=phi;
							gradphi(i, j, IPS)= phi /sqrt(phi *phi  +params.dx*params.dx* HALF_F * HALF_F);
						}
					}

				const DataArray Udata;
				const DataArray U2data;
				const DataArrayConst Qdata;
				const DataArray gradphi;
				real_t dt;
		};//ComputeTransportPhi_RK3_Step3
		class ComputePhiTransport1Functor2D : HydroBaseFunctor2D
		{
			public:
				ComputePhiTransport1Functor2D(HydroParams params_, DataArray Udata_,DataArray U2data_, 
						DataArrayConst Qdata_, DataArray gradphi_, real_t dt_):
					HydroBaseFunctor2D(params_), Udata(Udata_), U2data(U2data_),Qdata(Qdata_),gradphi(gradphi_), dt(dt_) {};
				static void apply(HydroParams params,
						DataArray Udata, DataArray U2data, DataArrayConst Qdata, DataArray gradphi, real_t dt,
						int nbCells)
				{
					ComputePhiTransport1Functor2D functor(params, Udata, U2data, Qdata, gradphi, dt);
					Kokkos::parallel_for(nbCells, functor);
				}
				KOKKOS_INLINE_FUNCTION
					void operator()(int index) const
					{
						const int ghostWidth = params.ghostWidth;
						int i, j;
						index2coord(index, i, j, params.isize, params.jsize);

						if (j>=ghostWidth && j<=params.jmax-ghostWidth &&
								i>=ghostWidth && i<=params.imax-ghostWidth)
						{
							const real_t gradphix=gradphi(i, j, IPX);
							const real_t gradphiy=gradphi(i, j, IPY);
							real_t ux=Qdata(i, j, IU);
							real_t uy=Qdata(i, j, IV);
							real_t phi=Udata(i, j, IH);

							phi-=(ux*gradphix+uy*gradphiy)*dt;
							U2data(i, j, IH)=phi;
						}
					}

				const DataArray Udata;
				const DataArray U2data;
				const DataArrayConst Qdata;
				const DataArray gradphi;
				real_t dt;
		};//ComputeTransportPhi_RK3_Step1
		class ComputeGradPhiTransportFunctor2D : HydroBaseFunctor2D
		{
			public:
				ComputeGradPhiTransportFunctor2D(HydroParams params_, DataArray Udata_, DataArray gradphi_):
					HydroBaseFunctor2D(params_), Udata(Udata_), gradphi(gradphi_) {};
				static void apply(HydroParams params,
						DataArray Udata,  DataArray gradphi,
						int nbCells)
				{
					ComputeGradPhiTransportFunctor2D functor(params, Udata, gradphi);
					Kokkos::parallel_for(nbCells, functor);
				}
				KOKKOS_INLINE_FUNCTION
					void operator()(int index) const
					{
						const int ghostWidth = params.ghostWidth;
						int i, j;
						index2coord(index, i, j, params.isize, params.jsize);

						if (j>=ghostWidth && j<=params.jmax-ghostWidth &&
								i>=ghostWidth && i<=params.imax-ghostWidth)
						{
							const real_t u=Udata(i, j, IU);
							const real_t v=Udata(i, j, IV);

							const real_t phimx3=Udata(i-3, j, IH);
							const real_t phimx2=Udata(i-2, j, IH);
							const real_t phimx =Udata(i-1, j, IH);
							const real_t phipx =Udata(i+1, j, IH);
							const real_t phipx2=Udata(i+2, j, IH);
							const real_t phipx3=Udata(i+3, j, IH);

							const real_t phimy3=Udata(i, j-3, IH);
							const real_t phimy2=Udata(i, j-2, IH);
							const real_t phimy =Udata(i, j-1, IH);
							const real_t phipy =Udata(i, j+1, IH);
							const real_t phipy2=Udata(i, j+2, IH);
							const real_t phipy3=Udata(i, j+3, IH);

							const real_t phiLoc=Udata(i, j,   IH);

							const real_t onesurdx=params.onesurdx;
							const real_t onesurdy=params.onesurdy;
							if (u>ZERO_F)
							{
								gradphi(i, j, IPX)=(-ONE_THIRTIETH_F*phimx3+ONE_FOURTH_F*phimx2-              phimx\
										+ONE_THIRD_F    *phiLoc  +HALF_F   *phipx-ONE_TWENTIETH_F  *phipx2)*onesurdx;
							}

							else
							{
								gradphi(i, j, IPX)=(ONE_THIRTIETH_F*phipx3-ONE_FOURTH_F*phipx2+               phipx\
										-ONE_THIRD_F    *phiLoc  -HALF_F   *phimx+ONE_TWENTIETH_F  *phimx2)*onesurdx;

							}


							if (v>ZERO_F)
							{
								gradphi(i, j, IPY)=(-ONE_THIRTIETH_F*phimy3+ONE_FOURTH_F*phimy2-              phimy\
										+ONE_THIRD_F    *phiLoc  +HALF_F   *phipy-ONE_TWENTIETH_F  *phipy2)*onesurdy;
							}
							else
							{
								gradphi(i, j, IPY)=(ONE_THIRTIETH_F*phipy3-ONE_FOURTH_F*phipy2+               phipy\
										-ONE_THIRD_F    *phiLoc  -HALF_F   *phimy+ONE_TWENTIETH_F  *phimy2)*onesurdy;

							}



						}
					}

				const DataArray Udata;
				const DataArray gradphi;
		};//ComputeTransportGradPhiFunctor_Scheme_HOUC5

		class ComputeGradPhiFunctor2D : HydroBaseFunctor2D
		{
			public:
				ComputeGradPhiFunctor2D(HydroParams params_, DataArray Udata_, DataArray gradphi_):
					HydroBaseFunctor2D(params_), Udata(Udata_), gradphi(gradphi_) {};
				static void apply(HydroParams params,
						DataArray Udata,  DataArray gradphi,
						int nbCells)
				{
					ComputeGradPhiFunctor2D functor(params, Udata, gradphi);
					Kokkos::parallel_for(nbCells, functor);
				}
				KOKKOS_INLINE_FUNCTION
					void operator()(int index) const
					{
						const int ghostWidth = params.ghostWidth;
						int i, j;
						index2coord(index, i, j, params.isize, params.jsize);

						if (j>=ghostWidth && j<=params.jmax-ghostWidth &&
								i>=ghostWidth && i<=params.imax-ghostWidth)
						{

							const real_t phimx3=Udata(i-3, j, IH);
							const real_t phimx2=Udata(i-2, j, IH);
							const real_t phimx =Udata(i-1, j, IH);
							const real_t phipx =Udata(i+1, j, IH);
							const real_t phipx2=Udata(i+2, j, IH);
							const real_t phipx3=Udata(i+3, j, IH);

							const real_t phimy3=Udata(i, j-3, IH);
							const real_t phimy2=Udata(i, j-2, IH);
							const real_t phimy =Udata(i, j-1, IH);
							const real_t phipy =Udata(i, j+1, IH);
							const real_t phipy2=Udata(i, j+2, IH);
							const real_t phipy3=Udata(i, j+3, IH);

							const real_t phiLoc=Udata(i, j,   IH);

							const real_t onesurdx=params.onesurdx;
							const real_t onesurdy=params.onesurdy;

							const real_t q1_moins_x=(phimx2-phimx3)*onesurdx;
							const real_t q2_moins_x=(phimx -phimx2)*onesurdx;
							const real_t q3_moins_x=(phiLoc-phimx )*onesurdx;
							const real_t q4_moins_x=(phipx -phiLoc)*onesurdx;
							const real_t q5_moins_x=(phipx2-phipx )*onesurdx;

							const real_t dphi0_moins_x= ONE_THIRD_F*q1_moins_x-SEVEN_SIXTH_F*q2_moins_x+ELEVEN_SIXTH_F*q3_moins_x;
							const real_t dphi1_moins_x=-ONE_SIXTH_F*q2_moins_x+FIVE_SIXTH_F *q3_moins_x+ONE_THIRD_F   *q4_moins_x;
							const real_t dphi2_moins_x= ONE_THIRD_F*q3_moins_x+FIVE_SIXTH_F *q4_moins_x-ONE_SIXTH_F   *q5_moins_x;

							const real_t IS0_moins_x=THIRTEEN_TWELFTH_F*(q1_moins_x-TWO_F *q2_moins_x+        q3_moins_x)*(q1_moins_x-TWO_F *q2_moins_x+        q3_moins_x)\
										 +ONE_FOURTH_F     *(q1_moins_x-FOUR_F*q2_moins_x+THREE_F*q3_moins_x)*(q1_moins_x-FOUR_F*q2_moins_x+THREE_F*q3_moins_x);
							const real_t IS1_moins_x=THIRTEEN_TWELFTH_F*(q2_moins_x-TWO_F *q3_moins_x+        q4_moins_x)*(q2_moins_x-TWO_F *q3_moins_x+        q4_moins_x)\
										 +ONE_FOURTH_F     *(q2_moins_x-                          q4_moins_x)*(q2_moins_x                          -q4_moins_x);
							const real_t IS2_moins_x=THIRTEEN_TWELFTH_F*(q3_moins_x-TWO_F *q4_moins_x+        q5_moins_x)*(q3_moins_x-TWO_F *q4_moins_x+        q5_moins_x)\
										 +ONE_FOURTH_F     *(THREE_F*q3_moins_x-FOUR_F*q4_moins_x+q5_moins_x)*(THREE_F*q3_moins_x-FOUR_F*q4_moins_x+q5_moins_x);

							const real_t onesurIS0_moinsx=ONE_F/(epsi+IS0_moins_x);
							const real_t onesurIS1_moinsx=ONE_F/(epsi+IS1_moins_x);
							const real_t onesurIS2_moinsx=ONE_F/(epsi+IS2_moins_x);

							const real_t alpha0_moins_x=ZERO_ONE_F  *onesurIS0_moinsx*onesurIS0_moinsx;
							const real_t alpha1_moins_x=ZERO_SIX_F  *onesurIS1_moinsx*onesurIS1_moinsx;
							const real_t alpha2_moins_x=ZERO_THREE_F*onesurIS2_moinsx*onesurIS2_moinsx;


							const real_t onesursumalpha_moins_x=ONE_F/(alpha0_moins_x +alpha1_moins_x +alpha2_moins_x );

							const real_t omega0_moins_x=alpha0_moins_x *onesursumalpha_moins_x;
							const real_t omega1_moins_x=alpha1_moins_x *onesursumalpha_moins_x;
							const real_t omega2_moins_x=alpha2_moins_x *onesursumalpha_moins_x;


							const real_t a=omega0_moins_x*dphi0_moins_x+omega1_moins_x*dphi1_moins_x\
								       +omega2_moins_x*dphi2_moins_x;

							const real_t q1_plus_x =(phipx3-phipx2)*onesurdx;
							const real_t q2_plus_x =(phipx2-phipx )*onesurdx;
							const real_t q3_plus_x =(phipx -phiLoc)*onesurdx;
							const real_t q4_plus_x =(phiLoc -phimx)*onesurdx;
							const real_t q5_plus_x =(phimx -phimx2)*onesurdx;


							const real_t dphi0_plus_x = ONE_THIRD_F*q1_plus_x -SEVEN_SIXTH_F*q2_plus_x +ELEVEN_SIXTH_F*q3_plus_x ;
							const real_t dphi1_plus_x =-ONE_SIXTH_F*q2_plus_x +FIVE_SIXTH_F *q3_plus_x +ONE_THIRD_F   *q4_plus_x ;
							const real_t dphi2_plus_x = ONE_THIRD_F*q3_plus_x +FIVE_SIXTH_F *q4_plus_x -ONE_SIXTH_F   *q5_plus_x ;


							const real_t IS0_plus_x =THIRTEEN_TWELFTH_F*(q1_plus_x -TWO_F *q2_plus_x +        q3_plus_x) *(q1_plus_x - TWO_F*q2_plus_x +        q3_plus_x)\
										 +ONE_FOURTH_F     *(q1_plus_x -FOUR_F*q2_plus_x +THREE_F*q3_plus_x) *(q1_plus_x -FOUR_F*q2_plus_x +THREE_F*q3_plus_x) ;
							const real_t IS1_plus_x =THIRTEEN_TWELFTH_F*(q2_plus_x -TWO_F *q3_plus_x +        q4_plus_x) *(q2_plus_x - TWO_F*q3_plus_x +        q4_plus_x)\
										 +ONE_FOURTH_F     *(q2_plus_x                           -q4_plus_x) *(q2_plus_x        -q4_plus_x) ;
							const real_t IS2_plus_x =THIRTEEN_TWELFTH_F*(q3_plus_x -TWO_F *q4_plus_x         +q5_plus_x) *(q3_plus_x -TWO_F*q4_plus_x +q5_plus_x )\
										 +ONE_FOURTH_F     *(THREE_F*q3_plus_x -FOUR_F*q4_plus_x +q5_plus_x) *(THREE_F*q3_plus_x -FOUR_F*q4_plus_x +q5_plus_x) ;


							const real_t onesurIS0_plusx=ONE_F/(epsi+IS0_plus_x);
							const real_t onesurIS1_plusx=ONE_F/(epsi+IS1_plus_x);
							const real_t onesurIS2_plusx=ONE_F/(epsi+IS2_plus_x);

							const real_t alpha0_plus_x =ZERO_ONE_F  *onesurIS0_plusx*onesurIS0_plusx;
							const real_t alpha1_plus_x =ZERO_SIX_F  *onesurIS1_plusx*onesurIS1_plusx;
							const real_t alpha2_plus_x =ZERO_THREE_F*onesurIS2_plusx*onesurIS2_plusx;

							const real_t onesursumalpha_plus_x=ONE_F/(alpha0_plus_x +alpha1_plus_x +alpha2_plus_x );

							const real_t omega0_plus_x =alpha0_plus_x *onesursumalpha_plus_x;
							const real_t omega1_plus_x =alpha1_plus_x *onesursumalpha_plus_x;
							const real_t omega2_plus_x =alpha2_plus_x *onesursumalpha_plus_x;

							const real_t b=omega0_plus_x *dphi0_plus_x +omega1_plus_x *dphi1_plus_x \
								       +omega2_plus_x*dphi2_plus_x;

							const real_t q1_moins_y=(phimy2-phimy3)*onesurdy;
							const real_t q2_moins_y=(phimy -phimy2)*onesurdy;
							const real_t q3_moins_y=(phiLoc-phimy )*onesurdy;
							const real_t q4_moins_y=(phipy -phiLoc)*onesurdy;
							const real_t q5_moins_y=(phipy2-phipy )*onesurdy;

							const real_t dphi0_moins_y= ONE_THIRD_F*q1_moins_y-SEVEN_SIXTH_F*q2_moins_y+ELEVEN_SIXTH_F*q3_moins_y;
							const real_t dphi1_moins_y=-ONE_SIXTH_F*q2_moins_y+FIVE_SIXTH_F *q3_moins_y+ONE_THIRD_F   *q4_moins_y;
							const real_t dphi2_moins_y= ONE_THIRD_F*q3_moins_y+FIVE_SIXTH_F *q4_moins_y-ONE_SIXTH_F   *q5_moins_y;

							const real_t IS0_moins_y=THIRTEEN_TWELFTH_F*(q1_moins_y-TWO_F *q2_moins_y+        q3_moins_y)*(q1_moins_y-TWO_F *q2_moins_y+        q3_moins_y)\
										 +ONE_FOURTH_F     *(q1_moins_y-FOUR_F*q2_moins_y+THREE_F*q3_moins_y)*(q1_moins_y-FOUR_F*q2_moins_y+THREE_F*q3_moins_y);
							const real_t IS1_moins_y=THIRTEEN_TWELFTH_F*(q2_moins_y-TWO_F *q3_moins_y+        q4_moins_y)*(q2_moins_y-TWO_F *q3_moins_y+        q4_moins_y)\
										 +ONE_FOURTH_F     *(q2_moins_y      -q4_moins_y)                    *(q2_moins_y      -q4_moins_y);
							const real_t IS2_moins_y=THIRTEEN_TWELFTH_F*(q3_moins_y-TWO_F*q4_moins_y +        q5_moins_y)*(q3_moins_y-TWO_F*q4_moins_y+q5_moins_y)\
										 +ONE_FOURTH_F     *(THREE_F*q3_moins_y-FOUR_F*q4_moins_y+q5_moins_y)*(THREE_F*q3_moins_y-FOUR_F*q4_moins_y+q5_moins_y);

							const real_t onesurIS0_moinsy=ONE_F/(epsi+IS0_moins_y);
							const real_t onesurIS1_moinsy=ONE_F/(epsi+IS1_moins_y);
							const real_t onesurIS2_moinsy=ONE_F/(epsi+IS2_moins_y);

							const real_t alpha0_moins_y=ZERO_ONE_F  *onesurIS0_moinsy*onesurIS0_moinsy;
							const real_t alpha1_moins_y=ZERO_SIX_F  *onesurIS1_moinsy*onesurIS1_moinsy;
							const real_t alpha2_moins_y=ZERO_THREE_F*onesurIS2_moinsy*onesurIS2_moinsy;

							const real_t onesursumalpha_moins_y=ONE_F/(alpha0_moins_y +alpha1_moins_y +alpha2_moins_y );

							const real_t omega0_moins_y=alpha0_moins_y*onesursumalpha_moins_y;
							const real_t omega1_moins_y=alpha1_moins_y*onesursumalpha_moins_y;
							const real_t omega2_moins_y=alpha2_moins_y*onesursumalpha_moins_y;

							const real_t c=omega0_moins_y*dphi0_moins_y+omega1_moins_y*dphi1_moins_y\
								       +omega2_moins_y*dphi2_moins_y;

							const real_t q1_plus_y =(phipy3-phipy2)*onesurdy;
							const real_t q2_plus_y =(phipy2-phipy )*onesurdy;
							const real_t q3_plus_y =(phipy -phiLoc)*onesurdy;
							const real_t q4_plus_y =(phiLoc-phimy )*onesurdy;
							const real_t q5_plus_y =(phimy -phimy2)*onesurdy;


							const real_t dphi0_plus_y = ONE_THIRD_F*q1_plus_y -SEVEN_SIXTH_F*q2_plus_y +ELEVEN_SIXTH_F*q3_plus_y ;
							const real_t dphi1_plus_y =-ONE_SIXTH_F*q2_plus_y +FIVE_SIXTH_F *q3_plus_y +ONE_THIRD_F   *q4_plus_y ;
							const real_t dphi2_plus_y = ONE_THIRD_F*q3_plus_y +FIVE_SIXTH_F *q4_plus_y -ONE_SIXTH_F   *q5_plus_y ;


							const real_t IS0_plus_y =THIRTEEN_TWELFTH_F*(q1_plus_y -TWO_F *q2_plus_y +        q3_plus_y) *(q1_plus_y -TWO_F *q2_plus_y +        q3_plus_y)\
										 +ONE_FOURTH_F     *(q1_plus_y -FOUR_F*q2_plus_y +THREE_F*q3_plus_y) *(q1_plus_y -FOUR_F*q2_plus_y +THREE_F*q3_plus_y) ;
							const real_t IS1_plus_y =THIRTEEN_TWELFTH_F*(q2_plus_y -TWO_F *q3_plus_y +        q4_plus_y) *(q2_plus_y -TWO_F *q3_plus_y +        q4_plus_y)\
										 +ONE_FOURTH_F     *(q2_plus_y        -q4_plus_y)                    *(q2_plus_y        -q4_plus_y) ;
							const real_t IS2_plus_y =THIRTEEN_TWELFTH_F*(q3_plus_y -TWO_F *q4_plus_y +        q5_plus_y) *(q3_plus_y -TWO_F *q4_plus_y +        q5_plus_y)\
										 +ONE_FOURTH_F     *(THREE_F*q3_plus_y -FOUR_F*q4_plus_y +q5_plus_y )*(THREE_F*q3_plus_y -FOUR_F*q4_plus_y +q5_plus_y) ;


							const real_t onesurIS0_plusy=ONE_F/(epsi+IS0_plus_y);
							const real_t onesurIS1_plusy=ONE_F/(epsi+IS1_plus_y);
							const real_t onesurIS2_plusy=ONE_F/(epsi+IS2_plus_y);

							const real_t alpha0_plus_y =ZERO_ONE_F  *onesurIS0_plusy*onesurIS0_plusy;
							const real_t alpha1_plus_y =ZERO_SIX_F  *onesurIS1_plusy*onesurIS1_plusy;
							const real_t alpha2_plus_y =ZERO_THREE_F*onesurIS2_plusy*onesurIS2_plusy;

							const real_t onesursumalpha_plus_y=ONE_F/(alpha0_plus_y +alpha1_plus_y +alpha2_plus_y );

							const real_t omega0_plus_y =alpha0_plus_y *onesursumalpha_plus_y;
							const real_t omega1_plus_y =alpha1_plus_y *onesursumalpha_plus_y;
							const real_t omega2_plus_y =alpha2_plus_y *onesursumalpha_plus_y;

							const real_t d=omega0_plus_y *dphi0_plus_y +omega1_plus_y *dphi1_plus_y \
								       +omega2_plus_y *dphi2_plus_y ;
							if (phiLoc>0.)
							{
								const real_t max_x=(a>0.)?a:0.;
								const real_t min_x=(b>0.)?0.:b;

								const real_t max_y=(c>0.)?c:0.;
								const real_t min_y=(d>0.)?0.:d;

								gradphi(i, j, IPX)=fabs(max_x)>fabs(min_x)?max_x:min_x;
								gradphi(i, j, IPY)=fabs(max_y)>fabs(min_y)?max_y:min_y;



							}
							else
							{
								const real_t max_x=(b>0.)?b:0.;
								const real_t min_x=(a>0.)?0.:a;

								const real_t max_y=(d>0.)?d:0.;
								const real_t min_y=(c>0.)?0.:c;

								gradphi(i, j, IPX)=fabs(max_x)>fabs(min_x)?max_x:min_x;
								gradphi(i, j, IPY)=fabs(max_y)>fabs(min_y)?max_y:min_y;

							}

						}
					}

				const DataArray Udata;
				const DataArray gradphi;
		};//ComputeGrad_Phi_Redistancing_Scheme_WENO5

		class ComputeTransportPhiXFunctor2D : HydroBaseFunctor2D
		{
			public:
				ComputeTransportPhiXFunctor2D(HydroParams params_, DataArray Udata_, DataArray U2data_, DataArrayConst Qdata_,DataArray gradphi_, real_t dt_):
					HydroBaseFunctor2D(params_), Udata(Udata_), U2data(U2data_), Qdata(Qdata_), gradphi(gradphi_), dt(dt_) {};
				static void apply(HydroParams params,
						DataArray Udata, DataArray U2data, DataArrayConst Qdata, DataArray gradphi, real_t dt,
						int nbCells)
				{
					ComputeTransportPhiXFunctor2D functor(params, Udata, U2data, Qdata, gradphi, dt);
					Kokkos::parallel_for(nbCells, functor);
				}
				KOKKOS_INLINE_FUNCTION
					void operator()(int index) const
					{
						const int ghostWidth = params.ghostWidth;
						int i, j;
						index2coord(index, i, j, params.isize, params.jsize);

						if (j>=ghostWidth && j<=params.jmax-ghostWidth &&
								i>=ghostWidth && i<=params.imax-ghostWidth)
						{
							const real_t ux=Qdata(i,j, IU);
							const real_t xnux=fabs(ux)*dt*params.onesurdx;
							const real_t xcoex2=HALF_F-xnux*HALF_F;
							const real_t xcoex3=xcoex2*(ONE_THIRD_F+xnux*ONE_THIRD_F);
							const real_t xcoex4=xcoex3*(xnux*ONE_FOURTH_F-HALF_F);
							const real_t xcoex5=xcoex4*(xnux*0.2-0.6);

							real_t phiLoc=Udata(i  , j, IH);

							real_t gradphix;
							if (ux>0.)
							{

								const real_t coe1=xcoex5+xcoex4;
								const real_t coe2=xcoex3-4.*xcoex4-5.*xcoex5;
								const real_t coe3=xcoex5*10.+6.*xcoex4-3.*xcoex3+xcoex2-1.;
								const real_t coe4=1.-2.*xcoex2+3*xcoex3-4.*xcoex4-10.*xcoex5;
								const real_t coe5=5.*xcoex5+xcoex4-xcoex3+xcoex2;

								const real_t phimx3=Udata(i-3, j, IH);
								const real_t phimx2=Udata(i-2, j, IH);
								const real_t phimx =Udata(i-1, j, IH);
								const real_t phipx =Udata(i+1, j, IH);
								const real_t phipx2=Udata(i+2, j, IH);

								gradphix=(coe1*phimx3+coe2*phimx2+coe3*phimx+coe4*phiLoc\
										+coe5*phipx-xcoex5*phipx2)*params.onesurdx;

							}
							else
							{
								const real_t coe1=-xcoex5-xcoex4;
								const real_t coe2=-xcoex3+4.*xcoex4+5.*xcoex5;
								const real_t coe3=-xcoex5*10.-6.*xcoex4+3.*xcoex3-xcoex2+1.;
								const real_t coe4=-1.+2.*xcoex2-3*xcoex3+4.*xcoex4+10.*xcoex5;
								const real_t coe5=-5.*xcoex5-xcoex4+xcoex3-xcoex2;

								const real_t phimx2=Udata(i-2, j, IH);
								const real_t phimx =Udata(i-1, j, IH);
								const real_t phipx =Udata(i+1, j, IH);
								const real_t phipx2=Udata(i+2, j, IH);
								const real_t phipx3=Udata(i+3, j, IH);

								gradphix=(coe1*phipx3+coe2*phipx2+coe3*phipx+coe4*phiLoc\
										+coe5*phimx+xcoex5*phimx2)*params.onesurdx;
							}


							phiLoc-=gradphix*ux*dt;
							U2data(i, j, IH)=phiLoc;
							gradphi(i, j, IPS)= phiLoc /sqrt(phiLoc *phiLoc  +params.dx*params.dx);


						}
					}

				const DataArray Udata;
				const DataArray U2data;
				const DataArrayConst Qdata;
				const DataArray gradphi;
				real_t dt;
		};//ComputeTransportPhiXFunctor_Scheme_OS5

		class ComputeTransportPhiYFunctor2D : HydroBaseFunctor2D
		{
			public:
				ComputeTransportPhiYFunctor2D(HydroParams params_, DataArray Udata_, DataArray U2data_, DataArrayConst Qdata_, DataArray gradphi_, real_t dt_):
					HydroBaseFunctor2D(params_), Udata(Udata_), U2data(U2data_), Qdata(Qdata_), gradphi(gradphi_), dt(dt_) {};
				static void apply(HydroParams params,
						DataArray Udata, DataArray U2data, DataArrayConst Qdata, DataArray gradphi, real_t dt,
						int nbCells)
				{
					ComputeTransportPhiYFunctor2D functor(params, Udata, U2data, Qdata, gradphi, dt);
					Kokkos::parallel_for(nbCells, functor);
				}
				KOKKOS_INLINE_FUNCTION
					void operator()(int index) const
					{
						const int ghostWidth = params.ghostWidth;
						int i, j;
						index2coord(index, i, j, params.isize, params.jsize);

						if (j>=ghostWidth && j<=params.jmax-ghostWidth &&
								i>=ghostWidth && i<=params.imax-ghostWidth)
						{
							const real_t uy=Qdata(i,j, IV);
							const real_t xnuy=fabs(uy)*dt*params.onesurdy;
							const real_t xcoey2=HALF_F-xnuy*HALF_F;
							const real_t xcoey3=xcoey2*(ONE_THIRD_F+xnuy*ONE_THIRD_F);
							const real_t xcoey4=xcoey3*(xnuy*ONE_FOURTH_F-HALF_F);
							const real_t xcoey5=xcoey4*(xnuy*0.2-0.6);
							real_t gradphiy;

							real_t phiLoc=Udata(i  , j, IH);

							if (uy>0.)
							{
								const real_t coe1=xcoey5+xcoey4;
								const real_t coe2=xcoey3-4.*xcoey4-5.*xcoey5;
								const real_t coe3=xcoey5*10.+6.*xcoey4-3.*xcoey3+xcoey2-1.;
								const real_t coe4=1.-2.*xcoey2+3*xcoey3-4.*xcoey4-10.*xcoey5;
								const real_t coe5=5.*xcoey5+xcoey4-xcoey3+xcoey2;

								const real_t phimy3=Udata(i, j-3, IH);
								const real_t phimy2=Udata(i, j-2, IH);
								const real_t phimy =Udata(i, j-1, IH);
								const real_t phipy =Udata(i, j+1, IH);
								const real_t phipy2=Udata(i, j+2, IH);

								gradphiy=(coe1*phimy3+coe2*phimy2+coe3*phimy\
										+coe4*phiLoc+coe5*phipy-xcoey5*phipy2)*params.onesurdy;
							}
							else
							{
								const real_t coe1=-xcoey5-xcoey4;
								const real_t coe2=-xcoey3+4.*xcoey4+5.*xcoey5;
								const real_t coe3=-xcoey5*10.-6.*xcoey4+3.*xcoey3-xcoey2+1.;
								const real_t coe4=-1.+2.*xcoey2-3*xcoey3+4.*xcoey4+10.*xcoey5;
								const real_t coe5=-5.*xcoey5-xcoey4+xcoey3-xcoey2;

								const real_t phimy2=Udata(i, j-2, IH);
								const real_t phimy =Udata(i, j-1, IH);
								const real_t phipy =Udata(i, j+1, IH);
								const real_t phipy2=Udata(i, j+2, IH);
								const real_t phipy3=Udata(i, j+3, IH);

								gradphiy =(coe1*phipy3+coe2*phipy2+coe3*phipy\
										+coe4*phiLoc+coe5*phimy+xcoey5*phimy2)*params.onesurdy;
							}
							phiLoc-=gradphiy*uy*dt;
							U2data(i, j, IH)=phiLoc;
							gradphi(i, j, IPS)= phiLoc /sqrt(phiLoc *phiLoc  +params.dx*params.dx);




						}
					}

				const DataArray Udata;
				const DataArray U2data;
				const DataArrayConst Qdata;
				const DataArray gradphi;
				real_t dt;
		};//ComputeTransportPhiYFunctor_Scheme_OS5
	}
}
