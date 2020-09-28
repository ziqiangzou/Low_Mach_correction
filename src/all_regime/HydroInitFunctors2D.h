#pragma once

#include <limits> // for std::numeric_limits
#ifdef __CUDA_ARCH__
#include <math_constants.h> // for cuda math constants, e.g. CUDART_INF
#endif // __CUDA_ARCH__

#include "shared/kokkos_shared.h"
#include "HydroBaseFunctor2D.h"

// init conditions
#include "shared/BlastParams.h"
#include "shared/DamBreakParams.h"
#include "shared/GreshoParams.h"
#include "shared/IsentropicVortexParams.h"
#include "shared/PoiseuilleParams.h"
#include "shared/RayleighBenardParams.h"
#include "shared/NonIsothermParams.h"
#include "shared/StefanthermParams.h"
#include "shared/SuckingthermParams.h"
#include "shared/StaticBubbleParams.h"
#include "shared/TransientProcessParams.h"
#include "shared/RiemannProblemParams.h"

namespace euler_kokkos { namespace all_regime
	{

		class InitDamBreakFunctor2D : HydroBaseFunctor2D
		{
			public:
				InitDamBreakFunctor2D(HydroParams params_,
						DamBreakParams dbParams_,
						DataArray Udata_):
					HydroBaseFunctor2D(params_), dbParams(dbParams_), Udata(Udata_) {}

				static void apply(HydroParams params, DamBreakParams dbParams,
						DataArray Udata, int nbCells)
				{
					InitDamBreakFunctor2D functor(params, dbParams, Udata);
					Kokkos::parallel_for(nbCells, functor);
				}

				KOKKOS_INLINE_FUNCTION
					void operator()(const int i) const
					{
						const real_t ghostWidth = params.ghostWidth;
						const real_t jmin = params.jmin;
						const real_t jmax = params.jmax;
						const real_t ymax = params.ymax;
						const real_t nx = params.nx;
						const real_t ny = params.ny;
						const real_t dx = params.dx;
						const real_t dy = params.dy;

						const real_t x_int = dbParams.interface_position;
						const real_t g_y=params.settings.g_y;

#ifdef USE_MPI
						const int i_mpi = params.myMpiPos[IX];
						const int j_mpi = params.myMpiPos[IY];
#else
						const int i_mpi = 0;
						const int j_mpi = 0;
#endif

						if (i >= params.imin && i <= params.imax)
						{
							const real_t x = params.xmin + (HALF_F + i + nx*i_mpi - ghostWidth)*dx;


							for (int j=jmax-ghostWidth; j>=jmin+ghostWidth; --j)
							{
								const real_t y = params.ymin + (HALF_F + j + ny*j_mpi - ghostWidth)*dy;
								real_t rho    = ((x < x_int&&y<0.5*x_int) ? 1000.0 :  1.0);
								HydroState q;
								q[ID] = rho;
								if ((x<x_int&&y<0.5*x_int))
									q[IP] = params.Astate0-g_y*HALF_F*x_int+(y+HALF_F*x_int-ymax)*g_y*rho;
								else
									q[IP]=params.Astate0+rho*g_y*(y-ymax);

								if (x<x_int&&y<0.5*x_int)
								{
									q[IH]=fmax((x-x_int),(y-HALF_F*x_int));
								}
								else if (x>=x_int&&y<0.5*x_int)
								{
									q[IH]=x-x_int;
								}
								else if (x<x_int&&y>=0.5*x_int)
								{
									q[IH]=y-x_int*HALF_F;
								}
								else
								{
									q[IH]=sqrt((x-x_int)*(x-x_int)+(y-x_int*HALF_F)*(y-x_int*HALF_F));
								}
								q[IU] = ZERO_F;
								q[IV] = ZERO_F;

								setHydroState(Udata, computeConservatives(q), i, j);

							}
						}
					}

				DamBreakParams dbParams;
				DataArray Udata;
		}; // InitDamBreakFunctor2D


		class InitPoiseuilleFunctor2D : HydroBaseFunctor2D
		{
			public:
				InitPoiseuilleFunctor2D(HydroParams params_,
						PoiseuilleParams poiseuilleParams_,
						DataArray Udata_):
					HydroBaseFunctor2D(params_),
					poiseuilleParams(poiseuilleParams_),
					Udata(Udata_) {};

				static void apply(HydroParams params,
						PoiseuilleParams poiseuilleParams,
						DataArray Udata, int nbCells)
				{
					InitPoiseuilleFunctor2D functor(params, poiseuilleParams, Udata);
					Kokkos::parallel_for(nbCells, functor);
				}

				KOKKOS_INLINE_FUNCTION
					void operator()(const int index) const
					{
						const real_t ghostWidth = params.ghostWidth;
						const real_t xmin = params.xmin;
						const real_t ymin = params.ymin;
						const real_t nx = params.nx;
						const real_t ny = params.ny;
						const real_t dx = params.dx;
						const real_t dy = params.dy;

						int i,j;
						index2coord(index, i, j, params.isize, params.jsize);
#ifdef USE_MPI
						const int i_mpi = params.myMpiPos[IX];
						const int j_mpi = params.myMpiPos[IY];
#else
						const int i_mpi = 0;
						const int j_mpi = 0;
#endif
						const real_t pressure_gradient = poiseuilleParams.poiseuille_pressure_gradient;
						const real_t p0 = poiseuilleParams.poiseuille_pressure0;
						const real_t gamma0 = params.settings.gamma0;
						const real_t x = xmin + (HALF_F + i + nx*i_mpi-ghostWidth)*dx;
						const real_t y = ymin + (HALF_F + j + ny*j_mpi-ghostWidth)*dy;

						const real_t d = poiseuilleParams.poiseuille_density;
						const real_t u = ZERO_F;
						const real_t v = ZERO_F;
						real_t p;
						if (poiseuilleParams.poiseuille_flow_direction == IX)
						{
							p = p0 + (x - xmin) * pressure_gradient;
						}
						else
						{
							p = p0 + (y - ymin) * pressure_gradient;
						}

						Udata(i, j, ID) = d;
						Udata(i, j, IU) = d * u;
						Udata(i, j, IV) = d * v;
						Udata(i, j, IP) = p / (gamma0-ONE_F) + HALF_F * d * (u*u+v*v);
					}

				PoiseuilleParams poiseuilleParams;
				DataArray Udata;
		}; // InitPoiseuilleFunctor2D
		class InitTransientFunctor2D : HydroBaseFunctor2D
		{
			public:
				InitTransientFunctor2D(HydroParams params_,
						TransientParams transientParams_,
						DataArray Udata_):
					HydroBaseFunctor2D(params_), transientParams(transientParams_),
					Udata(Udata_){};

				static void apply(HydroParams params,
						TransientParams transientParams,
						DataArray Udata, int nbCells)
				{
					InitTransientFunctor2D functor(params, transientParams, Udata);
					Kokkos::parallel_for(nbCells, functor);
				}

				KOKKOS_INLINE_FUNCTION
					void operator()(const int index) const
					{
						const real_t ghostWidth = params.ghostWidth;
						const real_t ny = params.ny;
						const real_t nx = params.nx;
						const real_t dy = params.dy;
						const real_t dx = params.dx;

						const real_t T = transientParams.transient_fluid_temp;

						int i,j;
						index2coord(index, i, j, params.isize, params.jsize);

#ifdef USE_MPI
						const int i_mpi = params.myMpiPos[IX];
						const int j_mpi = params.myMpiPos[IY];
#else
						const int j_mpi = 0;
						const int i_mpi = 0;
#endif
						const real_t y = params.ymin + (HALF_F + j + ny*j_mpi - ghostWidth)*dy;
						const real_t x = params.xmin + (HALF_F + i + nx*i_mpi - ghostWidth)*dx;


						HydroState q;
						q[IH] =0.00002- sqrt((x-0.00005)*(x-0.00005)+(y-0.00005)*(y-0.00005));
						const real_t Rstar = q[IH]>ZERO_F? params.settings.Rstar0:params.settings.Rstar1;
						const real_t Bstate = q[IH]>ZERO_F? params.Bstate0:params.Bstate1;
						q[ID] = q[IH]>ZERO_F?ONE_F : 1000.;
						q[IP] = q[ID] *  Rstar * T - Bstate;
						q[IU] = ZERO_F;
						q[IV] = ZERO_F;

						setHydroState(Udata, computeConservatives(q), i, j);

					}

				const  TransientParams transientParams;
				DataArray Udata;
		}; // InitRayleighBenardFunctor2D
		class InitSuckingthermFunctor2D : HydroBaseFunctor2D
		{
			public:
				InitSuckingthermFunctor2D(HydroParams params_,
						SuckingthermParams suckingthermParams_,
						DataArray Udata_):
					HydroBaseFunctor2D(params_), suckingthermParams(suckingthermParams_),
					Udata(Udata_){};

				static void apply(HydroParams params,
						SuckingthermParams suckingthermParams,
						DataArray Udata, int nbCells)
				{
					InitSuckingthermFunctor2D functor(params, suckingthermParams, Udata);
					Kokkos::parallel_for(nbCells, functor);
				}

				KOKKOS_INLINE_FUNCTION
					void operator()(const int index) const
					{
						const real_t ghostWidth = params.ghostWidth;
						const real_t ny = params.ny;
						const real_t dy = params.dy;

						const real_t Tsat = params.settings.Tsat;
						const real_t Twall = suckingthermParams.wall_temp;
						const real_t time = suckingthermParams.time;

						int i,j;
						index2coord(index, i, j, params.isize, params.jsize);

#ifdef USE_MPI
						const int i_mpi = params.myMpiPos[IY];
#else
						const int j_mpi = 0;
#endif
						const real_t B0 = params.settings.Rstar1 / params.settings.latent_heat; 
						const real_t pi = acos(-ONE_F);
						real_t P = sqrt(TWO_F/pi)*(Twall - Tsat)*B0;
						real_t dpsi0 =P;
						for (int l=0; l<10; l++)
						{
							const real_t dpsi1 = dpsi0 * exp(dpsi0*dpsi0*HALF_F);
							dpsi0 -=  HALF_F * (dpsi1-P);
						}

						const real_t y = params.ymin + (HALF_F + j + ny*j_mpi - ghostWidth)*dy;
						const real_t Interface_position= dpsi0 * suckingthermParams.liquid_density / suckingthermParams.vapor_density * sqrt(TWO_F * params.settings.kappa1/suckingthermParams.liquid_density / params.settings.Rstar1 * time);

						HydroState q;

						q[IH] = y - (params.ymax - Interface_position) ;
						const real_t Rstar = q[IH]>ZERO_F? params.settings.Rstar0:params.settings.Rstar1;
						const real_t Bstate = q[IH]>ZERO_F? params.Bstate0:params.Bstate1;
						const real_t barotropic = q[IH]>ZERO_F? params.settings.barotropic0:params.settings.barotropic1;

						q[ID] = q[IH]>ZERO_F?  suckingthermParams.vapor_density :suckingthermParams.liquid_density;
						const real_t x0 = dpsi0 / sqrt(TWO_F); 
						const real_t x1 = (fabs(q[IH])/sqrt(TWO_F * params.settings.kappa1/params.settings.Rstar1/suckingthermParams.liquid_density * time)+dpsi0) / sqrt(TWO_F)/sqrt(1.2); 
						const real_t erfx0 = x0 <HALF_F ? TWO_F * exp(-x0*x0)/sqrt(pi)*(x0 + TWO_THIRD_F*x0*x0*x0+4./15.*x0*x0*x0*x0*x0) : ONE_F - exp(-1.9 *pow(x0, 1.3));
						const real_t erfx1 = x1 <HALF_F ? TWO_F * exp(-x1*x1)/sqrt(pi)*(x1 + TWO_THIRD_F*x1*x1*x1+4./15.*x1*x1*x1*x1*x1) : ONE_F - exp(-1.9 *pow(x1, 1.3));

						const real_t T = q[IH] > ZERO_F ?Tsat : Tsat + ( Twall - Tsat )* (erfx1 -erfx0 + erfx0) ;

						if (!barotropic)
							q[IP] = q[ID] *  Rstar * T - Bstate;
						else
							q[IP] = T;
						const real_t deltav=HALF_F * dpsi0 * suckingthermParams.liquid_density / suckingthermParams.vapor_density * sqrt(TWO_F * params.settings.kappa1/suckingthermParams.liquid_density / params.settings.Rstar1) /sqrt(time);

						q[IU] = ZERO_F;
						q[IV] = q[IH]>ZERO_F? ZERO_F : -deltav;

						setHydroState(Udata, computeConservatives(q), i, j);

					}

				const SuckingthermParams suckingthermParams;
				DataArray Udata;
		}; // InitSuckingtherm2D
		class InitStefanthermFunctor2D : HydroBaseFunctor2D
		{
			public:
				InitStefanthermFunctor2D(HydroParams params_,
						StefanthermParams stefanthermParams_,
						DataArray Udata_):
					HydroBaseFunctor2D(params_), stefanthermParams(stefanthermParams_),
					Udata(Udata_){};

				static void apply(HydroParams params,
						StefanthermParams stefanthermParams,
						DataArray Udata, int nbCells)
				{
					InitStefanthermFunctor2D functor(params, stefanthermParams, Udata);
					Kokkos::parallel_for(nbCells, functor);
				}

				KOKKOS_INLINE_FUNCTION
					void operator()(const int index) const
					{
						const real_t ghostWidth = params.ghostWidth;
						const real_t ny = params.ny;
						const real_t dy = params.dy;

						const real_t Tsat = params.settings.Tsat;
						const real_t Twall = stefanthermParams.wall_temp;
						const real_t Interface_position = stefanthermParams.interface_position;

						int i,j;
						index2coord(index, i, j, params.isize, params.jsize);

#ifdef USE_MPI
						const int i_mpi = params.myMpiPos[IY];
#else
						const int j_mpi = 0;
#endif
						const real_t y = params.ymin + (HALF_F + j + ny*j_mpi - ghostWidth)*dy;

						HydroState q;

						q[IH] = (y - Interface_position) ;
						const real_t Rstar = q[IH]>ZERO_F? params.settings.Rstar0:params.settings.Rstar1;
						const real_t Bstate = q[IH]>ZERO_F? params.Bstate0:params.Bstate1;
						const real_t barotropic = q[IH]>ZERO_F? params.settings.barotropic0:params.settings.barotropic1;

						q[ID] = q[IH]>ZERO_F?  stefanthermParams.vapor_density :stefanthermParams.liquid_density;

						const real_t T = q[IH] > ZERO_F ?Tsat + ( Twall - Tsat )* q[IH] / (params.ymax - Interface_position) : Tsat;

						if (!barotropic)
							q[IP] = q[ID] *  Rstar * T - Bstate;
						else
							q[IP] = T;
						const real_t deltav = (Twall - Tsat) * params.settings.kappa0/(params.ymax - Interface_position)/params.settings.latent_heat*(ONE_F/stefanthermParams.vapor_density-ONE_F/stefanthermParams.liquid_density);

						q[IU] = ZERO_F;
						q[IV] = q[IH]>ZERO_F? ZERO_F : -deltav;

						setHydroState(Udata, computeConservatives(q), i, j);

					}

				const StefanthermParams stefanthermParams;
				DataArray Udata;
		}; // InitStefanthermFunctor2D

		class InitNonIsothermFunctor2D : HydroBaseFunctor2D
		{
			public:
				InitNonIsothermFunctor2D(HydroParams params_,
						NonIsothermParams nonIsothermParams_,
						DataArray Udata_):
					HydroBaseFunctor2D(params_), nonIsothermParams(nonIsothermParams_),
					Udata(Udata_){};

				static void apply(HydroParams params,
						NonIsothermParams nonIsothermParams,
						DataArray Udata, int nbCells)
				{
					InitNonIsothermFunctor2D functor(params, nonIsothermParams, Udata);
					Kokkos::parallel_for(nbCells, functor);
				}

				KOKKOS_INLINE_FUNCTION
					void operator()(const int index) const
					{
						const real_t ghostWidth = params.ghostWidth;
						const real_t ny = params.ny;
						const real_t dy = params.dy;

						const real_t T = nonIsothermParams.non_isotherm_fluid_temp;

						int i,j;
						index2coord(index, i, j, params.isize, params.jsize);

#ifdef USE_MPI
						const int i_mpi = params.myMpiPos[IY];
#else
						const int j_mpi = 0;
#endif
						const real_t y = params.ymin + (HALF_F + j + ny*j_mpi - ghostWidth)*dy;

						HydroState q;
						q[IH] = fabs(y - 0.00005) - 0.000005;
						const real_t Rstar = q[IH]>ZERO_F? params.settings.Rstar0:params.settings.Rstar1;
						const real_t Bstate = q[IH]>ZERO_F? params.Bstate0:params.Bstate1;
						const real_t barotropic = q[IH]>ZERO_F? params.settings.barotropic0:params.settings.barotropic1;

						q[ID] = q[IH]>ZERO_F?1.20432809308 : 1000.;
						if (!barotropic)
							q[IP] = q[ID] *  Rstar * T - Bstate;
						else
							q[IP] = T;

						q[IU] = ZERO_F;
						q[IV] = ZERO_F;

						setHydroState(Udata, computeConservatives(q), i, j);

					}

				const NonIsothermParams nonIsothermParams;
				DataArray Udata;
		}; // InitNonIsothermFunctor2D

		class InitRayleighBenardFunctor2D : HydroBaseFunctor2D
		{
			public:
				InitRayleighBenardFunctor2D(HydroParams params_,
						RayleighBenardParams rayleighBenardParams_,
						DataArray Udata_):
					HydroBaseFunctor2D(params_), rayleighBenardParams(rayleighBenardParams_),
					Udata(Udata_){};

				static void apply(HydroParams params,
						RayleighBenardParams rayleighBenardParams,
						DataArray Udata, int nbCells)
				{
					InitRayleighBenardFunctor2D functor(params, rayleighBenardParams, Udata);
					Kokkos::parallel_for(nbCells, functor);
				}

				KOKKOS_INLINE_FUNCTION
					void operator()(const int index) const
					{
						const real_t ghostWidth = params.ghostWidth;
						const real_t jmin = params.jmin;
						const real_t ny = params.ny;
						const real_t dy = params.dy;
						// const real_t sigma = 0.1*Ly;
						const real_t y_mid = HALF_F * (params.ymax + params.ymin);

						const real_t T_top   = rayleighBenardParams.rayleigh_benard_temperature_top;
						const real_t T_bot   = rayleighBenardParams.rayleigh_benard_temperature_bottom;
						int i,j;
						index2coord(index, i, j, params.isize, params.jsize);

#ifdef USE_MPI
						const int i_mpi = params.myMpiPos[IX];
						const int j_mpi = params.myMpiPos[IY];
#else
						const int j_mpi = 0;
#endif

						const real_t y = params.ymin + (HALF_F + j + ny*j_mpi - ghostWidth)*dy;

						HydroState q;
						q[IH] = y - y_mid;
						const real_t Rstar = q[IH]>ZERO_F? params.settings.Rstar0:params.settings.Rstar1;
						const real_t T= (T_top - T_bot) / ny * (j-jmin - ghostWidth -1) + T_bot; 
						const real_t Bstate = q[IH]>ZERO_F? params.Bstate0:params.Bstate1;
						q[ID] = ONE_F;
						q[IP] = q[ID] *  Rstar * T - Bstate;
						q[IU] = ZERO_F;
						q[IV] = ZERO_F;

						setHydroState(Udata, computeConservatives(q), i, j);

					}

				const RayleighBenardParams rayleighBenardParams;
				DataArray Udata;
		}; // InitRayleighBenardFunctor2D
		class InitRayleighTaylorInstabilitiesFunctor2D : HydroBaseFunctor2D
		{
			public:
				InitRayleighTaylorInstabilitiesFunctor2D(HydroParams params_, DataArray Udata_) :
					HydroBaseFunctor2D(params_), Udata(Udata_) {};

				static void apply(HydroParams params,
						DataArray Udata, int nbCells)
				{
					InitRayleighTaylorInstabilitiesFunctor2D functor(params, Udata);
					Kokkos::parallel_for(nbCells, functor);
				}

				KOKKOS_INLINE_FUNCTION
					void operator()(const int index) const
					{
						const real_t ghostWidth = params.ghostWidth;
						const real_t xmin = params.xmin;
						const real_t ymin = params.ymin;
						const real_t xmax = params.xmax;
						const real_t ymax = params.ymax;
						const real_t dx = params.dx;
						const real_t dy = params.dy;
						const real_t nx = params.nx;
						const real_t ny = params.ny;

						int i,j;
						index2coord(index, i, j, params.isize, params.jsize);
#ifdef USE_MPI
						const int i_mpi = params.myMpiPos[IX];
						const int j_mpi = params.myMpiPos[IY];
#else
						const int i_mpi = 0;
						const int j_mpi = 0;
#endif
						const real_t x = xmin + (HALF_F + i + nx*i_mpi - ghostWidth)*dx;
						const real_t y = ymin + (HALF_F + j + ny*j_mpi - ghostWidth)*dy;
						const real_t Lx = xmax - xmin;
						const real_t Ly = ymax - ymin;
						const real_t Pi = std::acos(-ONE_F);

						const real_t A = ZERO_F;
						const real_t p0 = params.Astate0;

						const real_t rho = (y+sin(TWO_F*Pi*x)*0.15-ONE_F>ZERO_F) ? 1.8 : ONE_F;

						const real_t v = A * (1.0+std::cos(TWO_F*Pi*x/Lx))*(1.0+std::cos(TWO_F*Pi*y/Ly))/4.0;


						const int np=1000;
						const real_t dx0=Lx/np;

						Udata(i, j, IH)=TWO_F+TWO_F;

						for(int n=0;n<np;n++)
						{
							const real_t x0=xmin+n*dx0;
							const real_t x0p=x0+dx0;
							const real_t fx0 =(x0 -x)-0.3*Pi*cos(TWO_F*Pi*x0 )*(ONE_F-0.15*sin(TWO_F*Pi*x0 )-y);
							const real_t fx0p=(x0p-x)-0.3*Pi*cos(TWO_F*Pi*x0p)*(ONE_F-0.15*sin(TWO_F*Pi*x0p)-y);
							real_t x0d=ZERO_F,y0d=ZERO_F;
							if (fx0*fx0p<=ZERO_F)
							{
								if (fx0==ZERO_F)
								{x0d=x0;}
								if (fx0p==ZERO_F)
								{x0d=x0p;}
								if (fx0*fx0p<ZERO_F)
								{
									x0d=HALF_F*(x0+x0p);
									for (int n0=0; n0<20;n0++)
									{
										const real_t tangente=1+0.6*Pi*Pi*sin(TWO_F*Pi*x0d)*(ONE_F-0.15*sin(TWO_F*Pi*x0d)-y)\
												      +0.09*Pi*Pi*cos(TWO_F*Pi*x0d)*cos(TWO_F*Pi*x0d);
										const real_t result=(x0d-x)-0.3*Pi*cos(TWO_F*Pi*x0d)*(ONE_F-0.15*sin(TWO_F*Pi*x0d)-y);
										x0d= x0d-result/tangente;
									}
								}
								y0d=ONE_F-sin(TWO_F*Pi*x0d)*0.15;

								const real_t d=sqrt((x-x0d)*(x-x0d)+(y-y0d)*(y-y0d));

								if (fabs(d)<fabs(Udata(i, j, IH)))
								{
									if ((y+sin(TWO_F*Pi*x)*0.15-ONE_F)==ZERO_F)
									{Udata(i, j, IH)=ZERO_F;}
									else
									{Udata(i, j, IH)=d*(y+0.15*sin(TWO_F*Pi*x)-ONE_F)/fabs(y+0.15*sin(TWO_F*Pi*x)-ONE_F);

									}
								}

							}
							real_t d0=sqrt((x0-x)*(x0-x)+(1-0.15*sin(TWO_F*Pi*x0)-y)*(1-0.15*sin(TWO_F*Pi*x0)-y));
							if (d0<fabs(Udata(i, j, IH)))
							{
								if ((y+sin(TWO_F*Pi*x)*0.15-ONE_F)!=ZERO_F)
								{Udata(i, j, IH)=d0*(y+0.15*sin(TWO_F*Pi*x)-ONE_F)/fabs(y+0.15*sin(TWO_F*Pi*x)-ONE_F);}
								else
								{Udata(i, j, IH)=ZERO_F;}
							}
						}
						const real_t gamma =(Udata(i, j, IH)>ZERO_F)?params.settings.gamma0:params.settings.gamma1;
						const real_t Bstate=(Udata(i, j, IH)>ZERO_F)?params.Bstate0:params.Bstate1;
						const real_t g_y=params.settings.g_y;

						Udata(i, j, ID) = rho;
						Udata(i, j, IP) = (p0 + rho*params.settings.g_y*y+ gamma*Bstate)/(gamma-ONE_F) +HALF_F*rho*v*v ;
						Udata(i, j, IP) = (Udata(i, j, IH)>ZERO_F)?(p0+rho*g_y*y-(0.8*g_y*(ONE_F-0.15*sin(TWO_F*Pi*x)))+ gamma* Bstate) /(gamma-ONE_F):\
								  (p0+rho*g_y*y                                       + gamma*Bstate) /(gamma-ONE_F);   

						Udata(i, j, IU) = ZERO_F;
						Udata(i, j, IV) = rho * v;
					}

				DataArray Udata;

		}; // InitRayleighTaylorInstabilitiesFunctor2D
		class InitCase34Functor2D : HydroBaseFunctor2D
		{
			public:
				InitCase34Functor2D(HydroParams params_, DataArray Udata_) :
					HydroBaseFunctor2D(params_), Udata(Udata_) {};

				static void apply(HydroParams params,
						DataArray Udata, int nbCells)
				{
					InitCase34Functor2D functor(params, Udata);
					Kokkos::parallel_for(nbCells, functor);
				}

				KOKKOS_INLINE_FUNCTION
					void operator()(const int index) const
					{
						const real_t ghostWidth = params.ghostWidth;
						const real_t ymin = params.ymin;
						const real_t dy = params.dy;
						const real_t ny = params.ny;

						int i,j;
						index2coord(index, i, j, params.isize, params.jsize);
#ifdef USE_MPI
						const int j_mpi = params.myMpiPos[IY];
#else
						const int j_mpi = 0;
#endif
						const real_t y = ymin + (HALF_F + j + ny*j_mpi - ghostWidth)*dy;

						const real_t phi0= y;
						const real_t rho = (phi0>ZERO_F) ? 1. : 1000.;
						const real_t g_y=params.settings.g_y;
						const real_t p0  = params.Astate0+rho*g_y*y  ;
						const real_t Bstate=(phi0>ZERO_F)?params.Bstate0:params.Bstate1;
						const real_t gamma=(phi0>ZERO_F)?params.settings.gamma0:params.settings.gamma1;
						Udata(i, j, ID) = rho;
						Udata(i, j, IP) = (p0+ gamma*Bstate)/(gamma-ONE_F);

						Udata(i, j, IU) = ZERO_F;
						Udata(i, j, IV) = ZERO_F;
						Udata(i, j, IH) = phi0;
					}

				DataArray Udata;
		};
		class InitStaticBubbleFunctor2D : HydroBaseFunctor2D
		{
			public:
				InitStaticBubbleFunctor2D(HydroParams params_, StaticBubbleParams staticbubbleParams_, DataArray Udata_) :
					HydroBaseFunctor2D(params_), staticbubbleParams(staticbubbleParams_), Udata(Udata_) {};

				static void apply(HydroParams params, StaticBubbleParams staticbubbleParams,
						DataArray Udata, int nbCells)
				{
					InitStaticBubbleFunctor2D functor(params, staticbubbleParams, Udata);
					Kokkos::parallel_for(nbCells, functor);
				}

				KOKKOS_INLINE_FUNCTION
					void operator()(const int index) const
					{
						const real_t ghostWidth = params.ghostWidth;
						const real_t sigma=params.settings.sigma;
						const real_t xmin = params.xmin;
						const real_t ymin = params.ymin;
						const real_t dx = params.dx;
						const real_t dy = params.dy;
						const real_t nx = params.nx;
						const real_t ny = params.ny;

						int i,j;
						index2coord(index, i, j, params.isize, params.jsize);
#ifdef USE_MPI
						const int i_mpi = params.myMpiPos[IX];
						const int j_mpi = params.myMpiPos[IY];
#else
						const int i_mpi = 0;
						const int j_mpi = 0;
#endif
						const real_t x = xmin + (HALF_F + i + nx*i_mpi - ghostWidth)*dx;
						const real_t y = ymin + (HALF_F + j + ny*j_mpi - ghostWidth)*dy;

						const real_t phi= sqrt(x*x+y*y)-0.4;
						//const real_t phi= sqrt(x*x+y*y)-2;

						const real_t p0  = params.Astate0;
						const real_t rho = (phi>ZERO_F) ? staticbubbleParams.rho0 : staticbubbleParams.rho1;
						const real_t dp  = (phi>ZERO_F) ? ZERO_F: sigma*2.5;

						const real_t gamma=(phi>ZERO_F)?params.settings.gamma0:params.settings.gamma1;
						const real_t Bstate=(phi>ZERO_F)?params.Bstate0:params.Bstate1;

						Udata(i, j, ID) = rho;
						//Udata(i, j, ID) = cos(x)*sin(y) * 10;
						Udata(i, j, IP) = (p0+ dp+ gamma*Bstate)/(gamma-ONE_F);
						//Udata(i, j, IP) = cos(x)*sin(y) ;

						Udata(i, j, IU) = ZERO_F;
						Udata(i, j, IV) = ZERO_F;
						Udata(i, j, IH) = phi;
					}

				const StaticBubbleParams staticbubbleParams;
				DataArray Udata;
		}; // InitStaticBubbleFunctor2D
		class InitRisingBubbleFunctor2D : HydroBaseFunctor2D
		{
			public:
				InitRisingBubbleFunctor2D(HydroParams params_, DataArray Udata_) :
					HydroBaseFunctor2D(params_), Udata(Udata_) {};

				static void apply(HydroParams params,
						DataArray Udata, int nbCells)
				{
					InitRisingBubbleFunctor2D functor(params, Udata);
					Kokkos::parallel_for(nbCells, functor);
				}

				KOKKOS_INLINE_FUNCTION
					void operator()(const int index) const
					{
						const real_t ghostWidth = params.ghostWidth;
						const real_t sigma=params.settings.sigma;
						const real_t xmin = params.xmin;
						const real_t ymin = params.ymin;
						const real_t ymax = params.ymax;
						const real_t dx = params.dx;
						const real_t dy = params.dy;
						const real_t nx = params.nx;
						const real_t ny = params.ny;

						int i,j;
						index2coord(index, i, j, params.isize, params.jsize);
#ifdef USE_MPI
						const int i_mpi = params.myMpiPos[IX];
						const int j_mpi = params.myMpiPos[IY];
#else
						const int i_mpi = 0;
						const int j_mpi = 0;
#endif
						const real_t x = xmin + (HALF_F + i + nx*i_mpi - ghostWidth)*dx;
						const real_t y = ymin + (HALF_F + j + ny*j_mpi - ghostWidth)*dy;

						const real_t phi=sqrt(x*x+y*y)-0.025;

						const real_t p0  = params.Astate0;
						const real_t rho = (phi>ZERO_F) ? 1000. : 1.;

						const real_t dp  = (phi>ZERO_F) ? ZERO_F: sigma*40.;

						real_t v;

						v=0.;

						const real_t gamma=(phi>ZERO_F)?params.settings.gamma0:params.settings.gamma1;
						const real_t Bstate=(phi>ZERO_F)?params.Bstate0:params.Bstate1;

						Udata(i, j, ID) = rho;
						Udata(i, j, IP) = (p0 + dp + gamma* Bstate +1000*(y-ymax)*params.settings.g_y)/(gamma-ONE_F)+HALF_F*rho*v*v;

						Udata(i, j, IU) = ZERO_F;
						Udata(i, j, IV) = rho*v;
						Udata(i, j, IH) = phi;
					}

				DataArray Udata;

		}; // InitRisingBubbleFunctor2D

		class InitDropOsicillationFunctor2D : HydroBaseFunctor2D
		{
			public:
				InitDropOsicillationFunctor2D(HydroParams params_, DataArray Udata_) :
					HydroBaseFunctor2D(params_), Udata(Udata_) {};

				static void apply(HydroParams params,
						DataArray Udata, int nbCells)
				{
					InitDropOsicillationFunctor2D functor(params, Udata);
					Kokkos::parallel_for(nbCells, functor);
				}

				KOKKOS_INLINE_FUNCTION
					void operator()(const int index) const
					{
						const real_t ghostWidth = params.ghostWidth;
						const real_t sigma=params.settings.sigma;
						const real_t xmin = params.xmin;
						const real_t ymin = params.ymin;
						const real_t dx = params.dx;
						const real_t dy = params.dy;
						const real_t nx = params.nx;
						const real_t ny = params.ny;

						int i,j;
						index2coord(index, i, j, params.isize, params.jsize);
#ifdef USE_MPI
						const int i_mpi = params.myMpiPos[IX];
						const int j_mpi = params.myMpiPos[IY];
#else
						const int i_mpi = 0;
						const int j_mpi = 0;
#endif
						const real_t x = xmin + (HALF_F + i + nx*i_mpi - ghostWidth)*dx;
						const real_t y = ymin + (HALF_F + j + ny*j_mpi - ghostWidth)*dy;

						const real_t phi=sqrt(x*x+y*y)-0.2;

						const real_t V0=1.;
						const real_t r0=0.05;
						const real_t r=sqrt(x*x+y*y);
						const real_t u=V0*x/r0*(1-y*y/r0*r)*exp(-r/r0);
						const real_t v=-V0*y/r0*(1-x*x/r0*r)*exp(-r/r0);

						const real_t p0  = params.Astate0;
						const real_t rho = (phi>ZERO_F) ? 0.001 : ONE_F;
						const real_t dp  = (phi>ZERO_F) ? ZERO_F: sigma/0.2;

						const real_t gamma=(phi>ZERO_F)?params.settings.gamma0:params.settings.gamma1;
						const real_t Bstate=(phi>ZERO_F)?params.Bstate0:params.Bstate1;

						Udata(i, j, ID) = rho;
						Udata(i, j, IP) = (p0+ dp+ gamma*Bstate)/(gamma-ONE_F)+HALF_F*rho*(u*u+v*v);

						Udata(i, j, IU) = rho*u;
						Udata(i, j, IV) = rho*v;
						Udata(i, j, IH) = phi;
					}

				DataArray Udata;

		}; // InitStaticBubbleFunctor2D


		class InitRayleighTaylorFunctor2D : HydroBaseFunctor2D
		{
			public:
				InitRayleighTaylorFunctor2D(HydroParams params_, DataArray Udata_) :
					HydroBaseFunctor2D(params_), Udata(Udata_) {};

				static void apply(HydroParams params,
						DataArray Udata, int nbCells)
				{
					InitRayleighTaylorFunctor2D functor(params, Udata);
					Kokkos::parallel_for(nbCells, functor);
				}

				KOKKOS_INLINE_FUNCTION
					void operator()(const int index) const
					{
						const real_t ghostWidth = params.ghostWidth;
						const real_t xmin = params.xmin;
						const real_t ymin = params.ymin;
						const real_t xmax = params.xmax;
						const real_t ymax = params.ymax;
						const real_t dx = params.dx;
						const real_t dy = params.dy;
						const real_t nx = params.nx;
						const real_t ny = params.ny;

						int i,j;
						index2coord(index, i, j, params.isize, params.jsize);
#ifdef USE_MPI
						const int i_mpi = params.myMpiPos[IX];
						const int j_mpi = params.myMpiPos[IY];
#else
						const int i_mpi = 0;
						const int j_mpi = 0;
#endif
						const real_t x = xmin + (HALF_F + i + nx*i_mpi - ghostWidth)*dx;
						const real_t y = ymin + (HALF_F + j + ny*j_mpi - ghostWidth)*dy;
						const real_t Lx = xmax - xmin;
						const real_t Ly = ymax - ymin;
						const real_t Pi = std::acos(-ONE_F);

						const real_t A = 0.01;
						const real_t p0 = params.Astate0;
						const real_t rho = (y<=ZERO_F) ? ONE_F : TWO_F;
						const real_t v = A * (1.0+std::cos(TWO_F*Pi*x/Lx))*(1.0+std::cos(TWO_F*Pi*y/Ly))/4.0;

						const real_t gamma=(y>ZERO_F)?params.settings.gamma0:params.settings.gamma1;
						const real_t Bstate=(y>ZERO_F)?params.Bstate0:params.Bstate1;

						Udata(i, j, ID) = rho;
						Udata(i, j, IP) = (p0 + rho*params.settings.g_y*y+ gamma*Bstate)/(gamma-ONE_F) +HALF_F*rho*v*v ;

						Udata(i, j, IU) = ZERO_F;
						Udata(i, j, IV) = rho * v;
						Udata(i, j, IH) = y;
					}

				DataArray Udata;

		}; // InitRayleighTaylorFunctor2D


		class InitAtmosphereAtRestFunctor2D : HydroBaseFunctor2D
		{
			public:
				InitAtmosphereAtRestFunctor2D(HydroParams params_, DataArray Udata_) :
					HydroBaseFunctor2D(params_),
					params(params_), Udata(Udata_) {};

				static void apply(HydroParams params,
						DataArray Udata, int nbCells)
				{
					InitAtmosphereAtRestFunctor2D functor(params, Udata);
					Kokkos::parallel_for(nbCells, functor);
				}

				KOKKOS_INLINE_FUNCTION
					void operator()(const int i) const
					{
						const int jsize = params.jsize;
						const int jmin  = params.jmin;
						const int ghostWidth = params.ghostWidth;
						const real_t dy = params.dy;
						const real_t gamma0 = params.settings.gamma0;
						const real_t ny = params.ny;

#ifdef USE_MPI
						const int j_mpi = params.myMpiPos[IY];
#else
						const int j_mpi = 0;
#endif

						// To handle mpi dispatch
						real_t rho = ONE_F;
						for (int j_glob=1; j_glob<=ny*j_mpi; ++j_glob)
						{
							rho *= (TWO_F - dy) / (TWO_F + dy);
						}

						if(i >= params.imin && i <= params.imax)
						{
							for (int j=jmin+ghostWidth; j<jsize-ghostWidth; ++j)
							{
								rho *= (TWO_F - dy) / (TWO_F + dy);
								Udata(i, j, ID) = rho;
								Udata(i, j, IP) = rho / (gamma0-ONE_F);
								Udata(i, j, IU) = ZERO_F;
								Udata(i, j, IV) = ZERO_F;
							}
						}
					};
				const HydroParams params;
				DataArray Udata;
		}; // InitAtmosphereAtRestFunctor2D


		class InitGreshoFunctor2D : HydroBaseFunctor2D
		{
			public:
				InitGreshoFunctor2D(HydroParams params_, GreshoParams greshoParams_, DataArray Udata_):
					HydroBaseFunctor2D(params_), Udata(Udata_), greshoParams(greshoParams_) {};

				static void apply(HydroParams params, GreshoParams greshoParams,
						DataArray Udata, int nbCells)
				{
					InitGreshoFunctor2D functor(params, greshoParams, Udata);
					Kokkos::parallel_for(nbCells, functor);
				}

				KOKKOS_INLINE_FUNCTION
					void operator()(const int& index) const
					{
						const real_t ghostWidth = params.ghostWidth;
						const real_t xmin = params.xmin;
						const real_t ymin = params.ymin;
						const real_t dx = params.dx;
						const real_t dy = params.dy;
						const real_t nx = params.nx;
						const real_t ny = params.ny;

						int i,j;
						index2coord(index, i, j, params.isize, params.jsize);


#ifdef USE_MPI
						const int i_mpi = params.myMpiPos[IX];
						const int j_mpi = params.myMpiPos[IY];
#else
						const int i_mpi = 0;
						const int j_mpi = 0;
#endif

						real_t y = ymin + (j+ny*j_mpi-ghostWidth)*dy + dy/2.0;
						y -= greshoParams.gresho_center_y;

						real_t x = xmin + (i+nx*i_mpi-ghostWidth)*dx + dx/2.0;
						x -= greshoParams.gresho_center_x;

						const real_t gamma0 = params.settings.gamma0;
						const real_t Mach = greshoParams.gresho_mach;

						const real_t r = std::sqrt(x*x + y*y);
						const real_t theta = std::atan2(y, x);

						const real_t rho0 = greshoParams.rho0;
						const real_t rho1 = greshoParams.rho1;
						const real_t rho = r > 0.2 ? rho0 : rho1;

						const real_t p0 = rho0 / (gamma0 * Mach * Mach) - params.Bstate0;

						real_t vtheta;
						real_t p;
						if (r <= 0.2)
						{
							vtheta = 5.0 * r;
							p  = p0 + 12.5 * r * r * rho1;
						}
						else if (r <= 0.4)
						{
							vtheta = 2.0 - 5.0 * r;
							p  = p0 + 0.5 * rho1 + 12.5 * (r * r - 0.04)* rho0 + 4.0 * (1.0 - 5.0 * r + std::log(5.0*r)) * rho0;
						}
						else
						{
							vtheta = 0.0;
							p  = p0 + 0.5 * rho1 + 1.5* rho0 + 4.0 * (-1.0 + std::log(2.0)) * rho0;
						}

						const real_t vx = - vtheta * std::sin(theta);
						const real_t vy =   vtheta * std::cos(theta);

                                                const real_t Bstate = r-0.2>ZERO_F? params.Bstate0        :params.Bstate1;
						const real_t gamma1 = r-0.2>ZERO_F? params.settings.gamma0:params.settings.gamma1;

						Udata(i, j, ID) = rho;
						//Udata(i, j, ID) = cos(x) * sin(y);
						//Udata(i, j, IP) = cos(x) * sin(y);
						//Udata(i, j, IH) = sqrt(x*x + y* y) - TWO_F;
						Udata(i, j, IU) = rho * vx;
						Udata(i, j, IV) = rho * vy;
						Udata(i, j, IP) = (p + gamma1 * Bstate)/(gamma1-1.0) + HALF_F*rho*(vx*vx + vy*vy);
						Udata(i, j, IH) = r-0.2;

					};

				DataArray Udata;
				GreshoParams greshoParams;
		}; // InitGreshoFunctor2D


		class InitImplodeFunctor2D : public HydroBaseFunctor2D
		{
			public:
				InitImplodeFunctor2D(HydroParams params_, DataArray Udata_) :
					HydroBaseFunctor2D(params_), Udata(Udata_)  {};

				static void apply(HydroParams params,
						DataArray Udata, int nbCells)
				{
					InitImplodeFunctor2D functor(params, Udata);
					Kokkos::parallel_for(nbCells, functor);
				}

				KOKKOS_INLINE_FUNCTION
					void operator()(const int& index) const
					{
						const int isize = params.isize;
						const int jsize = params.jsize;
						const int ghostWidth = params.ghostWidth;

#ifdef USE_MPI
						const int i_mpi = params.myMpiPos[IX];
						const int j_mpi = params.myMpiPos[IY];
#else
						const int i_mpi = 0;
						const int j_mpi = 0;
#endif

						const int nx = params.nx;
						const int ny = params.ny;

						const real_t xmin = params.xmin;
						const real_t ymin = params.ymin;
						const real_t dx = params.dx;
						const real_t dy = params.dy;

						const real_t gamma0 = params.settings.gamma0;

						int i,j;
						index2coord(index,i,j,isize,jsize);

						real_t x = xmin + dx/2 + (i+nx*i_mpi-ghostWidth)*dx;
						real_t y = ymin + dy/2 + (j+ny*j_mpi-ghostWidth)*dy;

						real_t tmp = x+y*y;
						if (tmp > 0.5 && tmp < 1.5)
						{
							Udata(i  ,j  , ID) = 1.0;
							Udata(i  ,j  , IP) = 1.0/(gamma0-1.0);
							Udata(i  ,j  , IU) = 0.0;
							Udata(i  ,j  , IV) = 0.0;
						}
						else
						{
							Udata(i  ,j  , ID) = 0.125;
							Udata(i  ,j  , IP) = 0.14/(gamma0-1.0);
							Udata(i  ,j  , IU) = 0.0;
							Udata(i  ,j  , IV) = 0.0;
						}

					} // end operator ()

				DataArray Udata;
		}; // InitImplodeFunctor2D


		class InitBlastFunctor2D : public HydroBaseFunctor2D
		{
			public:
				InitBlastFunctor2D(HydroParams params_, BlastParams bParams_, DataArray Udata_) :
					HydroBaseFunctor2D(params_), bParams(bParams_), Udata(Udata_)  {};

				static void apply(HydroParams params, BlastParams blastParams,
						DataArray Udata, int nbCells)
				{
					InitBlastFunctor2D functor(params, blastParams, Udata);
					Kokkos::parallel_for(nbCells, functor);
				}

				KOKKOS_INLINE_FUNCTION
					void operator()(const int& index) const
					{
						const int isize = params.isize;
						const int jsize = params.jsize;
						const int ghostWidth = params.ghostWidth;

#ifdef USE_MPI
						const int i_mpi = params.myMpiPos[IX];
						const int j_mpi = params.myMpiPos[IY];
#else
						const int i_mpi = 0;
						const int j_mpi = 0;
#endif

						const int nx = params.nx;
						const int ny = params.ny;

						const real_t xmin = params.xmin;
						const real_t ymin = params.ymin;
						const real_t dx = params.dx;
						const real_t dy = params.dy;

						const real_t gamma0 = params.settings.gamma0;

						// blast problem parameters
						const real_t blast_radius      = bParams.blast_radius;
						const real_t radius2           = blast_radius*blast_radius;
						const real_t blast_center_x    = bParams.blast_center_x;
						const real_t blast_center_y    = bParams.blast_center_y;
						const real_t blast_density_in  = bParams.blast_density_in;
						const real_t blast_density_out = bParams.blast_density_out;
						const real_t blast_pressure_in = bParams.blast_pressure_in;
						const real_t blast_pressure_out= bParams.blast_pressure_out;

						int i,j;
						index2coord(index,i,j,isize,jsize);

						real_t x = xmin + dx/2 + (i+nx*i_mpi-ghostWidth)*dx;
						real_t y = ymin + dy/2 + (j+ny*j_mpi-ghostWidth)*dy;

						real_t d2 =
							(x-blast_center_x)*(x-blast_center_x)+
							(y-blast_center_y)*(y-blast_center_y);

						if (d2 < radius2)
						{
							Udata(i  ,j  , ID) = blast_density_in;
							Udata(i  ,j  , IP) = blast_pressure_in/(gamma0-1.0);
							Udata(i  ,j  , IU) = 0.0;
							Udata(i  ,j  , IV) = 0.0;
						}
						else
						{
							Udata(i  ,j  , ID) = blast_density_out;
							Udata(i  ,j  , IP) = blast_pressure_out/(gamma0-1.0);
							Udata(i  ,j  , IU) = 0.0;
							Udata(i  ,j  , IV) = 0.0;
						}

					} // end operator ()

				BlastParams bParams;
				DataArray Udata;
		}; // InitBlastFunctor2D


		class InitFourQuadrantFunctor2D : public HydroBaseFunctor2D
		{
			public:
				InitFourQuadrantFunctor2D(HydroParams params_,
						DataArray Udata_,
						int configNumber_,
						HydroState U0_,
						HydroState U1_,
						HydroState U2_,
						HydroState U3_,
						real_t xt_,
						real_t yt_) :
					HydroBaseFunctor2D(params_), Udata(Udata_),
					U0(U0_), U1(U1_), U2(U2_), U3(U3_), xt(xt_), yt(yt_) {};

				static void apply(HydroParams params,
						DataArray Udata,
						int configNumber,
						HydroState U0,
						HydroState U1,
						HydroState U2,
						HydroState U3,
						real_t xt,
						real_t yt,
						int nbCells)
				{
					InitFourQuadrantFunctor2D functor(params, Udata, configNumber,
							U0, U1, U2, U3, xt, yt);
					Kokkos::parallel_for(nbCells, functor);
				}

				KOKKOS_INLINE_FUNCTION
					void operator()(const int& index) const
					{
						const int isize = params.isize;
						const int jsize = params.jsize;
						const int ghostWidth = params.ghostWidth;

#ifdef USE_MPI
						const int i_mpi = params.myMpiPos[IX];
						const int j_mpi = params.myMpiPos[IY];
#else
						const int i_mpi = 0;
						const int j_mpi = 0;
#endif

						const int nx = params.nx;
						const int ny = params.ny;

						const real_t xmin = params.xmin;
						const real_t ymin = params.ymin;
						const real_t dx = params.dx;
						const real_t dy = params.dy;

						int i,j;
						index2coord(index,i,j,isize,jsize);

						real_t x = xmin + dx/2 + (i+nx*i_mpi-ghostWidth)*dx;
						real_t y = ymin + dy/2 + (j+ny*j_mpi-ghostWidth)*dy;

						if (x<xt)
						{
							if (y<yt)
							{
								// quarter 2
								Udata(i  ,j  , ID) = U2[ID];
								Udata(i  ,j  , IP) = U2[IP];
								Udata(i  ,j  , IU) = U2[IU];
								Udata(i  ,j  , IV) = U2[IV];
							}
							else
							{
								// quarter 1
								Udata(i  ,j  , ID) = U1[ID];
								Udata(i  ,j  , IP) = U1[IP];
								Udata(i  ,j  , IU) = U1[IU];
								Udata(i  ,j  , IV) = U1[IV];
							}
						}
						else
						{
							if (y<yt)
							{
								// quarter 3
								Udata(i  ,j  , ID) = U3[ID];
								Udata(i  ,j  , IP) = U3[IP];
								Udata(i  ,j  , IU) = U3[IU];
								Udata(i  ,j  , IV) = U3[IV];
							}
							else
							{
								// quarter 0
								Udata(i  ,j  , ID) = U0[ID];
								Udata(i  ,j  , IP) = U0[IP];
								Udata(i  ,j  , IU) = U0[IU];
								Udata(i  ,j  , IV) = U0[IV];
							}
						}

					} // end operator ()

				DataArray Udata;
				HydroState2d U0, U1, U2, U3;
				real_t xt, yt;
		}; // InitFourQuadrantFunctor2D


		class InitIsentropicVortexFunctor2D : public HydroBaseFunctor2D
		{
			public:
				InitIsentropicVortexFunctor2D(HydroParams params_,
						IsentropicVortexParams iparams_,
						DataArray Udata_) :
					HydroBaseFunctor2D(params_), iparams(iparams_), Udata(Udata_)  {};

				static void apply(HydroParams params, IsentropicVortexParams isentropicVortexParams,
						DataArray Udata, int nbCells)
				{
					InitIsentropicVortexFunctor2D functor(params, isentropicVortexParams, Udata);
					Kokkos::parallel_for(nbCells, functor);
				}

				KOKKOS_INLINE_FUNCTION
					void operator()(const int& index) const
					{
						const int isize = params.isize;
						const int jsize = params.jsize;
						const int ghostWidth = params.ghostWidth;

#ifdef USE_MPI
						const int i_mpi = params.myMpiPos[IX];
						const int j_mpi = params.myMpiPos[IY];
#else
						const int i_mpi = 0;
						const int j_mpi = 0;
#endif

						const int nx = params.nx;
						const int ny = params.ny;

						const real_t xmin = params.xmin;
						const real_t ymin = params.ymin;
						const real_t dx = params.dx;
						const real_t dy = params.dy;

						const real_t gamma0 = params.settings.gamma0;

						int i,j;
						index2coord(index,i,j,isize,jsize);

						real_t x = xmin + dx/2 + (i+nx*i_mpi-ghostWidth)*dx;
						real_t y = ymin + dy/2 + (j+ny*j_mpi-ghostWidth)*dy;

						// ambient flow
						const real_t rho_a = this->iparams.rho_a;
						//const real_t p_a   = this->iparams.p_a;
						const real_t T_a   = this->iparams.T_a;
						const real_t u_a   = this->iparams.u_a;
						const real_t v_a   = this->iparams.v_a;
						//const real_t w_a   = this->iparams.w_a;

						// vortex center
						const real_t vortex_x = this->iparams.vortex_x;
						const real_t vortex_y = this->iparams.vortex_y;

						// relative coordinates versus vortex center
						real_t xp = x - vortex_x;
						real_t yp = y - vortex_y;
						real_t r  = std::sqrt(xp*xp + yp*yp);

						const real_t beta = this->iparams.beta;

						real_t du = - yp * beta / (2 * M_PI) * std::exp(0.5*(1.0-r*r));
						real_t dv =   xp * beta / (2 * M_PI) * std::exp(0.5*(1.0-r*r));

						real_t T = T_a - (gamma0-1)*beta*beta/(8*gamma0*M_PI*M_PI)*std::exp(1.0-r*r);
						real_t rho = rho_a*std::pow(T/T_a,1.0/(gamma0-1));

						Udata(i  ,j  , ID) = rho;
						Udata(i  ,j  , IU) = rho*(u_a + du);
						Udata(i  ,j  , IV) = rho*(v_a + dv);
						//Udata(i  ,j  , IP) = std::pow(rho,gamma0)/(gamma0-1.0) +
						Udata(i  ,j  , IP) = rho*T/(gamma0-1.0) +
							0.5*rho*(u_a + du)*(u_a + du) +
							0.5*rho*(v_a + dv)*(v_a + dv) ;

					} // end operator ()

				IsentropicVortexParams iparams;
				DataArray Udata;
		}; // InitIsentropicVortexFunctor2D


		class InitRiemannProblemFunctor2D : public HydroBaseFunctor2D
		{
			public:
				InitRiemannProblemFunctor2D(HydroParams params_,
						RiemannProblemParams rp_params_,
						DataArray Udata_) :
					HydroBaseFunctor2D(params_), rp_params(rp_params_),
					Udata(Udata_)  {};

				static void apply(HydroParams params,
						RiemannProblemParams rp_params,
						DataArray Udata, int nbCells)
				{
					InitRiemannProblemFunctor2D functor(params, rp_params, Udata);
					Kokkos::parallel_for(nbCells, functor);
				}

				KOKKOS_INLINE_FUNCTION
					void operator()(const int index) const
					{
						const int isize = params.isize;
						const int jsize = params.jsize;
						const int ghostWidth = params.ghostWidth;

#ifdef USE_MPI
						const int i_mpi = params.myMpiPos[IX];
#else
						const int i_mpi = 0;
#endif

						const int nx = params.nx;

						const real_t xmin = params.xmin;
						const real_t xmax = params.xmax;
						const real_t dx = params.dx;

						int i,j;
						index2coord(index,i,j,isize,jsize);

						real_t x = xmin + dx/2 + (i+nx*i_mpi-ghostWidth)*dx;
						real_t x_middle = HALF_F * (xmin + xmax);

						HydroState q;
						if (x < x_middle)
						{
							q[ID] = rp_params.density_left;
							q[IP] = rp_params.pressure_left;
							q[IS] = ONE_F;
							q[IU] = rp_params.velocity_left;
							q[IV] = ZERO_F;
						}
						else
						{
							q[ID] = rp_params.density_right;
							q[IP] = rp_params.pressure_right;
							q[IS] = ONE_F;
							q[IU] = rp_params.velocity_right;
							q[IV] = ZERO_F;
						}

						setHydroState(Udata, computeConservatives(q), i, j);

					} // end operator ()

				RiemannProblemParams rp_params;
				DataArray Udata;
		}; // InitRiemannProblemFunctor2D

	} // namespace all_regime

} // namespace euler_kokkos
