
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
		class ComputeInterfaceRPositionFunctor2D : public HydroBaseFunctor2D
		{
			public:
				ComputeInterfaceRPositionFunctor2D(HydroParams params_, DataArrayConst Qdata_) :
					HydroBaseFunctor2D(params_), Qdata(Qdata_)  {};

				static void apply(HydroParams params, DataArrayConst Qdata, real_t& position,  int nbCells)
				{
					ComputeInterfaceRPositionFunctor2D functor(params, Qdata);
					Kokkos::parallel_reduce(nbCells, functor, position);
				}


				// Tell each thread how to initialize its reduction result.
				KOKKOS_INLINE_FUNCTION
					void init (real_t& dst) const
					{
						dst = ZERO_F;
					} // init

				/* this is a reduce (max) functor */
				KOKKOS_INLINE_FUNCTION
					void operator()(const int& index, real_t& position) const
					{
						const int isize = params.isize;
						const int jsize = params.jsize;
						const int ghostWidth = params.ghostWidth;

						int i,j;
						index2coord(index,i,j,isize,jsize);
						if (params.settings.problem_name=="Case34")
						{
							if(j >= ghostWidth && j <= jsize - ghostWidth &&
									i >= ghostWidth && i <= isize - ghostWidth)
							{
								const real_t phiLoc= Qdata(i, j, IH);
								const real_t phiPy= Qdata(i, j+1, IH);
								real_t distance=ZERO_F;
								if ((phiLoc*phiPy<=ZERO_F&&(phiLoc>ZERO_F||phiPy>ZERO_F))&&i==params.isize-params.ghostWidth)
								{
									const real_t y = params.ymin + (HALF_F + j  - params.ghostWidth)*params.dy;
									distance=y+(fabs(phiLoc)/fabs(phiLoc-phiPy)) * params.dy;
									position=(fabs(distance)>fabs(position))? distance : position;
								}
							}

						}
						else
						{
							if(j >= ghostWidth && j <= jsize - ghostWidth &&
									i >= ghostWidth && i <= isize - ghostWidth)
							{
								const real_t phiLoc= Qdata(i, j, IH);
								const real_t phiPy= Qdata(i, j+1, IH);
								real_t distance=ZERO_F;
								if ((phiLoc*phiPy<=ZERO_F&&(phiLoc>ZERO_F||phiPy>ZERO_F))&&phiLoc<=ZERO_F&&i==params.ghostWidth+1)
								{
									const real_t y = params.ymin + (HALF_F + j  - params.ghostWidth)*params.dy;
									distance=y+(fabs(phiLoc)/fabs(phiLoc-phiPy)) * params.dy;
									position=(fabs(distance)>fabs(position))? distance : position;
								}
							}
						}
					} // operator ()

				// "Join" intermediate results from different threads.
				// This should normally implement the same reduction
				// operation as operator() above. Note that both input
				// arguments MUST be declared volatile.
				KOKKOS_INLINE_FUNCTION
					void join (volatile real_t& dst,
							const volatile real_t& src) const
					{
						// max reduce
						if (fabs(dst) < fabs(src))
							dst = src;
					} // join

				const DataArrayConst Qdata;
		}; // ComputeInterfacePositionFunctor2D
		class ComputeInterfacePositionFunctor2D : public HydroBaseFunctor2D
		{
			public:
				ComputeInterfacePositionFunctor2D(HydroParams params_, DataArrayConst Qdata_) :
					HydroBaseFunctor2D(params_), Qdata(Qdata_)  {};

				static void apply(HydroParams params, DataArrayConst Qdata, real_t& position, int nbCells)
				{
					ComputeInterfacePositionFunctor2D functor(params, Qdata);
					Kokkos::parallel_reduce(nbCells, functor, position);
				}


				// Tell each thread how to initialize its reduction result.
				KOKKOS_INLINE_FUNCTION
					void init (real_t& dst) const
					{
						dst = ZERO_F;
					} // init

				/* this is a reduce (max) functor */
				KOKKOS_INLINE_FUNCTION
					void operator()(const int& index, real_t& position) const
					{
						const int isize = params.isize;
						const int jsize = params.jsize;
						const int ghostWidth = params.ghostWidth;

						int i,j;
						index2coord(index,i,j,isize,jsize);
						if (params.settings.problem_name=="Case34")
						{
							if(j >= ghostWidth && j <= jsize - ghostWidth &&
									i >= ghostWidth && i <= isize - ghostWidth)
							{
								const real_t phiLoc= Qdata(i, j, IH);
								const real_t phiPy= Qdata(i, j+1, IH);
								real_t distance=ZERO_F;
								if ((phiLoc*phiPy<=ZERO_F&&(phiLoc>ZERO_F||phiPy>ZERO_F))&&i==params.ghostWidth)
								{
									const real_t y = params.ymin + (HALF_F + j  - params.ghostWidth)*params.dy;
									distance=y+(fabs(phiLoc)/fabs(phiLoc-phiPy)) * params.dy;
									position=(fabs(distance)>fabs(position))? distance : position;
								}
							}
						}
						else
						{

							if(j >= ghostWidth && j <= jsize - ghostWidth &&
									i >= ghostWidth && i <= isize - ghostWidth)
							{
								const real_t phiLoc= Qdata(i, j, IH);
								const real_t phiPy= Qdata(i, j+1, IH);
								real_t distance=ZERO_F;
								if ((phiLoc*phiPy<=ZERO_F&&(phiLoc>ZERO_F||phiPy>ZERO_F))&&i==params.ghostWidth+1&&phiLoc>ZERO_F)
								{
									const real_t y = params.ymin + (HALF_F + j  - params.ghostWidth)*params.dy;
									distance=y+(fabs(phiLoc)/fabs(phiLoc-phiPy)) * params.dy;
									position=(fabs(distance)>fabs(position))? distance : position;
								}
							}
						}
					} // operator ()

				// "Join" intermediate results from different threads.
				// This should normally implement the same reduction
				// operation as operator() above. Note that both input
				// arguments MUST be declared volatile.
				KOKKOS_INLINE_FUNCTION
					void join (volatile real_t& dst,
							const volatile real_t& src) const
					{
						// max reduce
						if (fabs(dst) < fabs(src))
							dst = src;
					} // join

				const DataArrayConst Qdata;
		}; // ComputeInterfacePositionFunctor2D
		class ComputePressureCFunctor2D : public HydroBaseFunctor2D
		{
			public:
				ComputePressureCFunctor2D(HydroParams params_, DataArrayConst Qdata_) :
					HydroBaseFunctor2D(params_), Qdata(Qdata_)  {};

				static void apply(HydroParams params, DataArrayConst Qdata, real_t& p, int nbCells)
				{
					ComputePressureCFunctor2D functor(params, Qdata);
					Kokkos::parallel_reduce(nbCells, functor, p);
				}
				KOKKOS_INLINE_FUNCTION
					void init (real_t& dst) const
					{
						dst = ZERO_F;
					} // init

				KOKKOS_INLINE_FUNCTION
					void operator()(const int& index, real_t& p) const
					{
						const int isize = params.isize;
						const int jsize = params.jsize;
						const int ny = params.ny;
						const int nx = params.nx;
						const int ghostWidth = params.ghostWidth;

						int i,j;
						index2coord(index,i,j,isize,jsize);
						if(j == ghostWidth + ny/2 -1  &&
								i == ghostWidth+ nx/2 -1) 
						{
							p=Qdata(i, j , IP);

						}
					} // operator ()
				KOKKOS_INLINE_FUNCTION
					void join (volatile real_t& dst,
							const volatile real_t& src) const
					{
						// max reduce
						if (fabs(dst) < fabs(src))
							dst = src;
					} // join

				const DataArrayConst Qdata;
		}; // ComputePressureFunctor2D
		class ComputePressureRFunctor2D : public HydroBaseFunctor2D
		{
			public:
				ComputePressureRFunctor2D(HydroParams params_, DataArrayConst Qdata_) :
					HydroBaseFunctor2D(params_), Qdata(Qdata_)  {};

				static void apply(HydroParams params, DataArrayConst Qdata, real_t& p, int nbCells)
				{
					ComputePressureRFunctor2D functor(params, Qdata);
					Kokkos::parallel_reduce(nbCells, functor, p);
				}
				KOKKOS_INLINE_FUNCTION
					void init (real_t& dst) const
					{
						dst = ZERO_F;
					} // init

				KOKKOS_INLINE_FUNCTION
					void operator()(const int& index, real_t& p) const
					{
						const int isize = params.isize;
						const int jsize = params.jsize;
						const int ny = params.ny;
						const int ghostWidth = params.ghostWidth;

						int i,j;
						index2coord(index,i,j,isize,jsize);
						if(j == ghostWidth + ny -1  &&
								i == ghostWidth) 
						{
							p=Qdata(i, j , IP);

						}
					} // operator ()
				KOKKOS_INLINE_FUNCTION
					void join (volatile real_t& dst,
							const volatile real_t& src) const
					{
						// max reduce
						if (fabs(dst) < fabs(src))
							dst = src;
					} // join

				const DataArrayConst Qdata;
		}; // ComputePressureFunctor2D
		class ComputeVTProfileFunctor2D : HydroBaseFunctor2D
		{
			public:
				ComputeVTProfileFunctor2D(HydroParams params_, DataArrayConst Qdata_,const int n_) :
					HydroBaseFunctor2D(params_), Qdata(Qdata_), n(n_)  {};

				static void apply(HydroParams params, DataArrayConst Qdata,const int n, int nbCells)
				{
					ComputeVTProfileFunctor2D functor(params, Qdata, n);
					Kokkos::parallel_for(nbCells, functor);
				}
				KOKKOS_INLINE_FUNCTION
					void operator()(int index) const
					{
						const int isize = params.isize;
						const int jsize = params.jsize;
						const int nx = params.nx;
						const int ny = params.ny;

						const int ghostWidth = params.ghostWidth;

						int i,j;
						index2coord(index,i,j,isize,jsize);
						if (params.settings.problem_name=="riemann_problem")
						{
							if(j == ghostWidth&&i == ghostWidth) 
							{
								std::string outputDir    = "./";
								std::string outputPrefix = "rho_p_profile";

								std::ostringstream stepNum;
								stepNum.width(7);
								stepNum.fill('0');
								stepNum << n;

								std::string filename     = outputDir + "/" + outputPrefix+"_"+stepNum.str() + ".dat";

								std::fstream outFile;
								outFile.open(filename.c_str(), std::ios_base::out);
								for (int i0=i; i0<ghostWidth+nx; i0++)
								{
									const real_t x= (i0 - ghostWidth + HALF_F) * params.dx;

									outFile <<x<< " " <<Qdata(i0, j, ID)<< " "<<Qdata(i0, j, IP)<<"\n"  ;
								}
								outFile.close();

							}
						}
						else
						{
							if(j == ghostWidth&&i == ghostWidth) 
							{
								std::string outputDir    = "./";
								std::string outputPrefix = "vt_profile";

								std::ostringstream stepNum;
								stepNum.width(7);
								stepNum.fill('0');
								stepNum << n;

								std::string filename     = outputDir + "/" + outputPrefix+"_"+stepNum.str() + ".dat";

								std::fstream outFile;
								outFile.open(filename.c_str(), std::ios_base::out);
								for (int j0=j; j0<ghostWidth+ny; j0++)
								{
									const real_t y= (j0 - ghostWidth + HALF_F) * params.dy;
									const real_t Bstate=Qdata(i, j0, IH)>ZERO_F? params.Bstate0:params.Bstate1;
									const real_t Rstar=Qdata(i, j0, IH)>ZERO_F? params.settings.Rstar0:params.settings.Rstar1;

									outFile <<y<< " " <<Qdata(i, j0, IV)<< " "<<(Qdata(i, j0, IP)+Bstate)/Qdata(i, j0, ID)/Rstar<<"\n"  ;
								}
								outFile.close();

							}
						}
					} // operator ()
				const DataArrayConst Qdata;
				const int n;
		}; // ComputeVTFunctor2D
		class ComputePressureFunctor2D : public HydroBaseFunctor2D
		{
			public:
				ComputePressureFunctor2D(HydroParams params_, DataArrayConst Qdata_) :
					HydroBaseFunctor2D(params_), Qdata(Qdata_)  {};

				static void apply(HydroParams params, DataArrayConst Qdata, real_t& p, int nbCells)
				{
					ComputePressureFunctor2D functor(params, Qdata);
					Kokkos::parallel_reduce(nbCells, functor, p);
				}
				KOKKOS_INLINE_FUNCTION
					void init (real_t& dst) const
					{
						dst = ZERO_F;
					} // init

				KOKKOS_INLINE_FUNCTION
					void operator()(const int& index, real_t& p) const
					{
						const int isize = params.isize;
						const int jsize = params.jsize;
						//const int imin =  params.imin;
						//const int ymin =  params.ymin;
						const int ghostWidth = params.ghostWidth;

						int i,j;
						index2coord(index,i,j,isize,jsize);
						if(j == ghostWidth  &&
								i == ghostWidth) 
						{
							p=Qdata(i, j, IP);

						}
					} // operator ()
				KOKKOS_INLINE_FUNCTION
					void join (volatile real_t& dst,
							const volatile real_t& src) const
					{
						// max reduce
						if (fabs(dst) < fabs(src))
							dst = src;
					} // join

				const DataArrayConst Qdata;
		}; // ComputePressureFunctor2D
		class ComputeEnergyConservation2Functor2D : public HydroBaseFunctor2D
		{
			public:
				ComputeEnergyConservation2Functor2D(HydroParams params_, DataArrayConst Udata_, DataArrayConst gradphi_, DataArrayConst UGdata_) :
					HydroBaseFunctor2D(params_), Udata(Udata_), gradphi(gradphi_), UGdata(UGdata_)  {};

				static void apply(HydroParams params, DataArrayConst Udata,DataArrayConst gradphi, DataArrayConst UGdata, real_t& Energy, int nbCells)
				{
					ComputeEnergyConservation2Functor2D functor(params, Udata, gradphi, UGdata);
					Kokkos::parallel_reduce(nbCells, functor, Energy);
				}


				// Tell each thread how to initialize its reduction result.
				KOKKOS_INLINE_FUNCTION
					void init (real_t& dst) const
					{
						dst = ZERO_F;
					} // init

				/* this is a reduce (max) functor */
				KOKKOS_INLINE_FUNCTION
					void operator()(const int& index, real_t& Energy) const
					{
						const int isize = params.isize;
						const int jsize = params.jsize;
						const int ghostWidth = params.ghostWidth;

						int i,j;
						index2coord(index,i,j,isize,jsize);

						if(j >= ghostWidth && j < jsize - ghostWidth &&
								i >= ghostWidth && i < isize - ghostWidth)
						{
							const real_t phiLoc= Udata(i, j, IH);
							const real_t phiMx = Udata(i-1, j, IH);
							const real_t phiPx = Udata(i+1, j, IH);
							const real_t phiMy = Udata(i, j-1, IH);
							const real_t phiPy = Udata(i, j+1, IH);
							real_t EnergyLoc=ZERO_F;

							if ((phiLoc*phiMx<=ZERO_F)||(phiLoc*phiPx<=ZERO_F)||(phiLoc*phiMy<=ZERO_F)||(phiLoc*phiPy<=ZERO_F))
							{
								if (fmax(phiLoc, fmax(phiMx, fmax(phiPx, fmax(phiMy, phiPy))))>ZERO_F)
								{
									const real_t dphix = gradphi(i, j, IPX);
									const real_t dphiy = gradphi(i, j, IPY);
									real_t surface=HALF_F + fabs(phiLoc/params.dx) * fmax(fabs(dphix), fabs(dphiy)) /sqrt(dphix*dphix + dphiy*dphiy);
									surface = surface > ONE_F ? ONE_F : surface;

									if (phiLoc<=ZERO_F)
									{
										EnergyLoc =  surface * Udata(i, j, IP) * params.dx*params.dy;
									}
									else
									{

										const real_t total_energy = UGdata(i, j, IE1);
										EnergyLoc =  (ONE_F-surface) * total_energy * params.dx*params.dy;
									}
								}
								else
								{
									if (phiLoc<=ZERO_F)
									{
										EnergyLoc = Udata(i, j, IP) * params.dx * params.dy;
									}
								}
							}
							else
							{
								if (phiLoc<=ZERO_F)
								{
									EnergyLoc = Udata(i, j, IP) * params.dx * params.dy;
								}
							}


							Energy += EnergyLoc;
						}
					} // operator ()

				KOKKOS_INLINE_FUNCTION
					void join (volatile real_t& dst,
							const volatile real_t& src) const
					{
						// max reduce
						//if (fabs(dst) < fabs(src))
						dst += src;
					} // join

				const DataArrayConst Udata;
				const DataArrayConst gradphi;
				const DataArrayConst UGdata;
		}; 
		class ComputeEnergyConservation1Functor2D : public HydroBaseFunctor2D
		{
			public:
				ComputeEnergyConservation1Functor2D(HydroParams params_, DataArrayConst Udata_, DataArrayConst gradphi_, DataArrayConst UGdata_) :
					HydroBaseFunctor2D(params_), Udata(Udata_), gradphi(gradphi_), UGdata(UGdata_)  {};

				static void apply(HydroParams params, DataArrayConst Udata,DataArrayConst gradphi, DataArrayConst UGdata, real_t& Energy, int nbCells)
				{
					ComputeEnergyConservation1Functor2D functor(params, Udata, gradphi,  UGdata);
					Kokkos::parallel_reduce(nbCells, functor, Energy);
				}


				// Tell each thread how to initialize its reduction result.
				KOKKOS_INLINE_FUNCTION
					void init (real_t& dst) const
					{
						dst = ZERO_F;
					} // init

				/* this is a reduce (max) functor */
				KOKKOS_INLINE_FUNCTION
					void operator()(const int& index, real_t& Energy) const
					{
						const int isize = params.isize;
						const int jsize = params.jsize;
						const int ghostWidth = params.ghostWidth;

						int i,j;
						index2coord(index,i,j,isize,jsize);

						if(j >= ghostWidth && j < jsize - ghostWidth &&
								i >= ghostWidth && i < isize - ghostWidth)
						{
							const real_t phiLoc= Udata(i, j, IH);
							const real_t phiMx = Udata(i-1, j, IH);
							const real_t phiPx = Udata(i+1, j, IH);
							const real_t phiMy = Udata(i, j-1, IH);
							const real_t phiPy = Udata(i, j+1, IH);
							real_t EnergyLoc=ZERO_F;

							if ((phiLoc*phiMx<=ZERO_F)||(phiLoc*phiPx<=ZERO_F)||(phiLoc*phiMy<=ZERO_F)||(phiLoc*phiPy<=ZERO_F))
							{
								if (fmax(phiLoc, fmax(phiMx, fmax(phiPx, fmax(phiMy, phiPy))))>ZERO_F)
								{
									const real_t dphix = gradphi(i, j, IPX);
									const real_t dphiy = gradphi(i, j, IPY);
									real_t surface=HALF_F + fabs(phiLoc/params.dx) * fmax(fabs(dphix), fabs(dphiy)) /sqrt(dphix*dphix + dphiy*dphiy);
									surface = surface > ONE_F ? ONE_F : surface;

									if (phiLoc>ZERO_F)
									{
										EnergyLoc =  surface * Udata(i, j, IP) * params.dx*params.dy;
									}
									else
									{
										const real_t total_energy = UGdata(i, j, IE0);
										EnergyLoc =  (ONE_F-surface) * total_energy * params.dx*params.dy;
									}
								}
								else
								{
									if (phiLoc>ZERO_F)
									{
										EnergyLoc = Udata(i, j, IP) * params.dx * params.dy;
									}
								}
							}
							else
							{
								if (phiLoc>ZERO_F)
								{
									EnergyLoc = Udata(i, j, IP) * params.dx * params.dy;
								}
							}


							Energy += EnergyLoc;
						}
					} // operator ()

				KOKKOS_INLINE_FUNCTION
					void join (volatile real_t& dst,
							const volatile real_t& src) const
					{
						// max reduce
						//if (fabs(dst) < fabs(src))
						dst += src;
					} // join

				const DataArrayConst Udata;
				const DataArrayConst gradphi;
				const DataArrayConst UGdata;
		}; 
		class ComputeMassConservation1Functor2D : public HydroBaseFunctor2D
		{
			public:
				ComputeMassConservation1Functor2D(HydroParams params_, DataArrayConst Udata_, DataArrayConst gradphi_, DataArrayConst UGdata_) :
					HydroBaseFunctor2D(params_), Udata(Udata_), gradphi(gradphi_), UGdata(UGdata_)  {};

				static void apply(HydroParams params,  DataArrayConst Udata, DataArrayConst gradphi,  DataArrayConst UGdata, real_t& mass, int nbCells)
				{
					ComputeMassConservation1Functor2D functor(params,Udata, gradphi, UGdata);
					Kokkos::parallel_reduce(nbCells, functor, mass);
				}


				// Tell each thread how to initialize its reduction result.
				KOKKOS_INLINE_FUNCTION
					void init (real_t& dst) const
					{
						dst = ZERO_F;
					} // init

				/* this is a reduce (max) functor */
				KOKKOS_INLINE_FUNCTION
					void operator()(const int& index, real_t& mass) const
					{
						const int isize = params.isize;
						const int jsize = params.jsize;
						const int ghostWidth = params.ghostWidth;

						int i,j;
						index2coord(index,i,j,isize,jsize);

						if(j >= ghostWidth && j < jsize - ghostWidth &&
								i >= ghostWidth && i < isize - ghostWidth)
						{

							const real_t phiLoc =Udata(i, j, IH);
							const real_t phiMx = Udata(i-1, j, IH);
							const real_t phiPx = Udata(i+1, j, IH);
							const real_t phiMy = Udata(i, j-1, IH);
							const real_t phiPy = Udata(i, j+1, IH);
							real_t massLoc=ZERO_F;

							if ((phiLoc*phiMx<=ZERO_F)||(phiLoc*phiPx<=ZERO_F)||(phiLoc*phiMy<=ZERO_F)||(phiLoc*phiPy<=ZERO_F))
							{
								if (fmax(phiLoc, fmax(phiMx, fmax(phiPx, fmax(phiMy, phiPy))))>ZERO_F)
								{
									const real_t dphix = gradphi(i, j, IPX);
									const real_t dphiy = gradphi(i, j, IPY);
									real_t surface=HALF_F + fabs(phiLoc/params.dx) * fmax(fabs(dphix), fabs(dphiy)) /sqrt(dphix*dphix + dphiy*dphiy);
									surface = surface > ONE_F ? ONE_F : surface;
									if (phiLoc>ZERO_F)
									{
										massLoc = (surface * Udata(i, j, ID))*params.dx*params.dy;
									}
									else
									{
										const real_t density = UGdata(i, j, ID0);
										massLoc = (ONE_F - surface) * density * params.dx*params.dy;
									}
								}
								else
								{
									if (phiLoc>ZERO_F)
									{
										massLoc = Udata(i, j, ID) * params.dx * params.dy;
									}
								}
							}
							else
							{
								if (phiLoc>ZERO_F)
								{
									massLoc = Udata(i, j, ID) * params.dx * params.dy;
								}
							}


							mass += massLoc;
						}
					} // operator ()

				KOKKOS_INLINE_FUNCTION
					void join (volatile real_t& dst,
							const volatile real_t& src) const
					{
						dst += src;
					} // join

				const DataArrayConst Udata;
				const DataArrayConst gradphi;
				const DataArrayConst UGdata;
		}; 
		class ComputeMassConservation2Functor2D : public HydroBaseFunctor2D
		{
			public:
				ComputeMassConservation2Functor2D(HydroParams params_, DataArrayConst Udata_, DataArrayConst gradphi_, DataArrayConst UGdata_) :
					HydroBaseFunctor2D(params_), Udata(Udata_), gradphi(gradphi_) , UGdata(UGdata_) {};

				static void apply(HydroParams params,  DataArrayConst Udata,DataArrayConst gradphi, DataArrayConst UGdata, real_t& mass, int nbCells)
				{
					ComputeMassConservation2Functor2D functor(params,Udata, gradphi, UGdata);
					Kokkos::parallel_reduce(nbCells, functor, mass);
				}


				// Tell each thread how to initialize its reduction result.
				KOKKOS_INLINE_FUNCTION
					void init (real_t& dst) const
					{
						dst = ZERO_F;
					} // init

				/* this is a reduce (max) functor */
				KOKKOS_INLINE_FUNCTION
					void operator()(const int& index, real_t& mass) const
					{
						const int isize = params.isize;
						const int jsize = params.jsize;
						const int ghostWidth = params.ghostWidth;

						int i,j;
						index2coord(index,i,j,isize,jsize);

						if(j >= ghostWidth && j < jsize - ghostWidth &&
								i >= ghostWidth && i < isize - ghostWidth)
						{

							const real_t phiLoc =Udata(i, j, IH);
							const real_t phiMx = Udata(i-1, j, IH);
							const real_t phiPx = Udata(i+1, j, IH);
							const real_t phiMy = Udata(i, j-1, IH);
							const real_t phiPy = Udata(i, j+1, IH);
							real_t massLoc=ZERO_F;

							if ((phiLoc*phiMx<=ZERO_F)||(phiLoc*phiPx<=ZERO_F)||(phiLoc*phiMy<=ZERO_F)||(phiLoc*phiPy<=ZERO_F))
							{
								if (fmax(phiLoc, fmax(phiMx, fmax(phiPx, fmax(phiMy, phiPy))))>ZERO_F)
								{
									const real_t dphix = gradphi(i, j, IPX);
									const real_t dphiy = gradphi(i, j, IPY);
									real_t surface=HALF_F + fabs(phiLoc/params.dx) * fmax(fabs(dphix), fabs(dphiy)) /sqrt(dphix*dphix + dphiy*dphiy);
									surface = surface > ONE_F ? ONE_F : surface;
									if (phiLoc<=ZERO_F)
									{
										massLoc = (surface * Udata(i, j, ID))*params.dx*params.dy;
									}
									else
									{
										const real_t density = UGdata(i, j, ID1);
										massLoc = (ONE_F - surface) * density * params.dx*params.dy;
									}
								}
								else
								{
									if (phiLoc<=ZERO_F)
									{
										massLoc = Udata(i, j, ID) * params.dx * params.dy;
									}
								}
							}
							else
							{
								if (phiLoc<=ZERO_F)
								{
									massLoc = Udata(i, j, ID) * params.dx * params.dy;
								}
							}


							mass += massLoc;
						}
					} // operator ()

				KOKKOS_INLINE_FUNCTION
					void join (volatile real_t& dst,
							const volatile real_t& src) const
					{
						dst += src;
					} // join

				const DataArrayConst Udata;
				const DataArrayConst gradphi;
				const DataArrayConst UGdata;
		}; 
		class ComputeErrorGreshoFunctor2D : public HydroBaseFunctor2D
		{
			public:
				ComputeErrorGreshoFunctor2D(HydroParams params_, DataArrayConst Qdata_) :
					HydroBaseFunctor2D(params_), Qdata(Qdata_)  {};

				static void apply(HydroParams params, DataArrayConst Qdata, real_t& umax, int nbCells)
				{
					ComputeErrorGreshoFunctor2D functor(params, Qdata);
					Kokkos::parallel_reduce(nbCells, functor, umax);
				}


				// Tell each thread how to initialize its reduction result.
				KOKKOS_INLINE_FUNCTION
					void init (real_t& dst) const
					{
						dst = ZERO_F;
					} // init

				/* this is a reduce (max) functor */
				KOKKOS_INLINE_FUNCTION
					void operator()(const int& index, real_t& umax) const
					{
						const int isize = params.isize;
						const int jsize = params.jsize;
						const int ghostWidth = params.ghostWidth;

						int i,j;
						index2coord(index,i,j,isize,jsize);

						if(j >= ghostWidth && j < jsize - ghostWidth &&
								i >= ghostWidth && i < isize - ghostWidth)
						{
#ifdef USE_MPI
							const int i_mpi = params.myMpiPos[IX];
							const int j_mpi = params.myMpiPos[IY];
#else
							const int i_mpi = 0;
							const int j_mpi = 0;
#endif

							real_t y = params.ymin + (params.ymax-params.ymin)*(j+params.ny*j_mpi-ghostWidth)*params.dy + params.dy/2.0;
							y -= HALF_F;

							real_t x = params.xmin + (params.xmax-params.xmin)*(i+params.nx*i_mpi-ghostWidth)*params.dx + params.dx/2.0;
							x -= HALF_F;

							const real_t r = std::sqrt(x*x + y*y);
							real_t u;

							if (r <= 0.2)
							{
								u = 5.0 * r;
							}
							else if (r <= 0.4)
							{
								u = 2.0 - 5.0 * r;
							}
							else
							{
								u = 0.0;
							}

							const HydroState qLoc = getHydroState(Qdata, i, j);
							umax +=  FABS( u - sqrt(qLoc[IU]*qLoc[IU]+qLoc[IV]*qLoc[IV]));
						}
					} // operator ()

				KOKKOS_INLINE_FUNCTION
					void join (volatile real_t& dst,
							const volatile real_t& src) const
					{
						// max reduce
						//if (fabs(dst) < fabs(src))
						dst += src;
					} // join

				const DataArrayConst Qdata;
		}; 
		class ComputeMaxVelocityFunctor2D : public HydroBaseFunctor2D
		{
			public:
				ComputeMaxVelocityFunctor2D(HydroParams params_, DataArrayConst Qdata_) :
					HydroBaseFunctor2D(params_), Qdata(Qdata_)  {};

				static void apply(HydroParams params, DataArrayConst Qdata, real_t& umax, int nbCells)
				{
					ComputeMaxVelocityFunctor2D functor(params, Qdata);
					Kokkos::parallel_reduce(nbCells, functor, umax);
				}


				// Tell each thread how to initialize its reduction result.
				KOKKOS_INLINE_FUNCTION
					void init (real_t& dst) const
					{
						dst = ZERO_F;
					} // init

				/* this is a reduce (max) functor */
				KOKKOS_INLINE_FUNCTION
					void operator()(const int& index, real_t& umax) const
					{
						const int isize = params.isize;
						const int jsize = params.jsize;
						const int ghostWidth = params.ghostWidth;

						int i,j;
						index2coord(index,i,j,isize,jsize);

						if(j >= ghostWidth && j <= jsize - ghostWidth &&
								i >= ghostWidth && i <= isize - ghostWidth)
						{
							const HydroState qLoc = getHydroState(Qdata, i, j);
							umax = FMAX(umax, sqrt(qLoc[IU]*qLoc[IU]+qLoc[IV]*qLoc[IV]));
						}
					} // operator ()

				// "Join" intermediate results from different threads.
				// This should normally implement the same reduction
				// operation as operator() above. Note that both input
				// arguments MUST be declared volatile.
				KOKKOS_INLINE_FUNCTION
					void join (volatile real_t& dst,
							const volatile real_t& src) const
					{
						// max reduce
						if (dst < src)
							dst = src;
					} // join

				const DataArrayConst Qdata;
		}; // ComputeMaxVelocityFunctor2D
	}
}
