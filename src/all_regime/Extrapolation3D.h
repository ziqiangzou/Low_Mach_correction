
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
#include "HydroBaseFunctor3D.h"
#include "shared/RiemannSolvers.h"
#include "fstream"
#include<ctime>

namespace euler_kokkos { namespace all_regime
	{
		class CopyToGhostFunctor3D : HydroBaseFunctor3D
		{
			public:
				CopyToGhostFunctor3D(HydroParams params_,
						DataArray Udata_, DataArray U0data_) :
					HydroBaseFunctor3D(params_),
					Udata(Udata_), U0data(U0data_){};

				static void apply(HydroParams params,
						DataArray Udata, DataArray U0data, 
						int nbCells)
				{
					CopyToGhostFunctor3D functor(params, Udata, U0data);
					Kokkos::parallel_for(nbCells, functor);
				}
				KOKKOS_INLINE_FUNCTION
					void operator()(int index) const
					{
					}

				const DataArray Udata;
				const DataArray U0data;
		}; // CopyToGhostFunctor3D
		class ComputeFirstDerivativeFunctor3D : HydroBaseFunctor3D
		{
			public:
				ComputeFirstDerivativeFunctor3D(HydroParams params_,
						DataArray Qdata_, DataArray U0data_, DataArray gradphi_) :
					HydroBaseFunctor3D(params_),
					Qdata(Qdata_), U0data(U0data_), gradphi(gradphi_) {};

				static void apply(HydroParams params,
						DataArray Qdata, DataArray U0data, DataArray gradphi,
						int nbCells)
				{
					ComputeFirstDerivativeFunctor3D functor(params, Qdata, U0data, gradphi);
					Kokkos::parallel_for(nbCells, functor);
				}
				KOKKOS_INLINE_FUNCTION
					void operator()(int index) const
					{
					}

				const DataArray Qdata;
				const DataArray U0data;
				const DataArray gradphi;
		}; // ComputeFirstDerivetiveFunctor3D
		class ExtrapolateStep2Functor3D : HydroBaseFunctor3D
		{
			public:
				ExtrapolateStep2Functor3D(HydroParams params_,
						DataArray Qdata_, DataArray U0data_, DataArray gradphi_) :
					HydroBaseFunctor3D(params_),
					Qdata(Qdata_), U0data(U0data_), gradphi(gradphi_) {};

				static void apply(HydroParams params,
						DataArray Qdata, DataArray U0data, DataArray gradphi,
						int nbCells)
				{
					ExtrapolateStep2Functor3D functor(params, Qdata, U0data, gradphi);
					Kokkos::parallel_for(nbCells, functor);
				}

				KOKKOS_INLINE_FUNCTION
					void operator()(int index) const
					{
					}
				const DataArray Qdata;
				const DataArray U0data;
				const DataArray gradphi;
		};// ExtrapolateDensityPressureFunctor3D
		class ExtrapolateFunctor3D : HydroBaseFunctor3D
		{
			public:
				ExtrapolateFunctor3D(HydroParams params_,
						DataArray Qdata_, DataArray U0data_, DataArray gradphi_) :
					HydroBaseFunctor3D(params_),
					Qdata(Qdata_), U0data(U0data_), gradphi(gradphi_) {};

				static void apply(HydroParams params,
						DataArray Qdata, DataArray U0data, DataArray gradphi,
						int nbCells)
				{
					ExtrapolateFunctor3D functor(params, Qdata, U0data, gradphi);
					Kokkos::parallel_for(nbCells, functor);
				}

				KOKKOS_INLINE_FUNCTION
					void operator()(int index) const
					{
					}
				const DataArray Qdata;
				const DataArray U0data;
				const DataArray gradphi;
		};// ExtrapolateDensityPressureFunctor3D
		class ExtrapolateFirstDerivativeStep2Functor3D : HydroBaseFunctor3D
		{
			public:
				ExtrapolateFirstDerivativeStep2Functor3D(HydroParams params_,
						DataArray Qdata_, DataArray U0data_, DataArray gradphi_) :
					HydroBaseFunctor3D(params_),
					Qdata(Qdata_), U0data(U0data_), gradphi(gradphi_) {};

				static void apply(HydroParams params,
						DataArray Qdata, DataArray U0data, DataArray gradphi,
						int nbCells)
				{
					ExtrapolateFirstDerivativeStep2Functor3D functor(params, Qdata, U0data, gradphi);
					Kokkos::parallel_for(nbCells, functor);
				}

				KOKKOS_INLINE_FUNCTION
					void operator()(int index) const
					{
					}
				const DataArray Qdata;
				const DataArray U0data;
				const DataArray gradphi;
		};// ExtrapolateFirstDerivetiveFunctor3D
		class ExtrapolateFirstDerivativeFunctor3D : HydroBaseFunctor3D
		{
			public:
				ExtrapolateFirstDerivativeFunctor3D(HydroParams params_,
						DataArray Qdata_, DataArray U0data_, DataArray gradphi_) :
					HydroBaseFunctor3D(params_),
					Qdata(Qdata_), U0data(U0data_), gradphi(gradphi_) {};

				static void apply(HydroParams params,
						DataArray Qdata, DataArray U0data, DataArray gradphi,
						int nbCells)
				{
					ExtrapolateFirstDerivativeFunctor3D functor(params, Qdata, U0data, gradphi);
					Kokkos::parallel_for(nbCells, functor);
				}

				KOKKOS_INLINE_FUNCTION
					void operator()(int index) const
					{
					}
				const DataArray Qdata;
				const DataArray U0data;
				const DataArray gradphi;
		};// ExtrapolateFirstDerivetiveFunctor3D
	}
}
