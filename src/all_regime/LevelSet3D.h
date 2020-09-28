
#pragma once

#include <limits> // for std::numeric_limits
#ifdef __CUDA_ARCH__
#include <math_constants.h>
#endif // __CUDA_ARCH__

#include "shared/kokkos_shared.h"
#include "HydroBaseFunctor3D.h"
#include "shared/RiemannSolvers.h"

// init conditions
#include "shared/BlastParams.h"

namespace euler_kokkos { namespace all_regime
	{
		class ComputeTransportPhiXFunctor3D : HydroBaseFunctor3D
		{
			public:
				ComputeTransportPhiXFunctor3D(HydroParams params_, DataArray Udata_, DataArray U2data_, DataArrayConst Qdata_, DataArray gradphi_, real_t dt_):
					HydroBaseFunctor3D(params_), Udata(Udata_), U2data(U2data_), Qdata(Qdata_), gradphi(gradphi_), dt(dt_) {};
				static void apply(HydroParams params,
						DataArray Udata, DataArray U2data, DataArrayConst Qdata, DataArray gradphi, real_t dt,
						int nbCells)
				{
					ComputeTransportPhiXFunctor3D functor(params, Udata, U2data, Qdata, gradphi, dt);
					Kokkos::parallel_for(nbCells, functor);
				}
				KOKKOS_INLINE_FUNCTION
					void operator()(int index) const
					{

					}
				const DataArray Udata;
				const DataArray U2data;
				const DataArrayConst Qdata;
				const DataArray gradphi;
				real_t dt;
		};
		class ComputeCurvatureFunctor3D : HydroBaseFunctor3D
		{
			public:
				ComputeCurvatureFunctor3D(HydroParams params_,  DataArray U2data_, DataArray gradphi_):
					HydroBaseFunctor3D(params_), U2data(U2data_),  gradphi(gradphi_) {};
				static void apply(HydroParams params,
						DataArray U2data, DataArray gradphi,
						int nbCells)
				{
					ComputeCurvatureFunctor3D functor(params, U2data, gradphi);
					Kokkos::parallel_for(nbCells, functor);
				}
				KOKKOS_INLINE_FUNCTION
					void operator()(int index) const
					{
					}
				const DataArray U2data;
				const DataArray gradphi;
		};

		class ComputeGradPhiFunctor3D : HydroBaseFunctor3D
		{
			public:
				ComputeGradPhiFunctor3D(HydroParams params_, DataArray Udata_, DataArray gradphi_):
					HydroBaseFunctor3D(params_), Udata(Udata_), gradphi(gradphi_) {};
				static void apply(HydroParams params,
						DataArray Udata,  DataArray gradphi,
						int nbCells)
				{
					ComputeGradPhiFunctor3D functor(params, Udata, gradphi);
					Kokkos::parallel_for(nbCells, functor);
				}
				KOKKOS_INLINE_FUNCTION
					void operator()(int index) const
					{
					}
				const DataArray Udata;
				const DataArray gradphi;
		};
		class ComputeRedistancingPhi1Functor3D : HydroBaseFunctor3D
		{
			public:
				ComputeRedistancingPhi1Functor3D(HydroParams params_, DataArray Udata_, DataArray U2data_, DataArray gradphi_):
					HydroBaseFunctor3D(params_), Udata(Udata_),U2data(U2data_),  gradphi(gradphi_) {};
				static void apply(HydroParams params,
						DataArray Udata, DataArray U2data, DataArray gradphi,
						int nbCells)
				{
					ComputeRedistancingPhi1Functor3D functor(params, Udata, U2data, gradphi);
					Kokkos::parallel_for(nbCells, functor);
				}
				KOKKOS_INLINE_FUNCTION
					void operator()(int index) const
					{
					}
				const DataArray Udata;
				const DataArray U2data;
				const DataArray gradphi;
		};
		class ComputeRedistancingPhi2Functor3D : HydroBaseFunctor3D
		{
			public:
				ComputeRedistancingPhi2Functor3D(HydroParams params_, DataArray Udata_, DataArray U2data_, DataArray gradphi_):
					HydroBaseFunctor3D(params_), Udata(Udata_),U2data(U2data_),  gradphi(gradphi_) {};
				static void apply(HydroParams params,
						DataArray Udata, DataArray U2data, DataArray gradphi,
						int nbCells)
				{
					ComputeRedistancingPhi2Functor3D functor(params, Udata, U2data, gradphi);
					Kokkos::parallel_for(nbCells, functor);
				}
				KOKKOS_INLINE_FUNCTION
					void operator()(int index) const
					{
					}
				const DataArray Udata;
				const DataArray U2data;
				const DataArray gradphi;
		};
		class ComputeRedistancingPhi3Functor3D : HydroBaseFunctor3D
		{
			public:
				ComputeRedistancingPhi3Functor3D(HydroParams params_, DataArray Udata_, DataArray U2data_, DataArray gradphi_):
					HydroBaseFunctor3D(params_), Udata(Udata_),U2data(U2data_),  gradphi(gradphi_) {};
				static void apply(HydroParams params,
						DataArray Udata, DataArray U2data, DataArray gradphi,
						int nbCells)
				{
					ComputeRedistancingPhi3Functor3D functor(params, Udata, U2data, gradphi);
					Kokkos::parallel_for(nbCells, functor);
				}
				KOKKOS_INLINE_FUNCTION
					void operator()(int index) const
					{
					}
				const DataArray Udata;
				const DataArray U2data;
				const DataArray gradphi;
		};
		class ComputePhiTransport3Functor3D : HydroBaseFunctor3D
		{
			public:
				ComputePhiTransport3Functor3D(HydroParams params_, DataArray Udata_,DataArray U2data_, 
						DataArrayConst Qdata_, DataArray gradphi_, real_t dt_):
					HydroBaseFunctor3D(params_), Udata(Udata_), U2data(U2data_),Qdata(Qdata_),gradphi(gradphi_), dt(dt_) {};
				static void apply(HydroParams params,
						DataArray Udata, DataArray U2data, DataArrayConst Qdata, DataArray gradphi, real_t dt,
						int nbCells)
				{
					ComputePhiTransport3Functor3D functor(params, Udata, U2data, Qdata, gradphi, dt);
					Kokkos::parallel_for(nbCells, functor);
				}
				KOKKOS_INLINE_FUNCTION
					void operator()(int index) const
					{
					}

				const DataArray Udata;
				const DataArray U2data;
				const DataArrayConst Qdata;
				const DataArray gradphi;
				real_t dt;
		};//ComputeTransportGradPhiFunctor_Scheme_OS
		class ComputePhiTransport2Functor3D : HydroBaseFunctor3D
		{
			public:
				ComputePhiTransport2Functor3D(HydroParams params_, DataArray Udata_,DataArray U2data_, 
						DataArrayConst Qdata_, DataArray gradphi_, real_t dt_):
					HydroBaseFunctor3D(params_), Udata(Udata_), U2data(U2data_),Qdata(Qdata_),gradphi(gradphi_), dt(dt_) {};
				static void apply(HydroParams params,
						DataArray Udata, DataArray U2data, DataArrayConst Qdata, DataArray gradphi, real_t dt,
						int nbCells)
				{
					ComputePhiTransport2Functor3D functor(params, Udata, U2data, Qdata, gradphi, dt);
					Kokkos::parallel_for(nbCells, functor);
				}
				KOKKOS_INLINE_FUNCTION
					void operator()(int index) const
					{
					}

				const DataArray Udata;
				const DataArray U2data;
				const DataArrayConst Qdata;
				const DataArray gradphi;
				real_t dt;
		};//ComputeTransportGradPhiFunctor_Scheme_OS
		class ComputePhiTransport1Functor3D : HydroBaseFunctor3D
		{
			public:
				ComputePhiTransport1Functor3D(HydroParams params_, DataArray Udata_,DataArray U2data_, 
						DataArrayConst Qdata_, DataArray gradphi_, real_t dt_):
					HydroBaseFunctor3D(params_), Udata(Udata_), U2data(U2data_),Qdata(Qdata_),gradphi(gradphi_), dt(dt_) {};
				static void apply(HydroParams params,
						DataArray Udata, DataArray U2data, DataArrayConst Qdata, DataArray gradphi, real_t dt,
						int nbCells)
				{
					ComputePhiTransport1Functor3D functor(params, Udata, U2data, Qdata, gradphi, dt);
					Kokkos::parallel_for(nbCells, functor);
				}
				KOKKOS_INLINE_FUNCTION
					void operator()(int index) const
					{
					}

				const DataArray Udata;
				const DataArray U2data;
				const DataArrayConst Qdata;
				const DataArray gradphi;
				real_t dt;
		};//ComputeTransportGradPhiFunctor_Scheme_OS
		class ComputeGradPhiTransportFunctor3D : HydroBaseFunctor3D
		{
			public:
				ComputeGradPhiTransportFunctor3D(HydroParams params_, DataArray Udata_, DataArray gradphi_):
					HydroBaseFunctor3D(params_), Udata(Udata_), gradphi(gradphi_) {};
				static void apply(HydroParams params,
						DataArray Udata,  DataArray gradphi,
						int nbCells)
				{
					ComputeGradPhiTransportFunctor3D functor(params, Udata, gradphi);
					Kokkos::parallel_for(nbCells, functor);
				}
				KOKKOS_INLINE_FUNCTION
					void operator()(int index) const
					{
					}

				const DataArray Udata;
				const DataArray gradphi;
		};//ComputeTransportGradPhiFunctor_Scheme_OS
		class CopyLSFunctor3D : HydroBaseFunctor3D
		{
			public:
				CopyLSFunctor3D(HydroParams params_, DataArray Udata_, DataArray U2data_):
					HydroBaseFunctor3D(params_), Udata(Udata_), U2data(U2data_) {};
				static void apply(HydroParams params,
						DataArray Udata, DataArray U2data, 
						int nbCells)
				{
					CopyLSFunctor3D functor(params, Udata, U2data);
					Kokkos::parallel_for(nbCells, functor);
				}
				KOKKOS_INLINE_FUNCTION
					void operator()(int index) const
					{
					}
				const DataArray Udata;
				const DataArray U2data;
		};
		class ComputeTransportPhiYFunctor3D : HydroBaseFunctor3D
		{
			public:
				ComputeTransportPhiYFunctor3D(HydroParams params_, DataArray Udata_, DataArray U2data_, DataArrayConst Qdata_, DataArray gradphi_, real_t dt_):
					HydroBaseFunctor3D(params_), Udata(Udata_), U2data(U2data_), Qdata(Qdata_), gradphi(gradphi_), dt(dt_) {};
				static void apply(HydroParams params,
						DataArray Udata, DataArray U2data, DataArrayConst Qdata, DataArray gradphi, real_t dt,
						int nbCells)
				{
					ComputeTransportPhiYFunctor3D functor(params, Udata, U2data, Qdata, gradphi,  dt);
					Kokkos::parallel_for(nbCells, functor);
				}
				KOKKOS_INLINE_FUNCTION
					void operator()(int index) const
					{
					}
				const DataArray Udata;
				const DataArray U2data;
				const DataArrayConst Qdata;
				const DataArray gradphi;
				real_t dt;
		};
	}
}
