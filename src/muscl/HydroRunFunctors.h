#pragma once

#include "HydroRunFunctors2D.h"
#include "HydroRunFunctors3D.h"

namespace euler_kokkos { namespace muscl
{

template<int dim>
struct RunFunctors
{
};

template<>
struct RunFunctors<2>
{
    using ComputeGravityStep    = ComputeGravityStep2D;
    using ComputeAndStoreFluxes = ComputeAndStoreFluxesFunctor2D;
    using ComputeSlopes         = ComputeSlopesFunctor2D;
    template <Direction dir>
    using ComputeTraceAndFluxes = ComputeTraceAndFluxes_Functor2D<dir>;
    using Update                = UpdateFunctor2D;
    template <Direction dir>
    using UpdateDir             = UpdateDirFunctor2D<dir>;
    using TimeStep              = ComputeDtFunctor2D;
    using ConvertToPrimitives   = ConvertToPrimitivesFunctor2D;
};

template<>
struct RunFunctors<3>
{
    using ComputeGravityStep    = ComputeGravityStep3D;
    using ComputeAndStoreFluxes = ComputeAndStoreFluxesFunctor3D;
    using ComputeSlopes         = ComputeSlopesFunctor3D;
    template <Direction dir>
    using ComputeTraceAndFluxes = ComputeTraceAndFluxes_Functor3D<dir>;
    template <Direction dir>
    using UpdateDir             = UpdateDirFunctor3D<dir>;
    using Update                = UpdateFunctor3D;
    using TimeStep              = ComputeDtFunctor3D;
    using ConvertToPrimitives   = ConvertToPrimitivesFunctor3D;
};

} // namespace muscl

} // namespace euler_kokkos
