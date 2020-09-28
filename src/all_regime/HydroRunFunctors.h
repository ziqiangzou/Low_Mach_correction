#pragma once

#include "HydroRunFunctors2D.h"
#include "HydroRunFunctors3D.h"

namespace euler_kokkos { namespace all_regime
{

template<int dim>
struct RunFunctors
{
};

template<>
struct RunFunctors<2>
{
    using AcousticTimeStep    = ComputeAcousticDtFunctor2D;
    using TransportTimeStep   = ComputeTransportDtFunctor2D;
    using TimeStep            = ComputeDtFunctor2D;
    using AcousticStep        = ComputeAcousticStepFunctor2D;
    using TransportStep       = ComputeTransportStepFunctor2D;
    using StateChangeStep     = ComputeStateChangeFunctor2D;
    using ViscosityStep       = ComputeViscosityStepFunctor2D;
    using HeatDiffusionStep   = ComputeHeatDiffusionStepFunctor2D;
    using ConvertToPrimitives = ConvertToPrimitivesFunctor2D;
    using CopyGradient        = CopyGradientFunctor2D;
};

template<>
struct RunFunctors<3>
{
    using AcousticTimeStep    = ComputeAcousticDtFunctor3D;
    using TransportTimeStep   = ComputeTransportDtFunctor3D;
    using TimeStep            = ComputeDtFunctor3D;
    using AcousticStep        = ComputeAcousticStepFunctor3D;
    using TransportStep       = ComputeTransportStepFunctor3D;
    using StateChangeStep     = ComputeStateChangeFunctor3D;
    using ViscosityStep       = ComputeViscosityStepFunctor3D;
    using HeatDiffusionStep   = ComputeHeatDiffusionStepFunctor3D;
    using ConvertToPrimitives = ConvertToPrimitivesFunctor3D;
    using CopyGradient        = CopyGradientFunctor3D;
    
};

} // namespace all_regime

} // namespace euler_kokkos
