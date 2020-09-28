
#pragma once

#include "PostProcessing2D.h"
#include "PostProcessing3D.h"

namespace euler_kokkos { namespace all_regime
{

template<int dim>
struct PostProcessingFunctors
{
};

template<>
struct PostProcessingFunctors<2>
{
    using ComputeMaxVelocity    = ComputeMaxVelocityFunctor2D;
    using ComputeErrorGresho    = ComputeErrorGreshoFunctor2D;
    using ComputeMassConservation1 = ComputeMassConservation1Functor2D;
    using ComputeEnergyConservation1 = ComputeEnergyConservation1Functor2D;
    using ComputeMassConservation2 = ComputeMassConservation2Functor2D;
    using ComputeEnergyConservation2 = ComputeEnergyConservation2Functor2D;
    using ComputePressure      = ComputePressureFunctor2D;
    using ComputePressureR      = ComputePressureRFunctor2D;
    using ComputePressureC      = ComputePressureCFunctor2D;
    using ComputeVTProfile    = ComputeVTProfileFunctor2D;
    using ComputeInterfacePosition    = ComputeInterfacePositionFunctor2D;
    using ComputeInterfaceRPosition    = ComputeInterfaceRPositionFunctor2D;
};
template<>
struct PostProcessingFunctors<3>
{
    using ComputeErrorGresho    = ComputeErrorGreshoFunctor3D;
    using ComputeMassConservation1 = ComputeMassConservation1Functor3D;
    using ComputeEnergyConservation1 = ComputeEnergyConservation1Functor3D;
    using ComputeMassConservation2 = ComputeMassConservation2Functor3D;
    using ComputeEnergyConservation2 = ComputeEnergyConservation2Functor3D;
    using ComputeMaxVelocity    = ComputeMaxVelocityFunctor3D;
    using ComputePressure      = ComputePressureFunctor3D;
    using ComputeVTProfile    = ComputeVTProfileFunctor3D;
    using ComputePressureR      = ComputePressureRFunctor3D;
    using ComputePressureC      = ComputePressureCFunctor3D;
    using ComputeInterfacePosition    = ComputeInterfacePositionFunctor3D;
    using ComputeInterfaceRPosition    = ComputeInterfaceRPositionFunctor3D;
};
}
}
