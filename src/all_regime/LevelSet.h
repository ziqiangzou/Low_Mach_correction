
#pragma once

#include "LevelSet2D.h"
#include "LevelSet3D.h"
#include "HydroBaseFunctor2D.h"

namespace euler_kokkos { namespace all_regime
{

template<int dim>
struct LevelSetFunctors
{
};

template<>
struct LevelSetFunctors<2>
{
    using TransportPhiXStep   = ComputeTransportPhiXFunctor2D;
    using TransportPhiYStep   = ComputeTransportPhiYFunctor2D;
    using RedistancingPhiStep1 = ComputeRedistancingPhi1Functor2D;
    using RedistancingPhiStep2 = ComputeRedistancingPhi2Functor2D;
    using RedistancingPhiStep3 = ComputeRedistancingPhi3Functor2D;
    using TransportPhiStep1   = ComputePhiTransport1Functor2D;
    using TransportPhiStep2   = ComputePhiTransport2Functor2D;
    using TransportPhiStep3   = ComputePhiTransport3Functor2D;
    using GradientPhiTransport= ComputeGradPhiTransportFunctor2D;
    using CopyLS              = CopyLSFunctor2D;
    using ComputeGradPhiStep  = ComputeGradPhiFunctor2D;
    using ComputeCurvature    = ComputeCurvatureFunctor2D;
};
template<>
struct LevelSetFunctors<3>
{
    using TransportPhiXStep    = ComputeTransportPhiXFunctor3D;
    using TransportPhiYStep    = ComputeTransportPhiYFunctor3D;
    using RedistancingPhiStep1 = ComputeRedistancingPhi1Functor3D;
    using RedistancingPhiStep2 = ComputeRedistancingPhi2Functor3D;
    using RedistancingPhiStep3 = ComputeRedistancingPhi3Functor3D;
    using TransportPhiStep1   = ComputePhiTransport1Functor3D;
    using TransportPhiStep2   = ComputePhiTransport2Functor3D;
    using TransportPhiStep3   = ComputePhiTransport3Functor3D;
    using ComputeCurvature    = ComputeCurvatureFunctor3D;
    using GradientPhiTransport= ComputeGradPhiTransportFunctor3D;
    using ComputeGradPhiStep  = ComputeGradPhiFunctor3D;
    using CopyLS              = CopyLSFunctor3D;
};

} // namespace all_regime

} // namespace euler_kokkos
