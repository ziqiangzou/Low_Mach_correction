#pragma once

#include "HydroInitFunctors2D.h"
#include "HydroInitFunctors3D.h"

namespace euler_kokkos{ namespace muscl
{

template <int dim>
struct InitFunctors
{
};

template <>
struct InitFunctors<2>
{
    using AtmosphereAtRest = InitAtmosphereAtRestFunctor2D;
    using Blast            = InitBlastFunctor2D;
    using FourQuadrant     = InitFourQuadrantFunctor2D;
    using Gresho           = InitGreshoFunctor2D;
    using Implode          = InitImplodeFunctor2D;
    using IsentropicVortex = InitIsentropicVortexFunctor2D;
    using RayleighTaylor   = InitRayleighTaylorFunctor2D;
    using RiemannProblem   = InitRiemannProblemFunctor2D;
};

template <>
struct InitFunctors<3>
{
    using Blast          = InitBlastFunctor3D;
    using Implode        = InitImplodeFunctor3D;
};

} // namespace muscl

} // namespace euler_kokkos
