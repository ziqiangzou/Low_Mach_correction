#pragma once

#include "HydroInitFunctors2D.h"
#include "HydroInitFunctors3D.h"

namespace euler_kokkos{ namespace all_regime
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
    using DamBreak         = InitDamBreakFunctor2D;
    using FourQuadrant     = InitFourQuadrantFunctor2D;
    using Gresho           = InitGreshoFunctor2D;
    using Implode          = InitImplodeFunctor2D;
    using IsentropicVortex = InitIsentropicVortexFunctor2D;
    using Poiseuille       = InitPoiseuilleFunctor2D;
    using RayleighBenard   = InitRayleighBenardFunctor2D;
    using NonIsotherm      = InitNonIsothermFunctor2D;
    using Stefantherm      = InitStefanthermFunctor2D;
    using Suckingtherm     = InitSuckingthermFunctor2D;
    using Transient        = InitTransientFunctor2D;
    using RayleighTaylor   = InitRayleighTaylorFunctor2D;
    using DropOsicillation = InitDropOsicillationFunctor2D;
    using StaticBubble     = InitStaticBubbleFunctor2D;
    using RisingBubble     = InitRisingBubbleFunctor2D;
    using Case34           = InitCase34Functor2D;
    using RayleighTaylorInstabilities   = InitRayleighTaylorInstabilitiesFunctor2D;
    using RiemannProblem   = InitRiemannProblemFunctor2D;
};

template <>
struct InitFunctors<3>
{
    using Blast          = InitBlastFunctor3D;
    using Implode        = InitImplodeFunctor3D;
    using Poiseuille     = InitPoiseuilleFunctor3D;
    using RayleighTaylor = InitRayleighTaylorFunctor3D;
};

} // namespace all_regime

} // namespace euler_kokkos
