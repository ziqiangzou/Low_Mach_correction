
#pragma once

#include "Extrapolation2D.h"
#include "Extrapolation3D.h"

namespace euler_kokkos { namespace all_regime
	{

		template<int dim>
			struct ExtrapolationFunctors
			{
			};

		template<>
			struct ExtrapolationFunctors<2>
			{
				using ComputeFirstDerivative      = ComputeFirstDerivativeFunctor2D;
				using Extrapolate                 = ExtrapolateFunctor2D;
				using ExtrapolateFirstDerivative  = ExtrapolateFirstDerivativeFunctor2D;
				using ExtrapolateStep2                 = ExtrapolateStep2Functor2D;
				using ExtrapolateFirstDerivativeStep2  = ExtrapolateFirstDerivativeStep2Functor2D;
				using CopyToGhost                 = CopyToGhostFunctor2D;
                                 
			};
		template<>
			struct ExtrapolationFunctors<3>
			{
				using ComputeFirstDerivative      = ComputeFirstDerivativeFunctor3D;
				using Extrapolate                 = ExtrapolateFunctor3D;
				using ExtrapolateFirstDerivative  = ExtrapolateFirstDerivativeFunctor3D;
				using ExtrapolateStep2                 = ExtrapolateStep2Functor3D;
				using ExtrapolateFirstDerivativeStep2  = ExtrapolateFirstDerivativeStep2Functor3D;
				using CopyToGhost                 = CopyToGhostFunctor3D;
			};

	} // namespace all_regime

} // namespace euler_kokkos
