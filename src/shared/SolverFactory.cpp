#include "shared/SolverFactory.h"
#include "shared/SolverBase.h"
#include "muscl/SolverHydroMuscl.h"
#include "all_regime/SolverHydroAllRegime.h"

namespace euler_kokkos
{

// The main solver creation routine
SolverFactory::SolverFactory()
{
    /*
     * Register some possible solvers
     */
    registerSolver("Hydro_Muscl_2D", &muscl::SolverHydroMuscl<2>::create);
    registerSolver("Hydro_Muscl_3D", &muscl::SolverHydroMuscl<3>::create);
    registerSolver("Hydro_All_Regime_2D",   &all_regime::SolverHydroAllRegime<2>::create);
    registerSolver("Hydro_All_Regime_3D",   &all_regime::SolverHydroAllRegime<3>::create);
} // SolverFactory::SolverFactory

} // namespace euler_kokkos
