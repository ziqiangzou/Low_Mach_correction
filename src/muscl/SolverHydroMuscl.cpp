#include <string>
#include <cstdio>
#include <cstdbool>
#include <sstream>
#include <fstream>
#include <algorithm>

#include "muscl/SolverHydroMuscl.h"
#include "muscl/HydroInitFunctors.h"
#include "muscl/HydroRunFunctors.h"

#include "shared/HydroParams.h"

#include "shared/BlastParams.h"
#include "shared/initRiemannConfig2d.h"
#include "shared/IsentropicVortexParams.h"

namespace euler_kokkos { namespace muscl
{

// =======================================================
// =======================================================
template<>
void SolverHydroMuscl<2>::init(DataArray Udata)
{
    /*
     * initialize hydro array at t=0
     */
    if (m_problem_name == "implode")
    {
        InitFunctors<2>::Implode::apply(params, Udata, nbCells);
    }
    else if (m_problem_name == "atmosphere_at_rest")
    {
        InitFunctors<2>::AtmosphereAtRest::apply(params, Udata, nbCells);
    }
    else if (m_problem_name == "blast")
    {
        BlastParams blastParams = BlastParams(configMap);
        InitFunctors<2>::Blast::apply(params, blastParams, Udata, nbCells);
    }
    else if (m_problem_name == "gresho")
    {
        GreshoParams greshoParams(configMap);
        InitFunctors<2>::Gresho::apply(params, greshoParams, Udata, nbCells);
    }
    else if (m_problem_name == "rayleigh_taylor")
    {
        InitFunctors<2>::RayleighTaylor::apply(params, Udata, nbCells);
    }
    else if (m_problem_name == "four_quadrant")
    {
        int configNumber = configMap.getInteger("riemann2d","config_number",0);
        real_t xt = configMap.getFloat("riemann2d","x",0.8);
        real_t yt = configMap.getFloat("riemann2d","y",0.8);

        HydroState2d U0, U1, U2, U3;
        getRiemannConfig2d(configNumber, U0, U1, U2, U3);

        primToCons_2D(U0, params.settings.gamma0);
        primToCons_2D(U1, params.settings.gamma0);
        primToCons_2D(U2, params.settings.gamma0);
        primToCons_2D(U3, params.settings.gamma0);

        InitFunctors<2>::FourQuadrant::apply(params, Udata, configNumber,
                                             U0, U1, U2, U3,
                                             xt, yt, nbCells);
    }
    else if (m_problem_name == "isentropic_vortex")
    {
        IsentropicVortexParams iparams(configMap);
        InitFunctors<2>::IsentropicVortex::apply(params, iparams, Udata, nbCells);
    }
    else if (m_problem_name == "riemann_problem")
    {
        RiemannProblemParams rp_params(configMap);
        InitFunctors<2>::RiemannProblem::apply(params, rp_params, Udata, nbCells);
    }
    else
    {
        std::cout << "Problem : " << m_problem_name
                  << " is not recognized / implemented."
                  << std::endl;
        std::cout << "Exiting..." << std::endl;
        std::exit(EXIT_FAILURE);
    }
} // SolverHydroMuscl::init / 2d

// =======================================================
// =======================================================
template<>
void SolverHydroMuscl<3>::init(DataArray Udata)
{
    /*
     * initialize hydro array at t=0
     */
    if (m_problem_name == "implode")
    {
        InitFunctors<3>::Implode::apply(params, Udata, nbCells);
    }
    else if (m_problem_name == "blast")
    {
        BlastParams blastParams = BlastParams(configMap);
        InitFunctors<3>::Blast::apply(params, blastParams, Udata, nbCells);
    }
    else
    {
        std::cout << "Problem : " << m_problem_name
                  << " is not recognized / implemented."
                  << std::endl;
        std::cout << "Exiting..." << std::endl;
        std::exit(EXIT_FAILURE);
    }
} // SolverHydroMuscl<3>::init

// =======================================================
// =======================================================
// ///////////////////////////////////////////
// Actual computation of Godunov scheme - 2d
// ///////////////////////////////////////////
template<>
void SolverHydroMuscl<2>::godunov_unsplit_impl(DataArray data_in,
                                               DataArray data_out,
                                               real_t dt)
{
    const real_t dtdx = dt / params.dx;
    const real_t dtdy = dt / params.dy;

    // fill ghost cell in data_in
    timers[TIMER_BOUNDARIES]->start();
    make_boundaries(data_in);
    timers[TIMER_BOUNDARIES]->stop();

    // copy data_in into data_out (not necessary)
    // data_out = data_in;
    Kokkos::deep_copy(data_out, data_in);

    // start main computation
    timers[TIMER_NUM_SCHEME]->start();

    // convert conservative variable into primitives ones for the entire domain
    RunFunctors<2>::ConvertToPrimitives::apply(params, data_in, Q, nbCells);

    if (params.implementationVersion == 0)
    {
        RunFunctors<2>::ComputeAndStoreFluxes::apply(params, Q,
                                                     Fluxes_x, Fluxes_y,
                                                     dtdx, dtdy,
                                                     nbCells);

        RunFunctors<2>::Update::apply(params, data_out,
                                      Fluxes_x, Fluxes_y,
                                      nbCells);
    }
    else if (params.implementationVersion == 1)
    {
        // call device functor to compute slopes
        RunFunctors<2>::ComputeSlopes::apply(params, Q, Slopes_x, Slopes_y, nbCells);

        // now trace along X axis
        RunFunctors<2>::ComputeTraceAndFluxes<XDIR>::apply(params, Q,
                                                           Slopes_x, Slopes_y,
                                                           Fluxes_x,
                                                           dtdx, dtdy, nbCells);

        // and update along X axis
        RunFunctors<2>::UpdateDir<XDIR>::apply(params, data_out, Fluxes_x, nbCells);

        // now trace along Y axis
        RunFunctors<2>::ComputeTraceAndFluxes<YDIR>::apply(params, Q,
                                                           Slopes_x, Slopes_y,
                                                           Fluxes_y,
                                                           dtdx, dtdy, nbCells);

        // and update along Y axis
        RunFunctors<2>::UpdateDir<YDIR>::apply(params, data_out, Fluxes_y, nbCells);
    } // end params.implementationVersion == 1

    RunFunctors<2>::ComputeGravityStep::apply(params, data_out, dt, nbCells);

    timers[TIMER_NUM_SCHEME]->stop();
} // SolverHydroMuscl2D::godunov_unsplit_impl

// =======================================================
// =======================================================
// ///////////////////////////////////////////
// Actual computation of Godunov scheme - 3d
// ///////////////////////////////////////////
template<>
void SolverHydroMuscl<3>::godunov_unsplit_impl(DataArray data_in,
                                               DataArray data_out,
                                               real_t dt)
{
    const real_t dtdx = dt / params.dx;
    const real_t dtdy = dt / params.dy;
    const real_t dtdz = dt / params.dz;

    // fill ghost cell in data_in
    timers[TIMER_BOUNDARIES]->start();
    make_boundaries(data_in);
    timers[TIMER_BOUNDARIES]->stop();

    // copy data_in into data_out (not necessary)
    // data_out = data_in;
    Kokkos::deep_copy(data_out, data_in);

    // start main computation
    timers[TIMER_NUM_SCHEME]->start();

    // convert conservative variable into primitives ones for the entire domain
    RunFunctors<3>::ConvertToPrimitives::apply(params, data_in, Q, nbCells);

    if (params.implementationVersion == 0)
    {
        RunFunctors<3>::ComputeAndStoreFluxes::apply(params, Q,
                                                     Fluxes_x, Fluxes_y, Fluxes_z,
                                                     dtdx, dtdy, dtdz,
                                                     nbCells);

        RunFunctors<3>::Update::apply(params, data_out,
                                      Fluxes_x, Fluxes_y, Fluxes_z,
                                      nbCells);
    }
    else if (params.implementationVersion == 1)
    {
        // call device functor to compute slopes
        RunFunctors<3>::ComputeSlopes::apply(params, Q, Slopes_x, Slopes_y, Slopes_z, nbCells);

        // now trace along X axis
        RunFunctors<3>::ComputeTraceAndFluxes<XDIR>::apply(params, Q,
                                                           Slopes_x, Slopes_y, Slopes_z,
                                                           Fluxes_x,
                                                           dtdx, dtdy, dtdz, nbCells);

        // and update along X axis
        RunFunctors<3>::UpdateDir<XDIR>::apply(params, data_out, Fluxes_x, nbCells);

        // now trace along Y axis
        RunFunctors<3>::ComputeTraceAndFluxes<YDIR>::apply(params, Q,
                                                           Slopes_x, Slopes_y, Slopes_z,
                                                           Fluxes_y,
                                                           dtdx, dtdy, dtdz, nbCells);

        // and update along Y axis
        RunFunctors<3>::UpdateDir<YDIR>::apply(params, data_out, Fluxes_y, nbCells);

        // now trace along Z axis
        RunFunctors<3>::ComputeTraceAndFluxes<ZDIR>::apply(params, Q,
                                                           Slopes_x, Slopes_y, Slopes_z,
                                                           Fluxes_z,
                                                           dtdx, dtdy, dtdz, nbCells);

        // and update along Z axis
        RunFunctors<3>::UpdateDir<ZDIR>::apply(params, data_out, Fluxes_z, nbCells);
    } // end params.implementationVersion == 1

    RunFunctors<3>::ComputeGravityStep::apply(params, data_out, dt, nbCells);

    timers[TIMER_NUM_SCHEME]->stop();
} // SolverHydroMuscl<3>::godunov_unsplit_impl

} // namespace muscl

} // namespace euler_kokkos
