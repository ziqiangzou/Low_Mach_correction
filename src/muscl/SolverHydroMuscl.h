/**
 * Class SolverHydroMuscl implementation.
 *
 * Main class for solving hydrodynamics (Euler) with MUSCL-Hancock scheme for 2D/3D.
 */
#ifndef SOLVER_HYDRO_MUSCL_H_
#define SOLVER_HYDRO_MUSCL_H_

#include <string>
#include <cstdio>
#include <cstdbool>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <limits>

// shared
#include "shared/SolverBase.h"
#include "shared/HydroParams.h"
#include "shared/kokkos_shared.h"

// the actual computational functors called in HydroRun
#include "muscl/HydroRunFunctors.h"

// for IO
#include <utils/io/IO_ReadWrite.h>

namespace euler_kokkos { namespace muscl
{

/**
 * Main hydrodynamics data structure for 2D/3D MUSCL-Hancock scheme.
 */
template<int dim>
class SolverHydroMuscl : public euler_kokkos::SolverBase
{
public:
    //! Decide at compile-time which data array to use for 2d or 3d
    using DataArray = typename DataArrays<dim>::DataArray;

    //! Data array typedef for host memory space
    using DataArrayHost = typename DataArrays<dim>::DataArrayHost;

    //! Static creation method called by the solver factory.
    static SolverBase* create(HydroParams& params, ConfigMap& configMap);

    SolverHydroMuscl(HydroParams& params, ConfigMap& configMap);
    virtual ~SolverHydroMuscl();

    // fill boundaries / ghost 2d / 3d
    void make_boundaries(DataArray Udata);

    //! init wrapper (actual initialization)
    void init(DataArray Udata);

    //! compute time step inside an MPI process, at shared memory level.
    real_t compute_dt_local() override;

    //! perform 1 time step (time integration).
    void next_iteration_impl() override;

    //! numerical scheme
    void godunov_unsplit(real_t dt);

    void godunov_unsplit_impl(DataArray data_in,
                              DataArray data_out,
                              real_t dt);

    void computeFluxesAndUpdate(DataArray Udata,
                                real_t dt);

    // output
    void save_solution_impl() override;

    // Public Members
    DataArray     U;     /*!< hydrodynamics conservative variables arrays */
    DataArrayHost Uhost; /*!< U mirror on host memory space */
    DataArray     U2;    /*!< hydrodynamics conservative variables arrays */
    DataArray     Q;     /*!< hydrodynamics primitive    variables array  */

    /* implementation 0 */
    DataArray Fluxes_x; /*!< implementation 0 */
    DataArray Fluxes_y; /*!< implementation 0 */
    DataArray Fluxes_z; /*!< implementation 0 */

    /* implementation 1 only */
    DataArray Slopes_x; /*!< implementation 1 only */
    DataArray Slopes_y; /*!< implementation 1 only */
    DataArray Slopes_z; /*!< implementation 1 only */

    int isize;
    int jsize;
    int ksize;
    int nbCells;
}; // class SolverHydroMuscl

// =======================================================
// ==== CLASS SolverHydroMuscl IMPL ======================
// =======================================================

// =======================================================
// =======================================================
/**
 * Static creation method called by the solver factory.
 */
template<int dim>
SolverBase* SolverHydroMuscl<dim>::create(HydroParams& params, ConfigMap& configMap)
{
    SolverHydroMuscl<dim>* solver = new SolverHydroMuscl<dim>(params, configMap);

    return solver;
}

// =======================================================
// =======================================================
/**
 *
 */
template<int dim>
SolverHydroMuscl<dim>::SolverHydroMuscl(HydroParams& params,
                                        ConfigMap& configMap) :
    SolverBase(params, configMap),
    U(), U2(), Q(),
    Fluxes_x(), Fluxes_y(), Fluxes_z(),
    Slopes_x(), Slopes_y(), Slopes_z(),
    isize(params.isize),
    jsize(params.jsize),
    ksize(params.ksize),
    nbCells(params.isize*params.jsize)
{
    solver_type = SOLVER_MUSCL_HANCOCK;

    if (dim==3)
    {
        nbCells = params.isize*params.jsize*params.ksize;
    }

    m_nCells = nbCells;
    m_nDofsPerCell = 1;

    int nbvar = params.nbvar;

    long long int total_mem_size = 0;

    /*
     * memory allocation (use sizes with ghosts included).
     *
     * Note that Uhost is not just a view to U, Uhost will be used
     * to save data from multiple other device array.
     * That's why we didn't use create_mirror_view to initialize Uhost.
     */
    if (dim==2)
    {
        U     = DataArray("U", isize, jsize, nbvar);
        Uhost = Kokkos::create_mirror(U);
        U2    = DataArray("U2",isize, jsize, nbvar);
        Q     = DataArray("Q", isize, jsize, nbvar);

        total_mem_size += isize*jsize*nbvar * sizeof(real_t) * 3;// 1+1+1 for U+U2+Q

        if (params.implementationVersion == 0)
        {
            Fluxes_x = DataArray("Fluxes_x", isize, jsize, nbvar);
            Fluxes_y = DataArray("Fluxes_y", isize, jsize, nbvar);

            total_mem_size += isize*jsize*nbvar * sizeof(real_t) * 2;// 1+1 for Fluxes_x+Fluxes_y
        }
        else if (params.implementationVersion == 1)
        {
            Slopes_x = DataArray("Slope_x", isize, jsize, nbvar);
            Slopes_y = DataArray("Slope_y", isize, jsize, nbvar);

            // direction splitting (only need one flux array)
            Fluxes_x = DataArray("Fluxes_x", isize, jsize, nbvar);
            Fluxes_y = Fluxes_x;

            total_mem_size += isize*jsize*nbvar * sizeof(real_t) * 3;// 1+1+1 for Slopes_x+Slopes_y+Fluxes_x
        }
    }
    else
    {
        U     = DataArray("U", isize,jsize,ksize, nbvar);
        Uhost = Kokkos::create_mirror(U);
        U2    = DataArray("U2",isize,jsize,ksize, nbvar);
        Q     = DataArray("Q", isize,jsize,ksize, nbvar);

        total_mem_size += isize*jsize*ksize*nbvar*sizeof(real_t)*3;// 1+1+1=3 for U+U2+Q

        if (params.implementationVersion == 0)
        {
            Fluxes_x = DataArray("Fluxes_x", isize,jsize,ksize, nbvar);
            Fluxes_y = DataArray("Fluxes_y", isize,jsize,ksize, nbvar);
            Fluxes_z = DataArray("Fluxes_z", isize,jsize,ksize, nbvar);

            total_mem_size += isize*jsize*ksize*nbvar*sizeof(real_t)*3;// 1+1+1=3 Fluxes
        }
        else if (params.implementationVersion == 1)
        {
            Slopes_x = DataArray("Slope_x", isize,jsize,ksize, nbvar);
            Slopes_y = DataArray("Slope_y", isize,jsize,ksize, nbvar);
            Slopes_z = DataArray("Slope_z", isize,jsize,ksize, nbvar);

            // direction splitting (only need one flux array)
            Fluxes_x = DataArray("Fluxes_x", isize,jsize,ksize, nbvar);
            Fluxes_y = Fluxes_x;
            Fluxes_z = Fluxes_x;

            total_mem_size += isize*jsize*ksize*nbvar*sizeof(real_t)*4;// 1+1+1+1=4 Slopes
        }
    } // dim == 2 / 3

    if (m_restart_run_enabled)
    {
        io::IO_ReadWrite io_reader_writer(params, configMap, m_variables_names);
        io_reader_writer.load_data(U, Uhost, m_iteration, m_tStart);
        m_t = m_tStart;
    }
    else
    {
        // perform init condition
        init(U);
    }

    // initialize boundaries
    make_boundaries(U);

    // copy U into U2
    Kokkos::deep_copy(U2,U);

    // compute initialize time step
    compute_dt();

    int myRank=0;
#ifdef USE_MPI
    myRank = params.myRank;
#endif // USE_MPI

    if (myRank==0)
    {
        std::cout << "##########################" << "\n";
        std::cout << "Solver is " << m_solver_name << "\n";
        std::cout << "Problem (init condition) is " << m_problem_name << "\n";
        std::cout << "##########################" << "\n";

        // print parameters on screen
        params.print();
        std::cout << "##########################" << "\n";
        std::cout << "Memory requested : " << (total_mem_size / 1e6) << " MBytes\n";
        std::cout << "##########################" << "\n";
    }

} // SolverHydroMuscl::SolverHydroMuscl

// =======================================================
// =======================================================
/**
 *
 */
template<int dim>
SolverHydroMuscl<dim>::~SolverHydroMuscl()
{

} // SolverHydroMuscl::~SolverHydroMuscl

// =======================================================
// =======================================================
// //////////////////////////////////////////////////
// Fill ghost cells according to border condition :
// absorbant, reflexive or periodic
// //////////////////////////////////////////////////
template<int dim>
void SolverHydroMuscl<dim>::make_boundaries(DataArray Udata)
{
#ifdef USE_MPI
    make_boundaries_mpi(Udata);
#else
    make_boundaries_serial(Udata);
#endif // USE_MPI
} // SolverHydroMuscl<2>::make_boundaries

// =======================================================
// =======================================================
/**
 * Compute time step satisfying CFL condition.
 *
 * \return dt time step
 */
template<int dim>
real_t SolverHydroMuscl<dim>::compute_dt_local()
{
    real_t invDt = ZERO_F;
    DataArray Udata;

    // which array is the current one ?
    if (m_iteration % 2 == 0)
    {
        Udata = U;
    }
    else
    {
        Udata = U2;
    }

    // call device functor
    RunFunctors<dim>::TimeStep::apply(params, Udata, nbCells, invDt);

    const real_t dt = params.settings.cfl/invDt;

    return dt;
} // SolverHydroMuscl::compute_dt_local

// =======================================================
// =======================================================
template<int dim>
void SolverHydroMuscl<dim>::next_iteration_impl()
{
    int myRank=0;

#ifdef USE_MPI
    myRank = params.myRank;
#endif // USE_MPI

    std::ostringstream oss;
    oss << std::scientific;
    oss << std::setprecision(std::numeric_limits<real_t>::max_digits10);

    if (m_iteration % m_nlog == 0 || (params.enableOutput && should_save_solution()))
    {
        oss << "Step=" << std::setw(std::numeric_limits<int>::digits10) << std::setfill('.') << m_iteration;
        oss << " (dt=" << m_dt << " t=" << m_t << ")\n";
    }

    // output
    if (params.enableOutput && should_save_solution())
    {
        oss << "--> Saving results\n";
        save_solution();
    } // end enable output

    if (myRank == 0)
    {
        std::cout << oss.str();
    }

    // compute new dt
    timers[TIMER_DT]->start();
    compute_dt();
    timers[TIMER_DT]->stop();

    // perform one step integration
    godunov_unsplit(m_dt);
} // SolverHydroMuscl::next_iteration_impl

// =======================================================
// =======================================================
// ///////////////////////////////////////////
// Wrapper to the actual computation routine
// ///////////////////////////////////////////
template<int dim>
void SolverHydroMuscl<dim>::godunov_unsplit(real_t dt)
{
    if (m_iteration % 2 == 0)
    {
        godunov_unsplit_impl(U , U2, dt);
    }
    else
    {
        godunov_unsplit_impl(U2, U , dt);
    }
} // SolverHydroMuscl::godunov_unsplit

// =======================================================
// =======================================================
template<int dim>
void SolverHydroMuscl<dim>::save_solution_impl()
{
    timers[TIMER_IO]->start();
    if (m_iteration % 2 == 0)
    {
        save_data(U,  Uhost, m_times_saved, m_t);
    }
    else
    {
        save_data(U2, Uhost, m_times_saved, m_t);
    }
    timers[TIMER_IO]->stop();
} // SolverHydroMuscl::save_solution_impl()

} // namespace muscl

} // namespace euler_kokkos

#endif // SOLVER_HYDRO_MUSCL_H_
