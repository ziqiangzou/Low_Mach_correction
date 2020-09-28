/**
 * Hydro solver (Muscl-Hancock).
 *
 * \date April, 16 2016
 * \author P. Kestener
 */

#include <cstdlib>
#include <cstdio>
#include <cerrno>
#include "euler_kokkos.h"

#include "shared/real_type.h"    // choose between single and double precision
#include "shared/HydroParams.h"  // read parameter file
#include "shared/solver_utils.h" // print monitoring information

// solver
#include "shared/SolverFactory.h"

#ifdef USE_MPI
#include "utils/mpiUtils/GlobalMpiSession.h"
#endif // USE_MPI

// ===============================================================
// ===============================================================
// ===============================================================
int main(int argc, char* argv[])
{

    namespace ek = euler_kokkos;

    // Create MPI session if MPI enabled
#ifdef USE_MPI
    hydroSimu::GlobalMpiSession mpiSession(&argc, &argv);
    const int rank = hydroSimu::GlobalMpiSession::getRank();
#else
    const int rank = 0;
#endif // USE_MPI

    ek::initialize(argc, argv);

    std::atexit(ek::finalize);

    if (argc < 2)
    {
        ek::abort("Wrong number of arguments");
    }

    ek::print_kokkos_configuration();

    {
        // read parameter file and initialize parameter
        // parse parameters from input file
        const std::string filename(argv[1]);
        ConfigMap configMap(filename);
        if (configMap.ParseError() < 0)
        {
            ek::abort("Something went wrong when parsing file \""+filename+'"');
        }

        // test: create a HydroParams object
        HydroParams params = HydroParams();
        params.setup(configMap);

        // retrieve solver name from settings
        const std::string solver_name = configMap.getString("run", "solver_name", "Unknown");

        // initialize workspace memory (U, U2, ...)
        ek::SolverBase* solver = ek::SolverFactory::Instance().create(solver_name,
                                                                      params,
                                                                      configMap);

        // start computation
        if (rank == 0) std::cout << "Starting computation...\n";
        solver->timers[TIMER_TOTAL]->start();

        // save initialization
        if (params.nOutput != 0)
        {
            solver->save_solution();
        }

        // Hydrodynamics solver loop
        while (!solver->finished())
        {
            solver->next_iteration();
        } // end solver loop

        // save last time step
        if (params.nOutput != 0)
        {
            solver->save_solution();
        }
        if (rank == 0) std::cout << "final time is " << solver->m_t << std::endl;

        // end of computation
        solver->timers[TIMER_TOTAL]->stop();
        if (rank == 0) std::cout << "...end of computation\n";

        ek::print_solver_monitoring_info(solver);

        delete solver;
    }

    return EXIT_SUCCESS;

} // end main
