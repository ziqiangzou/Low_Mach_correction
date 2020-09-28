#include "euler_kokkos.h"

#include "shared/kokkos_shared.h"
#include "euler_kokkos_version.h"

#ifdef USE_MPI
#include "utils/mpiUtils/GlobalMpiSession.h"
#endif // USE_MPI

#ifdef USE_FPE_DEBUG
// for catching floating point errors
#include <fenv.h>
#include <signal.h>
#endif // USE_FPE_DEBUG

namespace euler_kokkos
{

namespace
{

#ifdef USE_FPE_DEBUG
// signal handler for catching floating point errors
void fpehandler(int sig_num)
{
    signal(SIGFPE, fpehandler);
    printf("SIGFPE: floating point exception occured of type %d, exiting.\n",sig_num);
    abort();
}
#endif // USE_FPE_DEBUG

}

void initialize(int& argc, char**& argv)
{
    Kokkos::initialize(argc, argv);

#ifdef USE_FPE_DEBUG
    /*
     * Install a signal handler for floating point errors.
     * This only usefull when debugging, doing a backtrace in gdb,
     * tracking for NaN
     */
    feenableexcept(FE_DIVBYZERO | FE_INVALID);
    signal(SIGFPE, fpehandler);
#endif // USE_FPE_DEBUG
}

void print_kokkos_configuration()
{
#ifdef USE_MPI
    const int rank = hydroSimu::GlobalMpiSession::getRank();
#else
    const int rank = 0;
#endif // USE_MPI

    if (rank==0)
    {
        std::cout << "##########################\n";
        std::cout << "Kokkos configuration      \n";
        std::cout << "##########################\n";

        std::ostringstream msg;
        if (Kokkos::hwloc::available())
        {
            msg << "hwloc( NUMA[" << Kokkos::hwloc::get_available_numa_count()
                << "] x CORE["    << Kokkos::hwloc::get_available_cores_per_numa()
                << "] x HT["      << Kokkos::hwloc::get_available_threads_per_core()
                << "] )"
                << std::endl ;
        }
        Kokkos::print_configuration(msg);
        std::cout << msg.str();
        std::cout << "##########################\n";
    }

#if defined(USE_MPI) && defined(KOKKOS_ENABLE_CUDA)
    if (rank==0)
    {
        std::cout << "##########################\n";
        std::cout << "GPU dispatching\n";
        std::cout << "##########################\n";
    }
    hydroSimu::GlobalMpiSession::synchronize();
    const int nRanks = hydroSimu::GlobalMpiSession::getNProc();
    int cudaDeviceId = -1;
    cudaGetDevice(&cudaDeviceId);
    std::ostringstream msg;
    msg << "I'm MPI task #" << rank << " out of " << nRanks
        << " in MPI_COMM_WORLD,"
        << " pinned to GPU #" << cudaDeviceId << std::endl;
    std::cout << msg.str();
    hydroSimu::GlobalMpiSession::synchronize();
    if (rank==0)
    {
        std::cout << "##########################\n";
    }
#endif // USE_MPI && KOKKOS_ENABLE_CUDA

    if (rank==0)
    {
        std::cout << "##########################\n";
        std::cout << "Git information\n";
        std::cout << "##########################\n";
        std::cout << "Branch " << version::git_branch
                  << ", commit " << version::git_build_string << '\n';
        std::cout << "##########################\n";
    }
}

void finalize()
{
    Kokkos::finalize();
}

void abort(const std::string& msg)
{
#ifdef USE_MPI
    const int rank = hydroSimu::GlobalMpiSession::getRank();
#else
    const int rank = 0;
#endif // USE_MPI
    if (rank == 0)
    {
        std::cerr << "Error: " << msg << std::endl;
    }
    std::exit(EXIT_FAILURE);
}

}
