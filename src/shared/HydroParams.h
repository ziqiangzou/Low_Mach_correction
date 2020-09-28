/**
 * \file HydroParams.h
 * \brief Hydrodynamics solver parameters.
 *
 * \date April, 16 2016
 * \author P. Kestener
 */
#ifndef HYDRO_PARAMS_H_
#define HYDRO_PARAMS_H_

#include "shared/kokkos_shared.h"
#include "shared/real_type.h"
#include "utils/config/ConfigMap.h"

#include <vector>
#include <stdbool.h>
#include <string>

#include "shared/enums.h"

#ifdef USE_MPI
#include "utils/mpiUtils/MpiCommCart.h"
#endif // USE_MPI

struct HydroSettings
{
    // hydro (numerical scheme) parameters
    real_t cfl;         /*!< Courant-Friedrich-Lewy parameter.*/
    real_t slope_type;  /*!< type of slope computation (2 for second order scheme).*/
    int    iorder;      /*!< */
    real_t smallr;      /*!< small density cut-off*/
    real_t smallc;      /*!< small speed of sound cut-off*/
    real_t smallp;      /*!< small pressure cut-off*/
    real_t smallpp;     /*!< smallp times smallr*/
    real_t K;
    std::string problem_name;
    bool   cut_cell;
    bool   low_mach_correction;
    bool   normal_direction;
    int    correction_type;
    bool   redistance;
    int    redistance_frequence;
    bool   conservative;

    real_t gamma0;      /*!< specific heat capacity ratio (adiabatic index)*/
    real_t gamma1;      /*!< specific heat capacity ratio (adiabatic index)*/
    bool barotropic0;      /*!< specific heat capacity ratio (adiabatic index)*/
    bool barotropic1;      /*!< specific heat capacity ratio (adiabatic index)*/
    real_t sound_speed0;      /*!< specific heat capacity ratio (adiabatic index)*/
    real_t sound_speed1;      /*!< specific heat capacity ratio (adiabatic index)*/
    real_t rho0;      /*!< specific heat capacity ratio (adiabatic index)*/
    real_t rho1;      /*!< specific heat capacity ratio (adiabatic index)*/
    real_t Tsat;      /*!< saturation temperature >*/
    real_t latent_heat;      /*!<laten heat >*/
    real_t mmw;         /*!< mean molecular weight*/
    real_t mu0;          /*!< dynamic viscosity */
    real_t mu1;          /*!< dynamic viscosity */
    real_t sigma;          /*!< surface tension */
    real_t kappa0;       /*!< thermal conductivity */
    real_t kappa1;       /*!< thermal conductivity */
    real_t gamma6;
    real_t Rstar0;       /*!< specific gas constant */
    real_t Rstar1;       /*!< specific gas constant */
    real_t cp0;          /*!< specific heat (constant pressure) */
    real_t cv0;          /*!< specific heat (constant volume) */
    real_t cp1;          /*!< specific heat (constant pressure) */
    real_t cv1;          /*!< specific heat (constant volume) */

    real_t g_x;         /*!< gravity in x-direction */
    real_t g_y;         /*!< gravity in y-direction */
    real_t g_z;         /*!< gravity in z-direction */

    KOKKOS_INLINE_FUNCTION
    HydroSettings() : cfl(-1.0), slope_type(-1.0), iorder(-1),
                      smallr(1e-8), smallc(1e-8), smallp(1e-6), smallpp(1e-6),
                      K(-1.0), problem_name("unknown"), cut_cell(false), low_mach_correction(false),normal_direction(false), correction_type(1), redistance(false), redistance_frequence(1),conservative(false),
                      gamma0(-1.0),gamma1(-1.0), barotropic0(false), barotropic1(false), sound_speed0(0.0), sound_speed1(0.0), rho0(0.0), rho1(0.0), Tsat(0.0), latent_heat(0.0), 
		       mmw(-1.0), mu0(-1.0),mu1(-1.0),sigma(-0.01), kappa0(-1.0),kappa1(-1.0),
                      gamma6(-1.0), Rstar0(-1.0),Rstar1(-1.0), cp0(-1.0), cv0(-1.0),cp1(-1.0), cv1(-1.0),
                      g_x(0.0), g_y(0.0), g_z(0.0) {}
}; // struct HydroSettings

/**
 * Hydro Parameters (declaration).
 */
struct HydroParams
{
#ifdef USE_MPI
    using MpiCommCart = hydroSimu::MpiCommCart;
#endif // USE_MPI

    // run parameters
    std::string solver_name;
    std::string LS_solver_name;
    int    nStepmax;   /*!< maximun number of time steps. */
    real_t tEnd;       /*!< end of simulation time. */
    int    nOutput;    /*!< number of time steps between 2 consecutive outputs. */
    bool   enableOutput; /*!< enable output file write. */
    int    nlog;      /*!<  number of time steps between 2 consecutive logs. */

    // geometry parameters
    int nx;     /*!< logical size along X (without ghost cells).*/
    int ny;     /*!< logical size along Y (without ghost cells).*/
    int nz;     /*!< logical size along Z (without ghost cells).*/
    int ghostWidth;
    int nbvar;  /*!< number of variables in HydroState */
    DimensionType dimType; //!< 2D or 3D.

    int imin;   /*!< index minimum at X border*/
    int imax;   /*!< index maximum at X border*/
    int jmin;   /*!< index minimum at Y border*/
    int jmax;   /*!< index maximum at Y border*/
    int kmin;   /*!< index minimum at Z border*/
    int kmax;   /*!< index maximum at Z border*/

    int isize;  /*!< total size (in cell unit) along X direction with ghosts.*/
    int jsize;  /*!< total size (in cell unit) along Y direction with ghosts.*/
    int ksize;  /*!< total size (in cell unit) along Z direction with ghosts.*/
    int ijSize;


    real_t xmin; /*!< domain bound */
    real_t xmax; /*!< domain bound */
    real_t ymin; /*!< domain bound */
    real_t ymax; /*!< domain bound */
    real_t zmin; /*!< domain bound */
    real_t zmax; /*!< domain bound */
    real_t dx;   /*!< x resolution */
    real_t dy;   /*!< y resolution */
    real_t dz;   /*!< z resolution */

    real_t onesurdx;
    real_t onesurdy;

    BoundaryConditionType boundary_type_xmin; /*!< boundary condition */
    BoundaryConditionType boundary_type_xmax; /*!< boundary condition */
    BoundaryConditionType boundary_type_ymin; /*!< boundary condition */
    BoundaryConditionType boundary_type_ymax; /*!< boundary condition */
    BoundaryConditionType boundary_type_zmin; /*!< boundary condition */
    BoundaryConditionType boundary_type_zmax; /*!< boundary condition */

    real_t Astate0;     
    real_t Astate1;     
    real_t Bstate0;     
    real_t Bstate1;     

    real_t f0;
    real_t f1;
    real_t g0;
    real_t g1;

    // IO parameters
    bool ioVTK;   /*!< enable VTK  output file format (using VTI).*/
    bool ioHDF5;  /*!< enable HDF5 output file format.*/

    //! hydro settings (gamma0, ...) to be passed to Kokkos device functions
    HydroSettings settings;

    bool useAllRegimeTimeSteps;

    int niter_riemann;  /*!< number of iteration usd in quasi-exact riemann solver*/
    int riemannSolverType;

    // other parameters
    int implementationVersion=0; /*!< triggers which implementation to use (currently 3 versions)*/

#ifdef USE_MPI
    //! runtime determination if we are using float ou double (for MPI communication)
    //! initialized in constructor to either MpiComm::FLOAT or MpiComm::DOUBLE
    int data_type;

    //! size of the MPI cartesian grid
    int mx,my,mz;

    //! MPI communicator in a cartesian virtual topology
    MpiCommCart *communicator;

    //! number of dimension
    int nDim;

    //! MPI rank of current process
    int myRank;

    //! number of MPI processes
    int nProcs;

    //! MPI cartesian coordinates inside MPI topology
    Kokkos::Array<int,3> myMpiPos;

    //! number of MPI process neighbors (4 in 2D and 6 in 3D)
    int nNeighbors;

    //! MPI rank of adjacent MPI processes
    Kokkos::Array<int,6> neighborsRank;

    //! boundary condition type with adjacent domains (corresponding to
    //! neighbor MPI processes)
    Kokkos::Array<BoundaryConditionType,6> neighborsBC;

#endif // USE_MPI

    HydroParams() :
         solver_name("unknown"),LS_solver_name("unknown"),
        nStepmax(0), tEnd(0.0), nOutput(0), enableOutput(true),
        nlog(10),
        nx(0), ny(0), nz(0), ghostWidth(2), nbvar(4), dimType(TWO_D),
        imin(0), imax(0), jmin(0), jmax(0), kmin(0), kmax(0),
        isize(0), jsize(0), ksize(0), ijSize(0),
        xmin(0.0), xmax(1.0), ymin(0.0), ymax(1.0), zmin(0.0), zmax(1.0),
        dx(0.0), dy(0.0), dz(0.0),
        boundary_type_xmin(BC_UNDEFINED),
        boundary_type_xmax(BC_UNDEFINED),
        boundary_type_ymin(BC_UNDEFINED),
        boundary_type_ymax(BC_UNDEFINED),
        boundary_type_zmin(BC_UNDEFINED),
        boundary_type_zmax(BC_UNDEFINED),
        Astate0(-1.0), Astate1(-1.0), Bstate0(-1.0), Bstate1(-1.0),
        f0(-1.0), f1(-1.0), g0(-1.0), g1(-1.0),
        ioVTK(true), ioHDF5(false),
        settings(),
        niter_riemann(10), riemannSolverType(),
        implementationVersion(0)
#ifdef USE_MPI
        // init MPI-specific parameters...
#endif // USE_MPI
    {}

    virtual ~HydroParams() {}

    //! This is the genuine initialiation / setup (fed by parameter file)
    virtual void setup(ConfigMap& map);

#ifdef USE_MPI
    //! Initialize MPI-specific parameters
    void setup_mpi(ConfigMap& map);
#endif // USE_MPI

    void init();
    void print();
}; // struct HydroParams

#endif // HYDRO_PARAMS_H_
