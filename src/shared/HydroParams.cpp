#include "HydroParams.h"

#include "euler_kokkos.h"
#include "shared/units.h"

#include <cstdlib> // for exit
#include <cstdio>  // for fprintf
#include <cstring> // for strcmp
#include <iostream>

#include "config/inih/ini.h" // our INI file reader
#include "HydroState.h"

#ifdef USE_MPI
using namespace hydroSimu;
#endif // USE_MPI

// =======================================================
// =======================================================
/*
 * Hydro Parameters (read parameter file)
 */
void HydroParams::setup(ConfigMap &configMap)
{
	/* initialize RUN parameters */
	nStepmax = configMap.getInteger("run","nstepmax",1000);
	tEnd     = configMap.getFloat  ("run","tend",0.0);
	nOutput  = configMap.getInteger("run","noutput",100);
	if (nOutput == 0)
	{
		enableOutput = false;
	}

	nlog        = configMap.getInteger("run","nlog",10);
	solver_name = configMap.getString("run", "solver_name", "unknown");
	LS_solver_name = configMap.getString("run", "LS_solver_name", "unknown");

	if (solver_name == "Hydro_Muscl_2D")
	{
		dimType = TWO_D;
		nbvar = HYDRO_2D_NBVAR;
		ghostWidth = 3;
	}
	else if (solver_name == "Hydro_Muscl_3D")
	{
		dimType = THREE_D;
		nbvar = HYDRO_3D_NBVAR;
		ghostWidth = 3;
	}
	else if (solver_name == "Hydro_All_Regime_2D")
	{
		dimType = TWO_D;
		nbvar = HYDRO_2D_NBVAR;
		ghostWidth = 3;
	}
	else if (solver_name == "Hydro_All_Regime_3D")
	{
		dimType = THREE_D;
		nbvar = HYDRO_3D_NBVAR;
		ghostWidth = 3;
	}
	else
	{
		euler_kokkos::abort("Solver \""+solver_name+"\" is not valid");
	}

	/* initialize MESH parameters */
	nx = configMap.getInteger("mesh","nx", 1);
	ny = configMap.getInteger("mesh","ny", 1);
	nz = configMap.getInteger("mesh","nz", 1);

	xmin = configMap.getFloat("mesh", "xmin", 0.0);
	ymin = configMap.getFloat("mesh", "ymin", 0.0);
	zmin = configMap.getFloat("mesh", "zmin", 0.0);

	xmax = configMap.getFloat("mesh", "xmax", 1.0);
	ymax = configMap.getFloat("mesh", "ymax", 1.0);
	zmax = configMap.getFloat("mesh", "zmax", 1.0);

	boundary_type_xmin  = static_cast<BoundaryConditionType>(configMap.getInteger("mesh","boundary_type_xmin", BC_DIRICHLET));
	boundary_type_xmax  = static_cast<BoundaryConditionType>(configMap.getInteger("mesh","boundary_type_xmax", BC_DIRICHLET));
	boundary_type_ymin  = static_cast<BoundaryConditionType>(configMap.getInteger("mesh","boundary_type_ymin", BC_DIRICHLET));
	boundary_type_ymax  = static_cast<BoundaryConditionType>(configMap.getInteger("mesh","boundary_type_ymax", BC_DIRICHLET));
	boundary_type_zmin  = static_cast<BoundaryConditionType>(configMap.getInteger("mesh","boundary_type_zmin", BC_DIRICHLET));
	boundary_type_zmax  = static_cast<BoundaryConditionType>(configMap.getInteger("mesh","boundary_type_zmax", BC_DIRICHLET));

	settings.cfl            = configMap.getFloat("hydro", "cfl", 0.5);
	settings.iorder         = configMap.getInteger("hydro","iorder", 2);
	settings.slope_type     = configMap.getFloat("hydro","slope_type",1.0);
	settings.smallc         = configMap.getFloat("hydro","smallc", 1e-10);
	settings.smallr         = configMap.getFloat("hydro","smallr", 1e-10);

	settings.K                   = configMap.getFloat("hydro", "K", 1.1);
	settings.problem_name        = configMap.getString("hydro", "problem", "unknown");
	settings.cut_cell            = configMap.getBool("hydro", "cut_cell", false);
	settings.low_mach_correction = configMap.getBool("hydro", "low_mach_correction", true);
	settings.normal_direction    = configMap.getBool("hydro", "normal_direction", false);
	settings.correction_type     = configMap.getInteger("hydro", "correction_type", 1);
	settings.redistance          = configMap.getBool("hydro", "redistance", false);
	settings.redistance_frequence= configMap.getInteger("hydro", "redistance_frequence", 200);
	settings.conservative        = configMap.getBool("hydro", "conservative", true);

	settings.gamma0         = configMap.getFloat("hydro","gamma0", 1.4);
	settings.gamma1         = configMap.getFloat("hydro","gamma1", 1.4);
	settings.barotropic0 = configMap.getBool("hydro", "barotropic0", false);
	settings.barotropic1 = configMap.getBool("hydro", "barotropic1", false);
	settings.sound_speed0         = configMap.getFloat("hydro","sound_speed0", 0.);
	settings.sound_speed1         = configMap.getFloat("hydro","sound_speed1", 0.);
	settings.rho0         = configMap.getFloat("hydro","rho0", 0.);
	settings.rho1         = configMap.getFloat("hydro","rho1", 0.);
	settings.Tsat         = configMap.getFloat("hydro","Tsat", 10000.);
	settings.latent_heat   = configMap.getFloat("hydro","latent_heat", 100000.);
	Astate0                 = configMap.getFloat("hydro","Astate0", 1000.);
	Astate1                 = configMap.getFloat("hydro","Astate1", 1000.);
	Bstate0                 = configMap.getFloat("hydro","Bstate0", 1000.);
	Bstate1                 = configMap.getFloat("hydro","Bstate1", 1000.);
	settings.mmw            = configMap.getFloat("hydro", "mmw", 1.0);
	settings.mu0            = configMap.getFloat("hydro", "mu0", 0.0);
	settings.mu1            = configMap.getFloat("hydro", "mu1", 0.0);
	settings.sigma          = configMap.getFloat("hydro", "sigma", 0.0);
	settings.Rstar0          = configMap.getFloat("hydro", "Rstar0", 0.0);
	settings.Rstar1          = configMap.getFloat("hydro", "Rstar1", 0.0);
	settings.kappa0          = configMap.getFloat("hydro", "kappa0", 0.0);
	settings.kappa1          = configMap.getFloat("hydro", "kappa1", 0.0);

	settings.g_x            = configMap.getFloat("gravity", "g_x", 0.0);
	settings.g_y            = configMap.getFloat("gravity", "g_y", 0.0);
	settings.g_z            = configMap.getFloat("gravity", "g_z", 0.0);


	niter_riemann  = configMap.getInteger("hydro","niter_riemann", 10);
	std::string riemannSolverStr(configMap.getString("hydro","riemann", "approx"));
	if (riemannSolverStr == "approx")
	{
		riemannSolverType = RIEMANN_APPROX;
	}
	else if (riemannSolverStr == "llf")
	{
		riemannSolverType = RIEMANN_LLF;
	}
	else if (riemannSolverStr == "hll")
	{
		riemannSolverType = RIEMANN_HLL;
	}
	else if (riemannSolverStr == "hllc")
	{
		riemannSolverType = RIEMANN_HLLC;
	}
	else
	{
		euler_kokkos::abort("Riemann Solver \""+riemannSolverStr+"\" is invalid");
	}

	useAllRegimeTimeSteps = configMap.getBool("hydro", "useAllRegimeTimeSteps", false);

	implementationVersion = configMap.getInteger("other","implementationVersion", 0);
	if (implementationVersion != 0 && implementationVersion != 1)
	{
		euler_kokkos::abort("Implementation version is invalid (must be 0 or 1)\n"
				"Check your parameter file, section \"other\"");
	}

	init();

#ifdef USE_MPI
	setup_mpi(configMap);
#endif // USE_MPI
} // HydroParams::setup

#ifdef USE_MPI
// =======================================================
// =======================================================
void HydroParams::setup_mpi(ConfigMap& configMap)
{
	// runtime determination if we are using float ou double (for MPI communication)
	data_type = std::is_same<real_t, float>::value ?
		hydroSimu::MpiComm::FLOAT : hydroSimu::MpiComm::DOUBLE;

	// MPI parameters :
	mx = configMap.getInteger("mpi", "mx", 1);
	my = configMap.getInteger("mpi", "my", 1);
	mz = configMap.getInteger("mpi", "mz", 1);

	// check that parameters are consistent
	bool error = false;
	error |= (mx < 1);
	error |= (my < 1);
	error |= (mz < 1);

	// get world communicator size and check it is consistent with mesh grid sizes
	nProcs = MpiComm::world().getNProc();
	if (nProcs != mx*my*mz)
	{
		euler_kokkos::abort("Inconsistent MPI cartesian virtual topology geometry\n"
				"mx*my*mz must match with parameter given to mpirun !!!");
	}

	// create the MPI communicator for our cartesian mesh
	if (dimType == TWO_D)
	{
		communicator = new MpiCommCart(mx, my, MPI_CART_PERIODIC_TRUE, MPI_REORDER_TRUE);
		nDim = 2;
	}
	else
	{
		communicator = new MpiCommCart(mx, my, mz, MPI_CART_PERIODIC_TRUE, MPI_REORDER_TRUE);
		nDim = 3;
	}

	// get my MPI rank inside topology
	myRank = communicator->getRank();

	// get my coordinates inside topology
	// myMpiPos[0] is between 0 and mx-1
	// myMpiPos[1] is between 0 and my-1
	// myMpiPos[2] is between 0 and mz-1
	communicator->getMyCoords(myMpiPos.data());

	/*
	 * compute MPI ranks of our neighbors and
	 * set default boundary condition types
	 */
	if (dimType == TWO_D)
	{
		nNeighbors = N_NEIGHBORS_2D;
		neighborsRank[X_MIN] = communicator->getNeighborRank<X_MIN>();
		neighborsRank[X_MAX] = communicator->getNeighborRank<X_MAX>();
		neighborsRank[Y_MIN] = communicator->getNeighborRank<Y_MIN>();
		neighborsRank[Y_MAX] = communicator->getNeighborRank<Y_MAX>();
		neighborsRank[Z_MIN] = 0;
		neighborsRank[Z_MAX] = 0;

		neighborsBC[X_MIN] = BC_COPY;
		neighborsBC[X_MAX] = BC_COPY;
		neighborsBC[Y_MIN] = BC_COPY;
		neighborsBC[Y_MAX] = BC_COPY;
		neighborsBC[Z_MIN] = BC_UNDEFINED;
		neighborsBC[Z_MAX] = BC_UNDEFINED;
	}
	else
	{
		nNeighbors = N_NEIGHBORS_3D;
		neighborsRank[X_MIN] = communicator->getNeighborRank<X_MIN>();
		neighborsRank[X_MAX] = communicator->getNeighborRank<X_MAX>();
		neighborsRank[Y_MIN] = communicator->getNeighborRank<Y_MIN>();
		neighborsRank[Y_MAX] = communicator->getNeighborRank<Y_MAX>();
		neighborsRank[Z_MIN] = communicator->getNeighborRank<Z_MIN>();
		neighborsRank[Z_MAX] = communicator->getNeighborRank<Z_MAX>();

		neighborsBC[X_MIN] = BC_COPY;
		neighborsBC[X_MAX] = BC_COPY;
		neighborsBC[Y_MIN] = BC_COPY;
		neighborsBC[Y_MAX] = BC_COPY;
		neighborsBC[Z_MIN] = BC_COPY;
		neighborsBC[Z_MAX] = BC_COPY;
	}

	/*
	 * identify outside boundaries (no actual communication if we are
	 * doing BC_DIRICHLET or BC_NEUMANN)
	 *
	 * Please notice the duality
	 * XMIN -- boundary_xmax
	 * XMAX -- boundary_xmin
	 *
	 */

	// X_MIN boundary
	if (myMpiPos[DIR_X] == 0)
	{
		neighborsBC[X_MIN] = boundary_type_xmin;
	}

	// X_MAX boundary
	if (myMpiPos[DIR_X] == mx-1)
	{
		neighborsBC[X_MAX] = boundary_type_xmax;
	}

	// Y_MIN boundary
	if (myMpiPos[DIR_Y] == 0)
	{
		neighborsBC[Y_MIN] = boundary_type_ymin;
	}

	// Y_MAX boundary
	if (myMpiPos[DIR_Y] == my-1)
	{
		neighborsBC[Y_MAX] = boundary_type_ymax;
	}

	if (dimType == THREE_D)
	{
		// Z_MIN boundary
		if (myMpiPos[DIR_Z] == 0)
		{
			neighborsBC[Z_MIN] = boundary_type_zmin;
		}

		// Z_MAX boundary
		if (myMpiPos[DIR_Z] == mz-1)
		{
			neighborsBC[Z_MAX] = boundary_type_zmax;
		}
	} // end THREE_D

	// fix space resolution :
	// need to take into account number of MPI process in each direction
	dx = (xmax - xmin)/(nx*mx);
	dy = (ymax - ymin)/(ny*my);
	dz = (zmax - zmin)/(nz*mz);

	// print information about current setup
	if (myRank == 0)
	{
		std::cout << "We are about to start simulation with the following characteristics\n";
		std::cout << "Global resolution : "
			<< nx*mx << " x " << ny*my << " x " << nz*mz << "\n";
		std::cout << "Local  resolution : "
			<< nx << " x " << ny << " x " << nz << "\n";
		std::cout << "MPI Cartesian topology : " << mx << "x" << my << "x" << mz << std::endl;
	}
} // HydroParams::setup_mpi

#endif // USE_MPI

// =======================================================
// =======================================================
void HydroParams::init()
{
	// set other parameters
	imin = 0;
	jmin = 0;
	kmin = 0;

	imax = nx - 1 + 2*ghostWidth;
	jmax = ny - 1 + 2*ghostWidth;
	kmax = nz - 1 + 2*ghostWidth;

	isize = imax - imin + 1;
	jsize = jmax - jmin + 1;
	ksize = kmax - kmin + 1;

	f0=ONE_F/(settings.gamma0-ONE_F);
	f1=ONE_F/(settings.gamma1-ONE_F);

	g0=settings.gamma0*(Bstate0-Astate0)/(settings.gamma0-ONE_F);
	g1=settings.gamma1*(Bstate1-Astate1)/(settings.gamma1-ONE_F);

	ijSize=isize*jsize;

	dx = (xmax - xmin) / nx;
	dy = (ymax - ymin) / ny;
	dz = (zmax - zmin) / nz;

	onesurdx=nx/(xmax-xmin);
	onesurdy=ny/(ymax-ymin);


	settings.smallp  = settings.smallc*settings.smallc/settings.gamma0;
	settings.smallpp = settings.smallr*settings.smallp;
	settings.gamma6  = (settings.gamma0 + ONE_F)/(TWO_F * settings.gamma0);
	//settings.Rstar   = euler_kokkos::code_units::constants::Rstar_h / settings.mmw;
	settings.cp0      = settings.gamma0 / (settings.gamma0-ONE_F) * settings.Rstar0;
	settings.cv0      = ONE_F / (settings.gamma0-ONE_F) * settings.Rstar0;
	settings.cp1      = settings.gamma1 / (settings.gamma1-ONE_F) * settings.Rstar1;
	settings.cv1      = ONE_F / (settings.gamma1-ONE_F) * settings.Rstar1;
} // HydroParams::init


// =======================================================
// =======================================================
void HydroParams::print()
{
	printf( "##########################\n");
	printf( "Simulation run parameters:\n");
	printf( "##########################\n");
	if (solver_name=="Hydro_Muscl_2D" || solver_name=="Hydro_Muscl_3D")
	{
		printf( "riemann       : %d\n", riemannSolverType);
		printf( "niter_riemann : %d\n", niter_riemann);
		printf( "iorder        : %d\n", settings.iorder);
		printf( "slope_type    : %f\n", settings.slope_type);
		printf( "implementation version : %d\n",implementationVersion);
	}
	if (solver_name=="Hydro_All_Regime_2D" || solver_name=="Hydro_All_Regime_3D")
	{
		printf( "low Mach correction: %s\n", settings.low_mach_correction ? "Enabled" : "Disabled");
		printf( "correction_type: %d\n", settings.correction_type);
		//printf( "conservative scheme: %s\n", settings.conservative ? "Enabled" : "Disabled");
	}
	printf( "nx         : %d\n", nx);
	printf( "ny         : %d\n", ny);
	printf( "nz         : %d\n", nz);

	printf( "dx         : %f\n", dx);
	printf( "dy         : %f\n", dy);
	printf( "dz         : %f\n", dz);

	printf( "imin       : %d\n", imin);
	printf( "imax       : %d\n", imax);

	printf( "jmin       : %d\n", jmin);
	printf( "jmax       : %d\n", jmax);

	printf( "kmin       : %d\n", kmin);
	printf( "kmax       : %d\n", kmax);

	printf( "ghostWidth : %d\n", ghostWidth);
	printf( "nbvar      : %d\n", nbvar);
	printf( "nStepmax   : %d\n", nStepmax);
	printf( "tEnd       : %f\n", tEnd);
	printf( "nOutput    : %d\n", nOutput);
	printf( "cfl        : %f\n", settings.cfl);
	printf( "smallr     : %12.10f\n", settings.smallr);
	printf( "smallc     : %12.10f\n", settings.smallc);
	printf( "smallp     : %12.10f\n", settings.smallp);
	printf( "smallpp    : %g\n", settings.smallpp);
	printf( "gamma0     : %f\n", settings.gamma0);
	printf( "gamma6     : %f\n", settings.gamma6);
	printf( "Rstar0 (specific gas constant) : %g\n", settings.Rstar0);
	printf( "Rstar1 (specific gas constant) : %g\n", settings.Rstar1);


	printf( "low Mach correction: %s\n", settings.low_mach_correction ? "Enabled" : "Disabled");

	if(settings.low_mach_correction)
	{
		if (settings.correction_type==1)
			printf( "correction type: %s\n", "Acoustic_Impedance_Weighted");
		else if (settings.correction_type==2)
			printf( "correction type: %s\n", "Pressure_Centering");
		else if (settings.correction_type==3)
			printf( "correction type: %s\n", "New_correction");
	}

	printf( "redistance: %s\n", settings.redistance ? "Enabled" : "Disabled");
	if(settings.redistance)
		printf( "redistance frequence   : %d\n", settings.redistance_frequence);

	printf( "cp0 (specific heat)            : %f\n", settings.cp0);
	printf( "cv0 (specific heat)            : %f\n", settings.cv0);
	printf( "cp1 (specific heat)            : %f\n", settings.cp1);
	printf( "cv1 (specific heat)            : %f\n", settings.cv1);
	printf( "mu0 (dynamic visosity)         : %f\n", settings.mu0);
	printf( "mu1 (dynamic visosity)         : %f\n", settings.mu1);
	printf( "Latent heat                    : %f\n", settings.latent_heat);
	printf( "Tsat                           : %f\n", settings.Tsat);
	printf( "sigma (surface tension)         : %f\n", settings.sigma);
	printf( "kappa0 (thermal diffusivity)   : %f\n", settings.kappa0);
	printf( "kappa1 (thermal diffusivity)   : %f\n", settings.kappa1);
	printf( "g_x        : %f\n", settings.g_x);
	printf( "g_y        : %f\n", settings.g_y);
	printf( "g_z        : %f\n", settings.g_z);
	printf( "##########################\n");
} // HydroParams::print
