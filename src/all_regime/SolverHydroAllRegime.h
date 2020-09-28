#pragma once

#include <string>
#include <cstdio>
#include <cstdbool>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <type_traits>
#include <iostream>
#include <iomanip>
#include <limits>

// shared
#include "shared/SolverBase.h"
#include "shared/HydroParams.h"
#include "shared/kokkos_shared.h"

#include "all_regime/HydroRunFunctors.h"
#include "all_regime/PostProcessing.h"
#include "all_regime/HydroInitFunctors.h"
#include "all_regime/LevelSet.h"
#include "all_regime/Extrapolation.h"

// for IO
#include <utils/io/IO_ReadWrite.h>

// for init condition
#include "shared/DamBreakParams.h"
#include "shared/GreshoParams.h"
#include "shared/IsentropicVortexParams.h"
#include "shared/PoiseuilleParams.h"
#include "shared/RayleighBenardParams.h"
#include "shared/NonIsothermParams.h"
#include "shared/StefanthermParams.h"
#include "shared/SuckingthermParams.h"
#include "shared/StaticBubbleParams.h"
#include "shared/TransientProcessParams.h"
#include "shared/initRiemannConfig2d.h"
#include "shared/BlastParams.h"

namespace euler_kokkos { namespace all_regime
	{

		/**
		 * Main hydrodynamics data structure for 2D/3D All-Regime scheme.
		 */
		template<int dim>
			class SolverHydroAllRegime : public euler_kokkos::SolverBase
		{
			public:
				//! Decide at compile-time which data array to use for 2d or 3d
				using DataArray = typename DataArrays<dim>::DataArray;

				//! Data array typedef for host memory space
				using DataArrayHost = typename DataArrays<dim>::DataArrayHost;

				//! Static creation method called by the solver factory.
				static SolverBase* create(HydroParams& params, ConfigMap& configMap);

				SolverHydroAllRegime(HydroParams& params, ConfigMap& configMap);
				virtual ~SolverHydroAllRegime();

				// fill boundaries / ghost 2d / 3d
				void make_boundaries(DataArray Udata);
				void make_boundaries_LS(DataArray Udata);
				void make_boundaries_LSC(DataArray Udata);
				void make_boundaries_GradPhi(DataArray Udata);

				void Extrapolation(DataArray Udata, DataArray U0data, DataArray gradphi, const int num_ite);
				//! init wrapper
				void init(DataArray Udata);

				//! compute time step inside an MPI process, at shared memory level.
				real_t compute_dt_local() override;

				//! perform 1 time step (time integration).
				void next_iteration_impl() override;

				//! numerical scheme
				void all_regime_scheme();

				// output
				void save_solution_impl() override;

				// Public Members
				DataArray     U;     /*!< hydrodynamics conservative variables arrays */
				DataArrayHost Uhost; /*!< U mirror on host memory space */
				DataArray     U2;    /*!< hydrodynamics conservative variables arrays */
				DataArray     Q;     /*!< hydrodynamics primitive    variables array  */
				DataArray     UExtra;     /*!< Extrapolation   variables array  */
				DataArray     gradPhi;     /*!< gradient of phi    variables array  */
				int isize, jsize, ksize;
				int nbCells;
		}; // class SolverHydroAllRegime

		// =======================================================
		// ==== CLASS SolverHydroAllRegime IMPL ==================
		// =======================================================

		// =======================================================
		// =======================================================
		/**
		 * Static creation method called by the solver factory.
		 */
		template<int dim>
			SolverBase* SolverHydroAllRegime<dim>::create(HydroParams& params, ConfigMap& configMap)
			{
				SolverHydroAllRegime<dim>* solver = new SolverHydroAllRegime<dim>(params, configMap);

				return solver;
			}

		// =======================================================
		// =======================================================
		/**
		 *
		 */
		template<int dim>
			SolverHydroAllRegime<dim>::SolverHydroAllRegime(HydroParams& params_, ConfigMap& configMap_) :
				SolverBase(params_, configMap_),
				U(), U2(), Q(),
				isize(params_.isize), jsize(params_.jsize), ksize(params_.ksize),
				nbCells(params_.isize*params_.jsize)
		{
			solver_type = SOLVER_ALL_REGIME;

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
				UExtra = DataArray("UExtra", isize, jsize, 12);
				gradPhi=DataArray("gradPhi", isize, jsize, 4);

				total_mem_size += isize*jsize*(3*nbvar+16) * sizeof(real_t);// 1+1+1 for U+U2+Q
				std::cout<<sizeof(real_t)<<std::endl;
			}
			else
			{
				U     = DataArray("U", isize, jsize, ksize, nbvar);
				Uhost = Kokkos::create_mirror(U);
				U2    = DataArray("U2",isize, jsize, ksize, nbvar);
				Q     = DataArray("Q", isize, jsize, ksize, nbvar);
				UExtra = DataArray("UExtra", isize, jsize, ksize, 12);
				gradPhi=DataArray("gradPhi", isize, jsize, ksize, 5);
				total_mem_size += isize*jsize*ksize*(nbvar*3+17)*sizeof(real_t);// 1+1+1=3 for U+U2+Q
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

			timers[TIMER_BOUNDARIES]->start();
			make_boundaries_LSC(U2);
			timers[TIMER_BOUNDARIES]->stop();

			LevelSetFunctors<dim>::ComputeGradPhiStep::apply(params, U2, gradPhi, nbCells);

			timers[TIMER_BOUNDARIES]->start();
			make_boundaries_GradPhi(gradPhi);
			timers[TIMER_BOUNDARIES]->stop();

			LevelSetFunctors<dim>::ComputeCurvature::apply(params, U2, gradPhi, nbCells);

			RunFunctors<dim>::ConvertToPrimitives::apply(params, U, Q, nbCells);

			Extrapolation(U, UExtra, gradPhi, TWO_F  * isize);

			compute_dt();

			int myRank=0;
#ifdef USE_MPI
			myRank = params.myRank;
#endif // USE_MPI

			if (myRank==0)
			{
				std::cout << "##########################" << std::endl;
				std::cout << "Solver is " << m_solver_name << std::endl;
				std::cout << "Problem (init condition) is " << m_problem_name << std::endl;
				std::cout << "##########################" << std::endl;

				// print parameters on screen
				params.print();
				std::cout << "##########################" << std::endl;
				std::cout << "Memory requested : " << (total_mem_size / 1e6) << " MBytes" << std::endl;
				std::cout << "##########################" << std::endl;
			}

		} // SolverHydroAllRegime::SolverHydroAllRegime

		// =======================================================
		// =======================================================
		/**
		 *
		 */
		template<int dim>
			SolverHydroAllRegime<dim>::~SolverHydroAllRegime()
			{

			} // SolverHydroAllRegime::~SolverHydroAllRegime

		template<int dim>
			void SolverHydroAllRegime<dim>::make_boundaries_LS(DataArray Udata)
			{
				make_boundaries_LS_serial(Udata);
			} // make_boundaries_LS
		template<int dim>
			void SolverHydroAllRegime<dim>::make_boundaries_LSC(DataArray Udata)
			{
				make_boundaries_LSC_serial(Udata);
			} // make_boundaries_LSC
		template<int dim>
			void SolverHydroAllRegime<dim>::make_boundaries_GradPhi(DataArray Udata)
			{
				make_boundaries_GradPhi_serial(Udata);
			} // make_boundaries

		template<int dim>
			void SolverHydroAllRegime<dim>::make_boundaries(DataArray Udata)
			{
#ifdef USE_MPI
				make_boundaries_mpi(Udata);
#else
				make_boundaries_serial(Udata);
#endif // USE_MPI
			} // make_boundaries

		// =======================================================
		// =======================================================
		/**
		 * Compute time step satisfying CFL condition.
		 *
		 * \return dt time step
		 */
		template<int dim>
			real_t SolverHydroAllRegime<dim>::compute_dt_local()
			{
				real_t invDt = ZERO_F;

				if (params.useAllRegimeTimeSteps)
				{
					// call device functor
					real_t invDtAcoustic = ZERO_F;
					RunFunctors<dim>::AcousticTimeStep::apply(params, Q, invDtAcoustic, nbCells);

					// call device functor
					real_t invDtTransport = ZERO_F;
					RunFunctors<dim>::TransportTimeStep::apply(params, Q,invDtTransport, nbCells);

					invDt = FMAX(invDtTransport, invDtAcoustic);
				}
				else
				{
					// call device functor
					RunFunctors<dim>::TimeStep::apply(params, U, invDt, nbCells);
				}

				const real_t dt = params.settings.cfl/invDt;

				return dt;
			} // SolverHydroAllRegime::compute_dt_local

		// =======================================================
		// =======================================================
		template<>
			void SolverHydroAllRegime<3>::Extrapolation(DataArray Udata, DataArray U0data, DataArray gradphi, const int num_ite)
			{

			}
		template<>
			void SolverHydroAllRegime<2>::Extrapolation(DataArray Udata, DataArray U0data, DataArray gradphi, const int num_ite)
			{

				ExtrapolationFunctors<2>::ComputeFirstDerivative::apply(params, Udata, U0data, gradphi, nbCells );


				for (int n=0; n<num_ite; n++)
				{
					RunFunctors<2>::CopyGradient::apply(params, U0data, FD0, U0data, RD0, nbCells);
					RunFunctors<2>::CopyGradient::apply(params, U0data, FD1, U0data, RD1, nbCells);
					RunFunctors<2>::CopyGradient::apply(params, U0data, FE0, U0data, RE0, nbCells);
					RunFunctors<2>::CopyGradient::apply(params, U0data, FE1, U0data, RE1, nbCells);

					ExtrapolationFunctors<2>::ExtrapolateFirstDerivative::apply(params, Udata, U0data, gradphi, nbCells);
					ExtrapolationFunctors<2>::ExtrapolateFirstDerivativeStep2::apply(params, Udata, U0data, gradphi, nbCells);

				}

				ExtrapolationFunctors<2>::CopyToGhost::apply(params, Udata, U0data, nbCells );


				for (int n=0; n<num_ite; n++)
				{
					RunFunctors<2>::CopyGradient::apply(params, U0data, ID0, U0data, RD0, nbCells);
					RunFunctors<2>::CopyGradient::apply(params, U0data, ID1, U0data, RD1, nbCells);
					RunFunctors<2>::CopyGradient::apply(params, U0data, IE0, U0data, RE0, nbCells);
					RunFunctors<2>::CopyGradient::apply(params, U0data, IE1, U0data, RE1, nbCells);

					ExtrapolationFunctors<2>::Extrapolate::apply(params, Udata, U0data, gradphi, nbCells );
					ExtrapolationFunctors<2>::ExtrapolateStep2::apply(params, Udata, U0data, gradphi, nbCells );


				}

			}
		template<int dim>
			void SolverHydroAllRegime<dim>::next_iteration_impl()
			{
				int myRank=0;

#ifdef USE_MPI
				myRank = params.myRank;
#endif // USE_MPI

				std::ostringstream oss;
				oss << std::scientific;
				oss << std::setprecision(std::numeric_limits<real_t>::max_digits10);

				if ((m_iteration % m_nlog == 0 || (params.enableOutput && should_save_solution())) )
				{
					oss << "Step=" << std::setw(std::numeric_limits<int>::digits10) << std::setfill('.') << m_iteration;
					oss << " (dt=" << m_dt << " t=" << m_t << ")"<<std::endl;

					//	m_times_saved++;
				}
				if (m_problem_name == "static_bubble")
				{
					real_t umax=ZERO_F;
					PostProcessingFunctors<dim>::ComputeMaxVelocity::apply(params, Q, umax, nbCells);
					std::ofstream location_out;

					location_out.open("velocity_fluctuations.dat", std::ios::out | std::ios::app); 
					location_out <<m_t*params.settings.mu0/0.64<< " " <<umax*sqrt(0.8/params.settings.sigma)<< "\n"  ;   
					location_out.close();
				}


				// output
				if (params.enableOutput && should_save_solution())
				{
					oss << "Step=" << std::setw(std::numeric_limits<int>::digits10) << std::setfill('.') << m_iteration;
					oss << " (dt=" << m_dt << " t=" << m_t << ")\n";
					oss << "--> Saving results\n";
					save_solution();
				} // end enable output

				if (myRank == 0)
				{
					std::cout << oss.str()<<std::flush;
				}

				// compute new dt
				timers[TIMER_DT]->start();
				compute_dt();
				timers[TIMER_DT]->stop();

				// perform one step integration
				all_regime_scheme();
			} // SolverHydroAllRegime::next_iteration_impl

		template<int dim>
			void SolverHydroAllRegime<dim>::all_regime_scheme()
			{
				// Time state :
				// U  [inner region] : n
				// U2 [inner region] : n
				// U  [ghost region] : n
				// U2 [ghost region] : n
				// Q  [whole region] : n

				timers[TIMER_NUM_SCHEME]->start();

				//RunFunctors<dim>::ConvertToPrimitives::apply(params, U, Q, nbCells);

				RunFunctors<dim>::AcousticStep::apply(params, U, Q, gradPhi, m_dt, nbCells);

				timers[TIMER_BOUNDARIES]->start();
				make_boundaries(U);
				timers[TIMER_BOUNDARIES]->stop();
				// Time state :
				// U  [inner region] : n+1-
				// U2 [inner region] : n
				// U  [ghost region] : n+1-
				// U2 [ghost region] : n
				// Q  [whole reigon] : n
				Extrapolation(U, UExtra, gradPhi, 8);

				RunFunctors<dim>::TransportStep::apply(params, U, Q, U2,gradPhi, UExtra, m_dt, nbCells);
				// Time state :
				// U  [inner region] : n+1-
				// U2 [inner region] : n+1
				// U  [ghost region] : n+1-
				// U2 [ghost region] : n
				// Q  [whole region] : n

				if (fmin(params.settings.kappa0,params.settings.kappa1) > ZERO_F)
				{
					RunFunctors<dim>::HeatDiffusionStep::apply(params, U2, Q, m_dt, nbCells);
				}

				if (params.settings.mu0 > ZERO_F)
				{
					RunFunctors<dim>::ViscosityStep::apply(params, U2, Q, m_dt, nbCells);
				}



				// fill ghost cell in U2
				timers[TIMER_BOUNDARIES]->start();
				make_boundaries(U2);
				timers[TIMER_BOUNDARIES]->stop();
				// Time state :
				// U  [inner region] : n+1-
				// U2 [inner region] : n+1
				// U  [ghost region] : n+1-
				// U2 [ghost region] : n+1
				// Q  [whole region] : n

				Kokkos::deep_copy(U, U2);
				// Time state :
				// U  [inner region] : n+1
				// U2 [inner region] : n+1
				// U  [ghost region] : n+1
				// U2 [ghost region] : n+1
				// Q  [whole region] : n

				// convert conservative variable into primitives ones for the
				// entire domain
				RunFunctors<dim>::ConvertToPrimitives::apply(params, U, Q, nbCells);
				// Time state :
				// U  [inner region] : n+1
				// U2 [inner region] : n+1
				// U  [ghost region] : n+1
				// U2 [ghost region] : n+1
				// Q  [whole region] : n+1
				Extrapolation(U, UExtra, gradPhi, 8);

				//  Start Level Set Advection      
				timers[TIMER_BOUNDARIES]->start();
				make_boundaries_LSC(U2);
				timers[TIMER_BOUNDARIES]->stop();

				if (m_iteration%2==0)
					LevelSetFunctors<dim>::TransportPhiXStep::apply(params, U, U2, Q, gradPhi, m_dt, nbCells);
				else
					LevelSetFunctors<dim>::TransportPhiYStep::apply(params, U, U2, Q, gradPhi, m_dt, nbCells);

				timers[TIMER_BOUNDARIES]->start();
				make_boundaries_LSC(U2);
				timers[TIMER_BOUNDARIES]->stop();

				LevelSetFunctors<dim>::CopyLS::apply(params, U, U2, nbCells);

				if (m_iteration%2==0)
					LevelSetFunctors<dim>::TransportPhiYStep::apply(params, U, U2, Q, gradPhi, m_dt, nbCells);
				else
					LevelSetFunctors<dim>::TransportPhiXStep::apply(params, U, U2, Q, gradPhi, m_dt, nbCells);
				//  End Level Set Advection      

				//  Start Level Set Redistance      
				if (params.settings.redistance)
				{

					if (m_iteration%params.settings.redistance_frequence==params.settings.redistance_frequence/2)
					{
						for (int i=0; i<4; i++)
						{
							LevelSetFunctors<dim>::CopyLS::apply(params, U, U2, nbCells);
							timers[TIMER_BOUNDARIES]->start();
							make_boundaries_LS(U2);
							timers[TIMER_BOUNDARIES]->stop();

							LevelSetFunctors<dim>::ComputeGradPhiStep::apply(params, U2, gradPhi, nbCells);

							LevelSetFunctors<dim>::RedistancingPhiStep1::apply(params,U, U2, gradPhi, nbCells);

							timers[TIMER_BOUNDARIES]->start();
							make_boundaries_LS(U2);
							timers[TIMER_BOUNDARIES]->stop();

							LevelSetFunctors<dim>::ComputeGradPhiStep::apply(params, U2, gradPhi, nbCells);

							LevelSetFunctors<dim>::RedistancingPhiStep2::apply(params,U, U2, gradPhi, nbCells);

							timers[TIMER_BOUNDARIES]->start();
							make_boundaries_LS(U2);
							timers[TIMER_BOUNDARIES]->stop();

							LevelSetFunctors<dim>::ComputeGradPhiStep::apply(params, U2, gradPhi, nbCells);

							LevelSetFunctors<dim>::RedistancingPhiStep3::apply(params,U, U2, gradPhi, nbCells);
						}
					}
				}
				//  End Level Set Redistance      //

				timers[TIMER_BOUNDARIES]->start();
				make_boundaries_LSC(U2);
				timers[TIMER_BOUNDARIES]->stop();

				LevelSetFunctors<dim>::ComputeGradPhiStep::apply(params, U2, gradPhi, nbCells);

				timers[TIMER_BOUNDARIES]->start();
				make_boundaries_GradPhi(gradPhi);
				timers[TIMER_BOUNDARIES]->stop();

				if (params.settings.sigma>ZERO_F)
					LevelSetFunctors<dim>::ComputeCurvature::apply(params, U2, gradPhi, nbCells);


				RunFunctors<dim>::StateChangeStep::apply(params, U2, Q, UExtra, nbCells);

				timers[TIMER_BOUNDARIES]->start();
				make_boundaries_LS(U2);
				timers[TIMER_BOUNDARIES]->stop();

				timers[TIMER_BOUNDARIES]->start();
				make_boundaries(U2);
				timers[TIMER_BOUNDARIES]->stop();

				Kokkos::deep_copy(U, U2);

				RunFunctors<dim>::ConvertToPrimitives::apply(params, U, Q, nbCells);

				timers[TIMER_NUM_SCHEME]->stop();
			}

		template<>
			void SolverHydroAllRegime<2>::init(DataArray Udata)
			{
				/*
				 * initialize hydro array at t=0
				 */
				if (m_problem_name == "implode")
				{
					InitFunctors<2>::Implode::apply(params, Udata, nbCells);
				}
				else if (m_problem_name == "dam_break")
				{
					DamBreakParams dbParams(configMap);
					InitFunctors<2>::DamBreak::apply(params, dbParams, Udata, nbCells);
				}
				else if (m_problem_name == "poiseuille")
				{
					PoiseuilleParams poiseuilleParams(configMap);
					InitFunctors<2>::Poiseuille::apply(params, poiseuilleParams, Udata, nbCells);
				}
				else if (m_problem_name == "blast")
				{
					BlastParams blastParams = BlastParams(configMap);
					InitFunctors<2>::Blast::apply(params, blastParams, Udata, nbCells);
				}
				else if (m_problem_name == "rayleigh_benard")
				{
					RayleighBenardParams rayleighBenardParams(configMap);
					InitFunctors<2>::RayleighBenard::apply(params, rayleighBenardParams, Udata, nbCells);
				}
				else if (m_problem_name == "stefantherm")
				{
					StefanthermParams StefanthermParams(configMap);
					InitFunctors<2>::Stefantherm::apply(params, StefanthermParams, Udata, nbCells);
				}
				else if (m_problem_name == "suckingtherm")
				{
					SuckingthermParams SuckingthermParams(configMap);
					InitFunctors<2>::Suckingtherm::apply(params, SuckingthermParams, Udata, nbCells);
				}
				else if (m_problem_name == "suckingtherm")
				{
					SuckingthermParams SuckingthermParams(configMap);
					InitFunctors<2>::Suckingtherm::apply(params, SuckingthermParams, Udata, nbCells);
				}
				else if (m_problem_name == "non_isotherm")
				{
					NonIsothermParams NonIsothermParams(configMap);
					InitFunctors<2>::NonIsotherm::apply(params, NonIsothermParams, Udata, nbCells);
				}
				else if (m_problem_name == "transient")
				{
					TransientParams TransientParams(configMap);
					InitFunctors<2>::Transient::apply(params, TransientParams, Udata, nbCells);
				}
				else if (m_problem_name == "rayleigh_taylor")
				{
					InitFunctors<2>::RayleighTaylor::apply(params, Udata, nbCells);
				}
				else if (m_problem_name == "rayleigh_taylor_instabilities")
				{
					InitFunctors<2>::RayleighTaylorInstabilities::apply(params, Udata, nbCells);
				}
				else if (m_problem_name == "static_bubble")
				{
					StaticBubbleParams StaticBubbleParams(configMap);
					InitFunctors<2>::StaticBubble::apply(params, StaticBubbleParams, Udata, nbCells);
				}
				else if (m_problem_name == "Case34")
				{
					InitFunctors<2>::Case34::apply(params, Udata, nbCells);
				}
				else if (m_problem_name == "rising_bubble")
				{
					InitFunctors<2>::RisingBubble::apply(params, Udata, nbCells);
				}
				else if (m_problem_name == "atmosphere_at_rest")
				{
					InitFunctors<2>::AtmosphereAtRest::apply(params, Udata, nbCells);
				}
				else if (m_problem_name == "gresho")
				{
					GreshoParams greshoParams(configMap);
					InitFunctors<2>::Gresho::apply(params, greshoParams, Udata, nbCells);
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
			} // SolverHydroMuscinit / 2d

		template<>
			void SolverHydroAllRegime<3>::init(DataArray Udata)
			{
				/*
				 * initialize hydro array at t=0
				 */
				if (m_problem_name == "implode")
				{
					InitFunctors<3>::Implode::apply(params, Udata, nbCells);
				}
				else if (m_problem_name == "poiseuille")
				{
					PoiseuilleParams poiseuilleParams(configMap);
					InitFunctors<3>::Poiseuille::apply(params, poiseuilleParams, Udata, nbCells);
				}
				else if (m_problem_name == "blast")
				{
					BlastParams blastParams = BlastParams(configMap);
					InitFunctors<3>::Blast::apply(params, blastParams, Udata, nbCells);
				}
				else if (m_problem_name == "rayleigh_taylor")
				{
					InitFunctors<3>::RayleighTaylor::apply(params, Udata, nbCells);
				}
				else
				{
					std::cout << "Problem : " << m_problem_name
						<< " is not recognized / implemented."
						<< std::endl;
					std::cout << "Exiting..." << std::endl;
					std::exit(EXIT_FAILURE);
				}
			} // SolverHydroMuscinit / 2d

		// =======================================================
		// =======================================================
		template<int dim>
			void SolverHydroAllRegime<dim>::save_solution_impl()
			{
				timers[TIMER_IO]->start();
				save_data(Q,  Uhost, m_times_saved, m_t);
				timers[TIMER_IO]->stop();
			} // SolverHydroAllRegime::save_solution_impl()

	} // namespace all_regime

} // namespace euler_kokkos
