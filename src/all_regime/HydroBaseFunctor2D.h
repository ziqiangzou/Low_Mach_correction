#pragma once

#include "shared/kokkos_shared.h"
#include "shared/HydroParams.h"
#include "shared/HydroState.h"
#include "shared/units.h"
#include <iomanip>


namespace euler_kokkos { namespace all_regime
	{

		/**
		 * Base class to derive actual kokkos functor for hydro 2D.
		 * params is passed by copy.
		 */
		class HydroBaseFunctor2D
		{
			public:
				using HydroState     = HydroState2d;
				using DataArray      = DataArray2d;
				using DataArrayConst = DataArray2dConst;

				HydroBaseFunctor2D(HydroParams params_) :
					params(params_), nbvar(params_.nbvar) {};
				virtual ~HydroBaseFunctor2D() {};

				const HydroParams params;
				const int nbvar;

				/**
				 * Compute gravitational potential at position (x, y)
				 * @param[in]  x    x-coordinate
				 * @param[in]  y    y-coordinate
				 * @param[out] psi  gravitational potential
				 */
				KOKKOS_INLINE_FUNCTION
					real_t psi(real_t x, real_t y) const
					{
						return - params.settings.g_x * x - params.settings.g_y * y;
					} // psi

				/**
				 * Compute gravitational potential at the center
				 * of the cell C(i, j)
				 * @param[in]  i    logical x-coordinate of the cell C
				 * @param[in]  j    logical y-coordinate of the cell C
				 * @param[out] psi  gravitational potential
				 */
				KOKKOS_INLINE_FUNCTION
					real_t psi(int i, int j) const
					{
#ifdef USE_MPI
						const int nx_mpi = params.nx * params.myMpiPos[IX];
						const int ny_mpi = params.ny * params.myMpiPos[IY];
						const real_t x = params.xmin + (HALF_F + i + nx_mpi - params.ghostWidth)*params.dx;
						const real_t y = params.ymin + (HALF_F + j + ny_mpi - params.ghostWidth)*params.dy;
#else
						const real_t x = params.xmin + (HALF_F + i - params.ghostWidth)*params.dx;
						const real_t y = params.ymin + (HALF_F + j - params.ghostWidth)*params.dy;
#endif

						return psi(x, y);
					} // psi

				/**
				 * Compute \mathcal{M} = \rho * \Delta \psi between two cells
				 * (to be renamed)
				 */
				KOKKOS_INLINE_FUNCTION
					real_t computeM(const HydroState& qLoc, real_t psiLoc,
							const HydroState& qNei, real_t psiNei) const
					{
						return HALF_F * (qLoc[ID]+qNei[ID]) * (psiNei - psiLoc);
					} // computeM

				/**
				 * Get HydroState from global array (either conservative or primitive)
				 * at cell C(i, j) (global to local)
				 * @param[in]  array global array
				 * @param[in]  i     logical x-coordinate of the cell C
				 * @param[in]  j     logical y-coordinate of the cell C
				 * @param[out] state HydroState of cell C(i, j)
				 */
				KOKKOS_INLINE_FUNCTION
					HydroState getHydroState(DataArrayConst array, int i, int j) const
					{
						HydroState state;
						state[ID] = array(i, j, ID);
						state[IP] = array(i, j, IP);
						state[IS] = array(i, j, IS);
						state[IU] = array(i, j, IU);
						state[IV] = array(i, j, IV);
						return state;
					}
				/**
				 * Set HydroState to global array (either conservative or primitive)
				 * at cell C(i, j) (local to global)
				 * @param[in, out]  array global array
				 * @param[in]       state HydroState of cell C(i, j)
				 * @param[in]       i     logical x-coordinate of the cell C
				 * @param[in]       j     logical y-coordinate of the cell C
				 */
				KOKKOS_INLINE_FUNCTION
					void setHydroState(DataArray array, const HydroState& state, int i, int j) const
					{
						array(i, j, ID) = state[ID];
						array(i, j, IP) = state[IP];
						array(i, j, IS) = state[IS];
						array(i, j, IU) = state[IU];
						array(i, j, IV) = state[IV];
					}
				KOKKOS_INLINE_FUNCTION
					void computeGradphiInter(const real_t gradphixLoc,const real_t gradphiyLoc,const real_t phiLoc,\
							const real_t gradphixNei,const real_t gradphiyNei,const real_t phiNei, real_t& dphix, real_t& dphiy)const
					{
						dphix=(fabs(phiNei)*gradphixLoc+fabs(phiLoc)*gradphixNei)/(fabs(phiNei)+fabs(phiLoc));
						dphiy=(fabs(phiNei)*gradphiyLoc+fabs(phiLoc)*gradphiyNei)/(fabs(phiNei)+fabs(phiLoc));

						const real_t dphiM=sqrt(dphix*dphix+dphiy*dphiy);
						dphix/=(dphiM);
						dphiy/=(dphiM);
					}
				KOKKOS_INLINE_FUNCTION
					HydroState  compute_GhostState(const int i, const int j, const HydroState& uLoc, const HydroState& uMx, const HydroState& uPx,\
							const HydroState& uMy,  const HydroState& uPy, const real_t dphix, const real_t dphiy) const
					{
						HydroState qghost;
						real_t phi =uLoc[IH];
						real_t p=ZERO_F, rho=ZERO_F, u=ZERO_F, v=ZERO_F, number = ZERO_F;
						int ghostWidth=params.ghostWidth;

						if (i>ghostWidth)
						{
							const real_t phiMx=uMx[IH];
							if (phiMx*phi<=ZERO_F&&fmax(phiMx, phi)>ZERO_F)
							{
								HydroState qMx=computePrimitives(uMx);

								real_t rho0=qMx[ID], p0=qMx[IP];
								const real_t num=ONE_F;

								const real_t vn    =qMx[IU]*dphix    +qMx[IV]*dphiy;
								const real_t vt    =qMx[IV]*dphix    -qMx[IU]*dphiy;

								rho+=rho0*num;
								p+=p0*num;
								u+=vn*num;
								v+=vt*num;
								number+=num;

							}
						}
						if (i<params.imax-ghostWidth)
						{
							const real_t phiPx=uPx[IH];
							if (phiPx*phi<=ZERO_F&&fmax(phiPx, phi)>ZERO_F)
							{
								HydroState qPx=computePrimitives(uPx);

								real_t rho0=qPx[ID], p0=qPx[IP];
								const real_t num=ONE_F;

								const real_t vn    =qPx[IU]*dphix    +qPx[IV]*dphiy;
								const real_t vt    =qPx[IV]*dphix    -qPx[IU]*dphiy;
								rho+=rho0*num;
								p+=p0*num;
								u+=vn*num;
								v+=vt*num;
								number+=num;
							}
						}
						if (j>ghostWidth)
						{
							const real_t phiMy=uMy[IH];
							if (phiMy*phi<=ZERO_F&&fmax(phiMy, phi)>ZERO_F)
							{
								HydroState qMy=computePrimitives(uMy);

								real_t rho0=qMy[ID], p0=qMy[IP];
								const real_t num=ONE_F;

								const real_t vn    =qMy[IU]*dphix    +qMy[IV]*dphiy;
								const real_t vt    =qMy[IV]*dphix    -qMy[IU]*dphiy;
								rho+=rho0*num;
								u+=vn*num;
								v+=vt*num;
								p+=p0*num;
								number+=num;
							}
						}
						if (j<params.jmax-ghostWidth)
						{
							const real_t phiPy=uPy[IH];
							if (phiPy*phi<=ZERO_F&&fmax(phiPy, phi)>ZERO_F)
							{
								HydroState qPy=computePrimitives(uPy);

								real_t rho0=qPy[ID], p0=qPy[IP];
								const real_t num=ONE_F;
								const real_t vn    =qPy[IU]*dphix    +qPy[IV]*dphiy;
								const real_t vt    =qPy[IV]*dphix    -qPy[IU]*dphiy;

								rho+=rho0*num;
								u+=vn*num;
								v+=vt*num;
								p+=p0*num;
								number+=num;
							}
						}
						rho/=number;
						p/=number;
						u/=number;
						v/=number;
						const real_t ux=(dphix *      u   -   v    * dphiy) / (dphix * dphix + dphiy * dphiy);
						const real_t uy=(dphix *      v    +  u    * dphiy) / (dphix * dphix + dphiy * dphiy);
						qghost[IH]=phi>ZERO_F? -ONE_F:ONE_F;

						qghost[ID]=rho;
						qghost[IV]=uy;
						qghost[IU]=ux;
						qghost[IP]=p;

						return computeConservatives(qghost);
					}
				KOKKOS_INLINE_FUNCTION
					void computeCurvatureInter(const real_t KLoc,const real_t  KNei, const real_t  phiLoc, const real_t phiNei, real_t& deltap)const
					{
						real_t Kinter=(fabs(phiLoc)*KNei+fabs(phiNei)*KLoc)/(fabs(phiLoc)+fabs(phiNei));
						Kinter=fabs(Kinter)>params.onesurdx?Kinter*params.onesurdx/fabs(Kinter):Kinter;
						deltap=(phiLoc<=ZERO_F)? params.settings.sigma*Kinter:-params.settings.sigma*Kinter;
					}


				/**
				 * Compute temperature using ideal gas law
				 * @param[in]  q  primitive variables array
				 * @param[out] T  temperature
				 */
				KOKKOS_INLINE_FUNCTION
					real_t computeTemperature(const HydroState& q) const
					{
						const real_t Rstar= q[IH]>ZERO_F? params.settings.Rstar0:params.settings.Rstar1;
						const real_t Bstate= q[IH]>ZERO_F? params.Bstate0:params.Bstate1;
						return (q[IP]+ Bstate) / (q[ID] * Rstar);

					} // computeTemperature

				/**
				 * Compute speed of sound using ideal gas law
				 * @param[in]  q  primitive variables array
				 * @param[out] c  speed of sound
				 */
				KOKKOS_INLINE_FUNCTION
					real_t computeSpeedSound(const HydroState& q) const
					{
						const real_t gamma=(q[IH]>ZERO_F)?  params.settings.gamma0 :params.settings.gamma1;
						const real_t Bstate=(q[IH]>ZERO_F)? params.Bstate0:params.Bstate1;

						return SQRT(gamma * (q[IP]+Bstate) / q[ID]);

					} // computeSpeedSound

				/**
				 * Convert conservative variables (rho, rho*u, rho*v, rho*E) to
				 * primitive variables (rho, u, v, p) using ideal gas law
				 * @param[in]  u  conservative variables array
				 * @param[out] q  primitive    variables array
				 */
				KOKKOS_INLINE_FUNCTION
					HydroState computePrimitives(const HydroState& u) const
					{
						const real_t phiLoc=u[IH];
						const real_t gamma=(phiLoc>ZERO_F)?params.settings.gamma0:params.settings.gamma1;
						const real_t Bstate=(phiLoc>ZERO_F)?params.Bstate0:params.Bstate1;
						const bool barotropic=(phiLoc>ZERO_F)?params.settings.barotropic0:params.settings.barotropic1;


						const real_t invD = ONE_F / u[ID];
						const real_t ekin = HALF_F*(u[IU]*u[IU]+u[IV]*u[IV])*invD;

						HydroState q;
						q[ID] = u[ID];
						q[IH] = u[IH];
						q[IU] = u[IU] * invD;
						q[IV] = u[IV] * invD;
						if (!barotropic)
							q[IP] = (gamma-ONE_F) * (u[IE] - ekin)-gamma* Bstate;
						else
							q[IP] = u[IE];


						return q;
					} // computePrimitives

				/**
				 * Convert primitive variables (rho, p, u, v) to
				 * conservative variables (rho, rho*E, rho*u, rho*v) using ideal gas law
				 * @param[in]  q  primitive    variables array
				 * @param[out] u  conservative variables array
				 */
				KOKKOS_INLINE_FUNCTION
					HydroState computeConservatives(const HydroState& q) const
					{
						const real_t phiLoc=q[IH];
						const real_t gamma=(phiLoc>ZERO_F)?params.settings.gamma0:params.settings.gamma1;
						const real_t Bstate=(phiLoc>ZERO_F)?params.Bstate0:params.Bstate1;
						const bool  barotropic=(phiLoc>ZERO_F)?params.settings.barotropic0:params.settings.barotropic1;


						const real_t ekin = HALF_F*q[ID]*(q[IU]*q[IU]+q[IV]*q[IV]);

						HydroState u;
						u[ID] = q[ID];
						u[IH] = q[IH];
						u[IU] = q[ID] * q[IU];
						u[IV] = q[ID] * q[IV];
						if (!barotropic)
							u[IE] = (q[IP]+gamma*Bstate)/(gamma-ONE_F) + ekin;
						else
							u[IE] = q[IP];


						return u;
					} // computeConservatives
		}; // class HydroBaseFunctor2D

	} // namespace all_regime

} // namespace euler_kokkos
