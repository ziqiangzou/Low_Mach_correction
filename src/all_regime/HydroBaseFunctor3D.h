#pragma once

#include "shared/kokkos_shared.h"
#include "shared/HydroParams.h"
#include "shared/HydroState.h"
#include "shared/units.h"

namespace euler_kokkos { namespace all_regime
{

/**
 * Base class to derive actual kokkos functor for hydro 3D.
 * params is passed by copy.
 */
class HydroBaseFunctor3D
{
public:
    using HydroState     = HydroState3d;
    using DataArray      = DataArray3d;
    using DataArrayConst = DataArray3dConst;

    HydroBaseFunctor3D(HydroParams params_) :
        params(params_), nbvar(params_.nbvar) {};
    virtual ~HydroBaseFunctor3D() {};

    const HydroParams params;
    const int nbvar;

    /**
     * Compute gravitational potential at position (x, y)
     * @param[in]  x    x-coordinate
     * @param[in]  y    y-coordinate
     * @param[in]  z    z-coordinate
     * @param[out] phi  gravitational potential
     */
    KOKKOS_INLINE_FUNCTION
    real_t phi(real_t x, real_t y, real_t z) const
    {
        return - params.settings.g_x * x - params.settings.g_y * y - params.settings.g_z * z;
    } // phi

    /**
     * Compute gravitational potential at the center
     * of the cell C(i, j, k)
     * @param[in]  i    logical x-coordinate of the cell C
     * @param[in]  j    logical y-coordinate of the cell C
     * @param[in]  k    logical z-coordinate of the cell C
     * @param[out] phi  gravitational potential
     */
    KOKKOS_INLINE_FUNCTION
    real_t phi(int i, int j, int k) const
    {
#ifdef USE_MPI
        const int nx_mpi = params.nx * params.myMpiPos[IX];
        const int ny_mpi = params.ny * params.myMpiPos[IY];
        const int nz_mpi = params.nz * params.myMpiPos[IZ];
        const real_t x = params.xmin + (HALF_F + i + nx_mpi - params.ghostWidth)*params.dx;
        const real_t y = params.ymin + (HALF_F + j + ny_mpi - params.ghostWidth)*params.dy;
        const real_t z = params.zmin + (HALF_F + k + nz_mpi - params.ghostWidth)*params.dz;
#else
        const real_t x = params.xmin + (HALF_F + i - params.ghostWidth)*params.dx;
        const real_t y = params.ymin + (HALF_F + j - params.ghostWidth)*params.dy;
        const real_t z = params.zmin + (HALF_F + k - params.ghostWidth)*params.dz;
#endif

        return phi(x, y, z);
    } // phi

    /**
     * Compute \mathcal{M} = \rho * \Delta \phi between two cells
     * (to be renamed)
     */
    KOKKOS_INLINE_FUNCTION
    real_t computeM(const HydroState& qLoc, real_t phiLoc,
                    const HydroState& qNei, real_t phiNei) const
    {
        return HALF_F * (qLoc[ID]+qNei[ID]) * (phiNei - phiLoc);
    } // computeM

    /**
     * Get HydroState from global array (either conservative or primitive)
     * at cell C(i, j, k) (global to local)
     * @param[in]  array global array
     * @param[in]  i     logical x-coordinate of the cell C
     * @param[in]  j     logical y-coordinate of the cell C
     * @param[in]  k     logical z-coordinate of the cell C
     * @param[out] state HydroState of cell C(i, j, k)
     */
    KOKKOS_INLINE_FUNCTION
    HydroState getHydroState(DataArrayConst array, int i, int j, int k) const
    {
        HydroState state;
        state[ID] = array(i, j, k, ID);
        state[IP] = array(i, j, k, IP);
        state[IS] = array(i, j, k, IS);
        state[IU] = array(i, j, k, IU);
        state[IV] = array(i, j, k, IV);
        state[IW] = array(i, j, k, IW);
        return state;
    } // getHydroState

    /**
     * Set HydroState to global array (either conservative or primitive)
     * at cell C(i, j, k) (local to global)
     * @param[in, out]  array global array
     * @param[in]       state HydroState of cell C(i, j, k)
     * @param[in]       i     logical x-coordinate of the cell C
     * @param[in]       j     logical y-coordinate of the cell C
     * @param[in]       k     logical z-coordinate of the cell C
     */
    KOKKOS_INLINE_FUNCTION
    void setHydroState(DataArray array, const HydroState & state, int i, int j, int k) const
    {
        array(i, j, k, ID) = state[ID];
        array(i, j, k, IP) = state[IP];
        array(i, j, k, IS) = state[IS];
        array(i, j, k, IU) = state[IU];
        array(i, j, k, IV) = state[IV];
        array(i, j, k, IW) = state[IW];
    } // setHydroState

    /**
     * Compute temperature using ideal gas law
     * @param[in]  q  primitive variables array
     * @param[out] T  temperature
     */
    KOKKOS_INLINE_FUNCTION
    real_t computeTemperature(const HydroState& q) const
    {
        const real_t Rstar=q[IH]>ZERO_F? params.settings.Rstar0:params.settings.Rstar1;
        return q[IP] / (q[ID] * Rstar);
    } // computeTemperature

    /**
     * Compute speed of sound using ideal gas law
     * @param[in]  q  primitive variables array
     * @param[out] c  speed of sound
     */
    KOKKOS_INLINE_FUNCTION
    real_t computeSpeedSound(const HydroState& q) const
    {
        return SQRT(params.settings.gamma0 * q[IP] / q[ID]);
    } // computeSpeedSound

    /**
     * Convert conservative variables (rho, rho*E, rho*u, rho*v, rho*w) to
     * primitive variables (rho, p, u, v, w) using ideal gas law
     * @param[in]  u  conservative variables array
     * @param[out] q  primitive    variables array
     */
    KOKKOS_INLINE_FUNCTION
    HydroState computePrimitives(const HydroState& u) const
    {
        const real_t gamma0 = params.settings.gamma0;
        const real_t invD = ONE_F / u[ID];
        const real_t ekin = HALF_F*(u[IU]*u[IU]+u[IV]*u[IV]+u[IW]*u[IW])*invD;

        HydroState q;
        q[ID] = u[ID];
        q[IP] = (gamma0-ONE_F) * (u[IE] - ekin);
        q[IS] = u[IS] * invD;
        q[IU] = u[IU] * invD;
        q[IV] = u[IV] * invD;
        q[IW] = u[IW] * invD;

        return q;
    } // computePrimitives

    /**
     * Convert primitive variables (rho, p, u, v, w) to
     * conservative variables (rho, rho*E, rho*u, rho*v, rho*w) using ideal gas law
     * @param[in]  q  primitive    variables array
     * @param[out] u  conservative variables array
     */
    KOKKOS_INLINE_FUNCTION
    HydroState computeConservatives(const HydroState& q) const
    {
        const real_t gamma0 = params.settings.gamma0;
        const real_t ekin = HALF_F*q[ID]*(q[IU]*q[IU]+q[IV]*q[IV]+q[IW]*q[IW]);

        HydroState u;
        u[ID] = q[ID];
        u[IE] = q[IP] / (gamma0-ONE_F) + ekin;
        u[IS] = q[ID] * q[IS];
        u[IU] = q[ID] * q[IU];
        u[IV] = q[ID] * q[IV];
        u[IW] = q[ID] * q[IW];

        return u;
    } // computeConservatives
}; // class HydroBaseFunctor3D

} // namespace all_regime

} // namespace euler_kokkos
