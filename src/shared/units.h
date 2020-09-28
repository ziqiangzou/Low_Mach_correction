#pragma once

#include <cmath>
#include <limits>
#include "shared/real_type.h"

// sqrtNewtonRaphson and my_sqrt come from github :
// https://gist.github.com/alexshtf/eb5128b3e3e143187794
namespace Detail
{
constexpr real_t sqrtNewtonRaphson(real_t x, real_t curr, real_t prev)
{
    return (curr == prev)
        ? curr
        : sqrtNewtonRaphson(x, HALF_F * (curr + x / curr), curr);
}
}

/*
 * Constexpr version of the square root
 * Return value:
 *      - For a finite and non-negative value of "x", returns an approximation for the square root of "x"
 *   - Otherwise, returns NaN
 */
constexpr real_t my_sqrt(real_t x)
{
    return (x >= ZERO_F && x < std::numeric_limits<real_t>::infinity())
        ? Detail::sqrtNewtonRaphson(x, x, ZERO_F)
        : std::numeric_limits<real_t>::quiet_NaN();
}

namespace euler_kokkos { namespace si_units
{

constexpr real_t length_u          = ONE_F; // in SI (m)
constexpr real_t mass_u            = ONE_F; // in SI (kg)
constexpr real_t time_u            = ONE_F; // in SI (s)
constexpr real_t temperature_u     = ONE_F; // in SI (K)
constexpr real_t matter_quantity_u = ONE_F; // in SI (mol)

}}

namespace euler_kokkos{ namespace si_units{ namespace constants
{

// Avogadro number in SI (mol^-1)
constexpr real_t Na      = static_cast<real_t>(6.022140857e+23l);
// Boltzmann constant in SI (J.K^-1)
constexpr real_t k_b     = static_cast<real_t>(1.38064852e-23l);
// Universal gas constant in SI (J.mol^-1.K^-1)
constexpr real_t R       = static_cast<real_t>(8.3144598e+0l);
// Mass of hydrogen in SI (kg)
constexpr real_t m_h     = static_cast<real_t>(1.6737236e-27l);
// Mean molecular weight of hydrogen (no unit)
constexpr real_t mmw_h   = m_h / m_h;
// Molar mass of hydrogen in SI (kg.mol^-1)
constexpr real_t M_h     = m_h * Na;
// Specific gas constant of hydrogen in SI (J.kg^-1.K^-1)
constexpr real_t Rstar_h = R / M_h;

}}}

namespace euler_kokkos{ namespace cgs_units
{

constexpr real_t length_u          = static_cast<real_t>(1.0e-2l); // in SI (m)
constexpr real_t mass_u            = static_cast<real_t>(1.0e-3l); // in SI (kg)
constexpr real_t time_u            = ONE_F; // in SI (s)
constexpr real_t temperature_u     = ONE_F; // in SI (K)
constexpr real_t matter_quantity_u = ONE_F; // in SI (mol)

}}

namespace euler_kokkos{ namespace rayleigh_benard_units
{

namespace si_constants = si_units::constants;

constexpr real_t length_u          = ONE_F; // in SI (m)
constexpr real_t mass_u            = ONE_F; // in SI (kg)
constexpr real_t temperature_u     = ONE_F; // in SI (K)
// the mean-molecular weight should appear in the unit time but it is a run-time parameter,
// so to be consistent with the set of parameter the mean molecular weight has to be equal to 1.0 in the parameter file (.ini)
constexpr real_t time_u            = length_u / my_sqrt(si_constants::k_b / (si_constants::mmw_h * si_constants::m_h) * temperature_u); // in SI (s)
constexpr real_t matter_quantity_u = ONE_F; // in SI (mol)

}}

namespace euler_kokkos{ namespace code_units
{

using namespace rayleigh_benard_units;

// Derived units
constexpr real_t volume_u   = length_u * length_u * length_u;
constexpr real_t density_u  = mass_u / volume_u;
constexpr real_t velocity_u = length_u / time_u;
constexpr real_t energy_u   = mass_u * velocity_u * velocity_u;
constexpr real_t pressure_u = energy_u / volume_u;

}}

namespace euler_kokkos{ namespace code_units{ namespace constants
{
namespace si_constants = si_units::constants;

constexpr real_t Na      = si_constants::Na  / (ONE_F / code_units::matter_quantity_u);
constexpr real_t k_b     = si_constants::k_b / (code_units::energy_u / code_units::temperature_u);
constexpr real_t R       = si_constants::R   / (code_units::energy_u / (code_units::matter_quantity_u * code_units::temperature_u));
constexpr real_t mmw_h   = si_constants::mmw_h;
constexpr real_t m_h     = si_constants::m_h / code_units::mass_u;
constexpr real_t M_h     = m_h * Na;
constexpr real_t Rstar_h = R / M_h;

}}}
