#pragma once

#include "shared/real_type.h"
#include "utils/config/ConfigMap.h"

struct RiemannProblemParams
{
    // gresho problem parameters
    real_t density_left;
    real_t pressure_left;
    real_t velocity_left;
    real_t density_right;
    real_t pressure_right;
    real_t velocity_right;

    RiemannProblemParams(ConfigMap& configMap)
    {
        density_left   = configMap.getFloat("riemann_problem", "density_left"  , 1.0);
        pressure_left  = configMap.getFloat("riemann_problem", "pressure_left" , 1.0);
        velocity_left  = configMap.getFloat("riemann_problem", "velocity_left" , 0.0);
        density_right  = configMap.getFloat("riemann_problem", "density_right" , 0.125);
        pressure_right = configMap.getFloat("riemann_problem", "pressure_right", 0.1);
        velocity_right = configMap.getFloat("riemann_problem", "velocity_right", 1.0);
    }
}; // struct RiemannProblemParams
