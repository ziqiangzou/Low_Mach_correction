#pragma once

#include "real_type.h"
#include "utils/config/ConfigMap.h"

#include <string>

struct PoiseuilleParams
{
    real_t poiseuille_mach;
    real_t poiseuille_reynolds;
    real_t poiseuille_velocity_max;
    int poiseuille_flow_direction;
    int poiseuille_normal_direction;

    real_t poiseuille_pressure0;
    real_t poiseuille_pressure1;
    real_t poiseuille_pressure_gradient;
    real_t poiseuille_density;

    PoiseuilleParams(ConfigMap& configMap)
    {
        poiseuille_mach             = configMap.getFloat("poiseuille", "poiseuille_mach", 5.0e-3);
        poiseuille_reynolds         = configMap.getFloat("poiseuille", "poiseuille_reynolds", 1.0);
        poiseuille_velocity_max     = configMap.getFloat("poiseuille", "poiseuille_velocity_max", 1.0e-3);
        poiseuille_flow_direction   = configMap.getInteger("poiseuille", "poiseuille_flow_direction", 0);
        poiseuille_normal_direction = configMap.getInteger("poiseuille", "poiseuille_normal_direction", 1);

        // Default values have to match those defined in HydroParams.cpp
        real_t xmin = configMap.getFloat("mesh", "xmin", 0.0);
        real_t xmax = configMap.getFloat("mesh", "xmax", 1.0);

        real_t ymin = configMap.getFloat("mesh", "ymin", 0.0);
        real_t ymax = configMap.getFloat("mesh", "ymax", 1.0);

        real_t zmin = configMap.getFloat("mesh", "zmin", 0.0);
        real_t zmax = configMap.getFloat("mesh", "zmax", 1.0);

        real_t l;
        switch (poiseuille_flow_direction)
        {
        case 0:
            l = xmax - xmin;
            break;
        case 1:
            l = ymax - ymin;
            break;
        case 2:
            l = zmax - zmin;
        default:
            l = xmax - xmin;
            break;
        }
        real_t h;
        switch (poiseuille_normal_direction)
        {
        case 0:
            h = xmax - xmin;
            break;
        case 1:
            h = ymax - ymin;
            break;
        case 2:
            h = zmax - zmin;
            break;
        default:
            h = ymax - ymin;
            break;
        }

        real_t mu = configMap.getFloat("hydro", "mu", 0.0);
        real_t gamma0 = configMap.getFloat("hydro", "gamma0", 1.4);

        poiseuille_density   = mu * poiseuille_reynolds / (poiseuille_velocity_max * l);
        poiseuille_pressure0 = poiseuille_density * poiseuille_velocity_max * poiseuille_velocity_max / (gamma0 * poiseuille_mach * poiseuille_mach);
        poiseuille_pressure1 = poiseuille_pressure0 - l / (h*h) * 8.0 * mu * poiseuille_velocity_max;
        poiseuille_pressure_gradient = (poiseuille_pressure1 - poiseuille_pressure0) / l;
    }
};
