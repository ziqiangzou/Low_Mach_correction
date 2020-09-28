#pragma once

#include "real_type.h"
#include "utils/config/ConfigMap.h"

struct RayleighBenardParams
{
    // blast problem parameters
    real_t rayleigh_benard_temperature_top;
    real_t rayleigh_benard_temperature_bottom;

    RayleighBenardParams(ConfigMap& configMap)
    {
        rayleigh_benard_temperature_top = configMap.getFloat("rayleigh_benard", "temperature_top", TWO_F);
        rayleigh_benard_temperature_bottom = configMap.getFloat("rayleigh_benard", "temperature_bottom", TWO_F);
    }
};
