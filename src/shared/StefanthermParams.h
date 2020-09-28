#pragma once

#include "real_type.h"
#include "utils/config/ConfigMap.h"

struct StefanthermParams
{
    // blast problem parameters
    real_t wall_temp;
    real_t interface_position;
    real_t liquid_density;
    real_t vapor_density;

    StefanthermParams(ConfigMap& configMap)
    {
        wall_temp = configMap.getFloat("stefantherm", "wall_temp", 383.15);
        interface_position = configMap.getFloat("stefantherm", "interface_position", 0.0005);
        liquid_density = configMap.getFloat("stefantherm", "liquid_density", 1000.);
        vapor_density = configMap.getFloat("stefantherm", "vapor_density", 1.);
    }
};
