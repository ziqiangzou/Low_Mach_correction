#pragma once

#include "real_type.h"
#include "utils/config/ConfigMap.h"

struct SuckingthermParams
{
    // blast problem parameters
    real_t wall_temp;
    real_t time;
    real_t liquid_density;
    real_t vapor_density;

    SuckingthermParams(ConfigMap& configMap)
    {
        wall_temp = configMap.getFloat("suckingtherm", "wall_temp", 383.15);
        time = configMap.getFloat("suckingtherm", "time", 0.1);
        liquid_density = configMap.getFloat("suckingtherm", "liquid_density", 1000.);
        vapor_density = configMap.getFloat("suckingtherm", "vapor_density", 1.);
    }
};
