#pragma once

#include "real_type.h"
#include "utils/config/ConfigMap.h"

struct NonIsothermParams
{
    // blast problem parameters
    real_t non_isotherm_fluid_temp;
    real_t non_isotherm_temp_top;

    NonIsothermParams(ConfigMap& configMap)
    {
        non_isotherm_fluid_temp = configMap.getFloat("non_isotherm", "fluid_temp", TWO_F);
        non_isotherm_temp_top = configMap.getFloat("non_isotherm", "temp_top", TWO_F);
    }
};
