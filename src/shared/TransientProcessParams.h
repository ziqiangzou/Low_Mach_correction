#pragma once

#include "real_type.h"
#include "utils/config/ConfigMap.h"

struct TransientParams
{
    // blast problem parameters
    real_t transient_fluid_temp;
    real_t transient_temp_top;
    real_t transient_temp_bot;

    TransientParams(ConfigMap& configMap)
    {
        transient_fluid_temp = configMap.getFloat("transient", "fluid_temp", TWO_F);
          transient_temp_top = configMap.getFloat("transient", "temp_top", TWO_F);
        transient_temp_bot = configMap.getFloat("transient", "temp_bot", TWO_F);
    }
};
