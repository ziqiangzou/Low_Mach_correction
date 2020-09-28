#pragma once

#include "real_type.h"
#include "utils/config/ConfigMap.h"

struct StaticBubbleParams
{
    // blast problem parameters
    real_t rho0;
    real_t rho1;

    StaticBubbleParams(ConfigMap& configMap)
    {
        rho0 = configMap.getFloat("staticbubble", "rho0", 1.);
        rho1 = configMap.getFloat("staticbubble", "rho1", 1.);
    }
};
