#pragma once

#include "shared/real_type.h"
#include "utils/config/ConfigMap.h"

struct DamBreakParams
{
    // dam break problem parameters
    real_t interface_position;

    DamBreakParams(ConfigMap& configMap)
    {
        interface_position = configMap.getFloat("dam_break", "interface_position", 0.5);
    }
}; // struct DamBreakParams
