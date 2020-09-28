#ifndef ISENTROPIC_VORTEX_PARAMS_H_
#define ISENTROPIC_VORTEX_PARAMS_H_

#include "real_type.h"
#include "utils/config/ConfigMap.h"

/**
 * isentropic vortex advection test parameters.
 */
struct IsentropicVortexParams
{
    // isentropic vortex ambient flow
    real_t rho_a;
    real_t p_a;
    real_t T_a;
    real_t u_a;
    real_t v_a;
    real_t w_a;

    // vortex center
    real_t vortex_x;
    real_t vortex_y;
    real_t vortex_z;

    // vortex strength
    real_t beta;

    // number of quadrature points (used to compute initial cell-averaged values)
    int nQuadPts;

    IsentropicVortexParams(ConfigMap& configMap)
    {
        real_t xmin = configMap.getFloat("mesh", "xmin", 0.0);
        real_t ymin = configMap.getFloat("mesh", "ymin", 0.0);
        real_t zmin = configMap.getFloat("mesh", "zmin", 0.0);

        real_t xmax = configMap.getFloat("mesh", "xmax", 1.0);
        real_t ymax = configMap.getFloat("mesh", "ymax", 1.0);
        real_t zmax = configMap.getFloat("mesh", "zmax", 1.0);

        rho_a  = configMap.getFloat("isentropic_vortex","density_ambient", 1.0);
        p_a  = configMap.getFloat("isentropic_vortex","pressure_ambient", 1.0);
        T_a  = configMap.getFloat("isentropic_vortex","temperature_ambient", 1.0);
        u_a  = configMap.getFloat("isentropic_vortex","vx_ambient", 1.0);
        v_a  = configMap.getFloat("isentropic_vortex","vy_ambient", 1.0);
        w_a  = configMap.getFloat("isentropic_vortex","vz_ambient", 1.0);

        vortex_x = configMap.getFloat("isentropic_vortex","center_x", (xmin+xmax)/2);
        vortex_y = configMap.getFloat("isentropic_vortex","center_y", (ymin+ymax)/2);
        vortex_z = configMap.getFloat("isentropic_vortex","center_z", (zmin+zmax)/2);

        beta = configMap.getFloat("isentropic_vortex","strength",5.0);

        nQuadPts = configMap.getInteger("isentropic_vortex", "num_quadrature_points",4);
    }
}; // struct IsentropicVortexParams

#endif // ISENTROPIC_VORTEX_PARAMS_H_
