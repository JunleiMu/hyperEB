#ifndef HYPEREB_PARM_H_
#define HYPEREB_PARM_H_

#include <AMReX_REAL.H>
#include <AMReX_GpuContainers.H>

struct Parm
{
    amrex::Real eos_gamma = 1.4;

    amrex::Real cv;
    amrex::Real cp;

    amrex::Real Pr = 0.72;      // Prandtl number
    amrex::Real C_S = 1.458e-5; // constant in Sutherland's law
    amrex::Real T_S = 110.4;    // Sutherland temperature
    bool viscosity_power_law = false;
    amrex::Real viscosity_reference_mu = 0.0;
    amrex::Real viscosity_reference_temperature = 0.0;
    amrex::Real viscosity_power_exponent = 0.0;

    amrex::Real smallr = 1.e-19;
    amrex::Real smallp = 1.e-10;

    amrex::Real const_visc_mu = -1.0;
    amrex::Real const_visc_ki = -1.0;
    amrex::Real const_lambda = -1.0;

    static constexpr int thermal_adiabatic = 0;
    static constexpr int thermal_isothermal = 1;
    int eb_thermal_boundary = thermal_adiabatic;
    amrex::Real eb_wall_temperature = 300.0;

    bool eb_noslip_segment = false;
    amrex::GpuArray<amrex::Real, 3> eb_noslip_origin{0.0, 0.0, 0.0};
    amrex::GpuArray<amrex::Real, 3> eb_noslip_tangent{1.0, 0.0, 0.0};
    amrex::Real eb_noslip_s_lo = 0.0;
    amrex::Real eb_noslip_s_hi = 1.0;

    amrex::GpuArray<amrex::Real, 3> prob_lo{0.0, 0.0, 0.0};
    amrex::GpuArray<amrex::Real, 3> cell_size{1.0, 1.0, 1.0};
    amrex::GpuArray<amrex::Real, 3> wall_translation{0.0, 0.0, 0.0};

};

#endif
