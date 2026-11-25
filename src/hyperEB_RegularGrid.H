#ifndef HYPEREB_HYPEREB_REGULAR_H_
#define HYPEREB_HYPEREB_REGULAR_H_

#include <AMReX_Array4.H>
#include <AMReX_Box.H>
#include <AMReX_GpuContainers.H>

#include <array>

namespace hypereb::kernels
{

    enum class EBThermalBoundary
    {
        adiabatic,
        isothermal
    };
    struct RegularPhysics
    {
        amrex::Real gamma;
        amrex::Real cv;
        amrex::Real cp;
        amrex::Real prandtl;
        amrex::Real sutherland_coefficient;
        amrex::Real sutherland_temperature;
        bool use_constant_viscosity;
        amrex::Real constant_mu;
        amrex::Real constant_xi;
        amrex::Real constant_lambda;
        bool viscosity_power_law = false;
        amrex::Real viscosity_reference_mu = 0.0;
        amrex::Real viscosity_reference_temperature = 0.0;
        amrex::Real viscosity_power_exponent = 0.0;
        // Density and pressure thresholds for invalid-state checks, not repairs.
        amrex::Real smallr = 1.0e-19;
        amrex::Real smallp = 1.0e-10;
        bool do_diffusion = true;
        bool eb_wall_diffusion = true;
        EBThermalBoundary eb_thermal_boundary = EBThermalBoundary::adiabatic;
        amrex::Real eb_wall_temperature = 300.0;
        bool eb_noslip_segment = false;
        amrex::GpuArray<amrex::Real, 3> eb_noslip_origin{0.0, 0.0, 0.0};
        amrex::GpuArray<amrex::Real, 3> eb_noslip_tangent{1.0, 0.0, 0.0};
        amrex::Real eb_noslip_s_lo = 0.0;
        amrex::Real eb_noslip_s_hi = 1.0;
        EBThermalBoundary wall_thermal_boundary = EBThermalBoundary::adiabatic;
        amrex::Real wall_temperature = 300.0;
        // Characteristic Roe MUSCL, Van Leer, HLL, and native WSR are fixed.
        static constexpr int srd_max_order = 0;
        static constexpr amrex::Real srd_target_volfrac = 0.5;
        amrex::GpuArray<amrex::Real, 3> problem_low{0.0, 0.0, 0.0};
        amrex::GpuArray<amrex::Real, 3> wall_translation{0.0, 0.0, 0.0};
        amrex::GpuArray<amrex::Real, 3> wall_acceleration{0.0, 0.0, 0.0};
    };

    void compute_regular_rhs(
        const amrex::Box &box,
        const amrex::Array4<amrex::Real> &derivative,
        const amrex::Array4<const amrex::Real> &state,
        const std::array<amrex::Array4<amrex::Real>, 3> &flux,
        const amrex::GpuArray<amrex::Real, 3> &spacing,
        const RegularPhysics &physics);

    RegularPhysics regular_physics();

    // Expensive state-admissibility probes used only for failure diagnosis.
    // Controlled by diagnostics.admissibility and disabled by default.
    bool admissibility_diagnostics_enabled() noexcept;

} // namespace hypereb::kernels

#endif
