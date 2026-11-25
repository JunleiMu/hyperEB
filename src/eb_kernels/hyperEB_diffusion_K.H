#ifndef HYPEREB_DIFFUSION_K_H_
#define HYPEREB_DIFFUSION_K_H_

#include "hyperEB_index_macros.H"
#include "hyperEB_parm.H"

#include <AMReX_FArrayBox.H>

#include <cmath>

// Shared by regular/EB diffusion, wall heat flux, diagnostics and the
// explicit viscous time-step estimate. The power law is a material model,
// not a positivity or stability correction; its parameters are explicit.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real hypereb_variable_viscosity(amrex::Real temperature,
                                  Parm const& parm) noexcept
{
    if (parm.viscosity_power_law) {
        return parm.viscosity_reference_mu
            * std::pow(temperature / parm.viscosity_reference_temperature,
                       parm.viscosity_power_exponent);
    }
    return parm.C_S * std::sqrt(temperature) * temperature
        / (temperature + parm.T_S);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void hypereb_transport_coefficients(amrex::Real temperature, bool constant,
                                Parm const& parm, amrex::Real& mu,
                                amrex::Real& xi, amrex::Real& conductivity) noexcept
{
    if (constant)
    {
        mu = parm.const_visc_mu;
        xi = parm.const_visc_ki;
        conductivity = parm.const_lambda;
    }
    else
    {
        mu = hypereb_variable_viscosity(temperature, parm);
        xi = amrex::Real(0.0);
        conductivity = mu * parm.cp / parm.Pr;
    }
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void hypereb_diffcoef(int i, int j, int k,
                  amrex::Array4<amrex::Real const> const& q,
                  amrex::Array4<amrex::Real> const& coefs,
                  Parm const& parm) noexcept
{
    using amrex::Real;
    Real mu, xi, conductivity;
    hypereb_transport_coefficients(q(i,j,k,QTEMP), false, parm, mu, xi, conductivity);
    coefs(i,j,k,CETA) = mu;
    coefs(i,j,k,CXI) = xi;
    coefs(i,j,k,CLAM) = conductivity;
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void hypereb_constcoef(int i, int j, int k,
                   amrex::Array4<amrex::Real> const& coefs,
                   Parm const& parm) noexcept
{
    amrex::Real mu, xi, conductivity;
    hypereb_transport_coefficients(0.0, true, parm, mu, xi, conductivity);
    coefs(i,j,k,CETA) = mu;
    coefs(i,j,k,CXI) = xi;
    coefs(i,j,k,CLAM) = conductivity;
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void hypereb_diff_x(int i, int j, int k,
                amrex::Array4<amrex::Real const> const& q,
                amrex::Array4<amrex::Real const> const& coefs,
                amrex::GpuArray<amrex::Real,3> const& dxinv,
                amrex::Array4<amrex::Real> const& flux) noexcept
{
    using amrex::Real;
    constexpr Real two_thirds = Real(2.0) / Real(3.0);

    const Real dtdx = (q(i,j,k,QTEMP)-q(i-1,j,k,QTEMP))*dxinv[0];
    const Real dudx = (q(i,j,k,QU)-q(i-1,j,k,QU))*dxinv[0];
    const Real dvdx = (q(i,j,k,QV)-q(i-1,j,k,QV))*dxinv[0];
    const Real dwdx = (q(i,j,k,QW)-q(i-1,j,k,QW))*dxinv[0];
    const Real dudy = (q(i,j+1,k,QU)+q(i-1,j+1,k,QU)
                     - q(i,j-1,k,QU)-q(i-1,j-1,k,QU))*(Real(0.25)*dxinv[1]);
    const Real dvdy = (q(i,j+1,k,QV)+q(i-1,j+1,k,QV)
                     - q(i,j-1,k,QV)-q(i-1,j-1,k,QV))*(Real(0.25)*dxinv[1]);
    const Real dudz = (q(i,j,k+1,QU)+q(i-1,j,k+1,QU)
                     - q(i,j,k-1,QU)-q(i-1,j,k-1,QU))*(Real(0.25)*dxinv[2]);
    const Real dwdz = (q(i,j,k+1,QW)+q(i-1,j,k+1,QW)
                     - q(i,j,k-1,QW)-q(i-1,j,k-1,QW))*(Real(0.25)*dxinv[2]);
    const Real divu = dudx + dvdy + dwdz;
    const Real muf = Real(0.5)*(coefs(i,j,k,CETA)+coefs(i-1,j,k,CETA));
    const Real xif = Real(0.5)*(coefs(i,j,k,CXI)+coefs(i-1,j,k,CXI));
    const Real tauxx = muf*(Real(2.0)*dudx-two_thirds*divu) + xif*divu;
    const Real tauxy = muf*(dudy+dvdx);
    const Real tauxz = muf*(dudz+dwdx);

    flux(i,j,k,URHO) = Real(0.0);
    flux(i,j,k,UMX) = -tauxx;
    flux(i,j,k,UMY) = -tauxy;
    flux(i,j,k,UMZ) = -tauxz;
    flux(i,j,k,UEDEN) = -Real(0.5)*(
        (q(i,j,k,QU)+q(i-1,j,k,QU))*tauxx
        +(q(i,j,k,QV)+q(i-1,j,k,QV))*tauxy
        +(q(i,j,k,QW)+q(i-1,j,k,QW))*tauxz
        +(coefs(i,j,k,CLAM)+coefs(i-1,j,k,CLAM))*dtdx);
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void hypereb_diff_y(int i, int j, int k,
                amrex::Array4<amrex::Real const> const& q,
                amrex::Array4<amrex::Real const> const& coefs,
                amrex::GpuArray<amrex::Real,3> const& dxinv,
                amrex::Array4<amrex::Real> const& flux) noexcept
{
    using amrex::Real;
    constexpr Real two_thirds = Real(2.0) / Real(3.0);

    const Real dtdy = (q(i,j,k,QTEMP)-q(i,j-1,k,QTEMP))*dxinv[1];
    const Real dudy = (q(i,j,k,QU)-q(i,j-1,k,QU))*dxinv[1];
    const Real dvdy = (q(i,j,k,QV)-q(i,j-1,k,QV))*dxinv[1];
    const Real dwdy = (q(i,j,k,QW)-q(i,j-1,k,QW))*dxinv[1];
    const Real dudx = (q(i+1,j,k,QU)+q(i+1,j-1,k,QU)
                     - q(i-1,j,k,QU)-q(i-1,j-1,k,QU))*(Real(0.25)*dxinv[0]);
    const Real dvdx = (q(i+1,j,k,QV)+q(i+1,j-1,k,QV)
                     - q(i-1,j,k,QV)-q(i-1,j-1,k,QV))*(Real(0.25)*dxinv[0]);
    const Real dvdz = (q(i,j,k+1,QV)+q(i,j-1,k+1,QV)
                     - q(i,j,k-1,QV)-q(i,j-1,k-1,QV))*(Real(0.25)*dxinv[2]);
    const Real dwdz = (q(i,j,k+1,QW)+q(i,j-1,k+1,QW)
                     - q(i,j,k-1,QW)-q(i,j-1,k-1,QW))*(Real(0.25)*dxinv[2]);
    const Real divu = dudx + dvdy + dwdz;
    const Real muf = Real(0.5)*(coefs(i,j,k,CETA)+coefs(i,j-1,k,CETA));
    const Real xif = Real(0.5)*(coefs(i,j,k,CXI)+coefs(i,j-1,k,CXI));
    const Real tauyy = muf*(Real(2.0)*dvdy-two_thirds*divu) + xif*divu;
    const Real tauxy = muf*(dudy+dvdx);
    const Real tauyz = muf*(dwdy+dvdz);

    flux(i,j,k,URHO) = Real(0.0);
    flux(i,j,k,UMX) = -tauxy;
    flux(i,j,k,UMY) = -tauyy;
    flux(i,j,k,UMZ) = -tauyz;
    flux(i,j,k,UEDEN) = -Real(0.5)*(
        (q(i,j,k,QU)+q(i,j-1,k,QU))*tauxy
        +(q(i,j,k,QV)+q(i,j-1,k,QV))*tauyy
        +(q(i,j,k,QW)+q(i,j-1,k,QW))*tauyz
        +(coefs(i,j,k,CLAM)+coefs(i,j-1,k,CLAM))*dtdy);
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void hypereb_diff_z(int i, int j, int k,
                amrex::Array4<amrex::Real const> const& q,
                amrex::Array4<amrex::Real const> const& coefs,
                amrex::GpuArray<amrex::Real,3> const& dxinv,
                amrex::Array4<amrex::Real> const& flux) noexcept
{
    using amrex::Real;
    constexpr Real two_thirds = Real(2.0) / Real(3.0);

    const Real dtdz = (q(i,j,k,QTEMP)-q(i,j,k-1,QTEMP))*dxinv[2];
    const Real dudz = (q(i,j,k,QU)-q(i,j,k-1,QU))*dxinv[2];
    const Real dvdz = (q(i,j,k,QV)-q(i,j,k-1,QV))*dxinv[2];
    const Real dwdz = (q(i,j,k,QW)-q(i,j,k-1,QW))*dxinv[2];
    const Real dudx = (q(i+1,j,k,QU)+q(i+1,j,k-1,QU)
                     - q(i-1,j,k,QU)-q(i-1,j,k-1,QU))*(Real(0.25)*dxinv[0]);
    const Real dwdx = (q(i+1,j,k,QW)+q(i+1,j,k-1,QW)
                     - q(i-1,j,k,QW)-q(i-1,j,k-1,QW))*(Real(0.25)*dxinv[0]);
    const Real dvdy = (q(i,j+1,k,QV)+q(i,j+1,k-1,QV)
                     - q(i,j-1,k,QV)-q(i,j-1,k-1,QV))*(Real(0.25)*dxinv[1]);
    const Real dwdy = (q(i,j+1,k,QW)+q(i,j+1,k-1,QW)
                     - q(i,j-1,k,QW)-q(i,j-1,k-1,QW))*(Real(0.25)*dxinv[1]);
    const Real divu = dudx + dvdy + dwdz;
    const Real muf = Real(0.5)*(coefs(i,j,k,CETA)+coefs(i,j,k-1,CETA));
    const Real xif = Real(0.5)*(coefs(i,j,k,CXI)+coefs(i,j,k-1,CXI));
    const Real tauxz = muf*(dudz+dwdx);
    const Real tauyz = muf*(dvdz+dwdy);
    const Real tauzz = muf*(Real(2.0)*dwdz-two_thirds*divu) + xif*divu;

    flux(i,j,k,URHO) = Real(0.0);
    flux(i,j,k,UMX) = -tauxz;
    flux(i,j,k,UMY) = -tauyz;
    flux(i,j,k,UMZ) = -tauzz;
    flux(i,j,k,UEDEN) = -Real(0.5)*(
        (q(i,j,k,QU)+q(i,j,k-1,QU))*tauxz
        +(q(i,j,k,QV)+q(i,j,k-1,QV))*tauyz
        +(q(i,j,k,QW)+q(i,j,k-1,QW))*tauzz
        +(coefs(i,j,k,CLAM)+coefs(i,j,k-1,CLAM))*dtdz);
}

#endif
