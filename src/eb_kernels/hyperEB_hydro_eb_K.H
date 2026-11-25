#ifndef HYPEREB_HYDRO_EB_K_H_
#define HYPEREB_HYDRO_EB_K_H_

#include "hyperEB_index_macros.H"
#include "hyperEB_parm.H"
#include "hyperEB_hydro_K.H"

#include <AMReX_EBCellFlag.H>
#include <AMReX_FArrayBox.H>

#include <cmath>

// ======================================================
// EB hyperbolic reconstruction interface
//
// Characteristic Roe MUSCL, Van Leer slopes, and HLL fluxes are shared
// with regular faces. Closed faces have zero flux. An unavailable connected
// outer stencil leaves that side cell-centered; invalid fluid states are fatal.
// ======================================================

AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
hypereb_zero_eb_flux_x(
    int i, int j, int k,
    amrex::Array4<amrex::Real> const &fx) noexcept
{
    using amrex::Real;

    fx(i, j, k, URHO)  = Real(0.0);
    fx(i, j, k, UMX)   = Real(0.0);
    fx(i, j, k, UMY)   = Real(0.0);

#if (AMREX_SPACEDIM == 3)
    fx(i, j, k, UMZ)   = Real(0.0);
#endif

    fx(i, j, k, UEDEN) = Real(0.0);
}


AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
hypereb_zero_eb_flux_y(
    int i, int j, int k,
    amrex::Array4<amrex::Real> const &fy) noexcept
{
    using amrex::Real;

    fy(i, j, k, URHO)  = Real(0.0);
    fy(i, j, k, UMY)   = Real(0.0);
    fy(i, j, k, UMX)   = Real(0.0);

#if (AMREX_SPACEDIM == 3)
    fy(i, j, k, UMZ)   = Real(0.0);
#endif

    fy(i, j, k, UEDEN) = Real(0.0);
}


#if (AMREX_SPACEDIM == 3)

AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
hypereb_zero_eb_flux_z(
    int i, int j, int k,
    amrex::Array4<amrex::Real> const &fz) noexcept
{
    using amrex::Real;

    fz(i, j, k, URHO)  = Real(0.0);
    fz(i, j, k, UMZ)   = Real(0.0);
    fz(i, j, k, UMX)   = Real(0.0);
    fz(i, j, k, UMY)   = Real(0.0);
    fz(i, j, k, UEDEN) = Real(0.0);
}

#endif


// ======================================================
// EB Cartesian-face connectivity
//
// Hyperbolic face fluxes are evaluated on a wider box than
// the divergence box because the cut-face interpolation uses
// transverse neighbors. EBCellFlag belongs to the basic EB
// data and is available on that wider support. Area fractions
// belong to full EB data and are intentionally only used later,
// inside the divergence kernel where their support is guaranteed.
// ======================================================

AMREX_GPU_DEVICE AMREX_FORCE_INLINE bool
hypereb_eb_face_open_x(
    int i, int j, int k,
    amrex::Array4<amrex::EBCellFlag const> const &flag) noexcept
{
    return flag(i - 1, j, k).isConnected(1, 0, 0);
}


AMREX_GPU_DEVICE AMREX_FORCE_INLINE bool
hypereb_eb_face_open_y(
    int i, int j, int k,
    amrex::Array4<amrex::EBCellFlag const> const &flag) noexcept
{
    return flag(i, j - 1, k).isConnected(0, 1, 0);
}


#if (AMREX_SPACEDIM == 3)

AMREX_GPU_DEVICE AMREX_FORCE_INLINE bool
hypereb_eb_face_open_z(
    int i, int j, int k,
    amrex::Array4<amrex::EBCellFlag const> const &flag) noexcept
{
    return flag(i, j, k - 1).isConnected(0, 0, 1);
}

#endif


// ======================================================
// EB MUSCL stencil availability
//
// The old EB PLM implementation required both
// directional neighbors of a cell to be connected before
// assigning a second-order slope.
//
// The face-based reconstruction preserves exactly that
// intent:
//
// X face i:
//
//   left state belongs to cell i-1
//   right state belongs to cell i
//
// left_stencil_ok:
//   cell i-1 connected to i-2 and i
//
// right_stencil_ok:
//   cell i connected to i-1 and i+1
//
// Analogous definitions are used for Y and Z.
// ======================================================

AMREX_GPU_DEVICE AMREX_FORCE_INLINE bool
hypereb_eb_left_stencil_x(
    int i, int j, int k,
    amrex::Array4<amrex::EBCellFlag const> const &flag) noexcept
{
    return
        flag(i - 1, j, k).isConnected(-1, 0, 0) &&
        flag(i - 1, j, k).isConnected( 1, 0, 0);
}


AMREX_GPU_DEVICE AMREX_FORCE_INLINE bool
hypereb_eb_right_stencil_x(
    int i, int j, int k,
    amrex::Array4<amrex::EBCellFlag const> const &flag) noexcept
{
    return
        flag(i, j, k).isConnected(-1, 0, 0) &&
        flag(i, j, k).isConnected( 1, 0, 0);
}


AMREX_GPU_DEVICE AMREX_FORCE_INLINE bool
hypereb_eb_left_stencil_y(
    int i, int j, int k,
    amrex::Array4<amrex::EBCellFlag const> const &flag) noexcept
{
    return
        flag(i, j - 1, k).isConnected(0, -1, 0) &&
        flag(i, j - 1, k).isConnected(0,  1, 0);
}


AMREX_GPU_DEVICE AMREX_FORCE_INLINE bool
hypereb_eb_right_stencil_y(
    int i, int j, int k,
    amrex::Array4<amrex::EBCellFlag const> const &flag) noexcept
{
    return
        flag(i, j, k).isConnected(0, -1, 0) &&
        flag(i, j, k).isConnected(0,  1, 0);
}


#if (AMREX_SPACEDIM == 3)

AMREX_GPU_DEVICE AMREX_FORCE_INLINE bool
hypereb_eb_left_stencil_z(
    int i, int j, int k,
    amrex::Array4<amrex::EBCellFlag const> const &flag) noexcept
{
    return
        flag(i, j, k - 1).isConnected(0, 0, -1) &&
        flag(i, j, k - 1).isConnected(0, 0,  1);
}


AMREX_GPU_DEVICE AMREX_FORCE_INLINE bool
hypereb_eb_right_stencil_z(
    int i, int j, int k,
    amrex::Array4<amrex::EBCellFlag const> const &flag) noexcept
{
    return
        flag(i, j, k).isConnected(0, 0, -1) &&
        flag(i, j, k).isConnected(0, 0,  1);
}

#endif


// ======================================================
// Second-order EB MUSCL face fluxes
//
// These functions only determine EB-specific geometric
// stencil availability.
//
// Reconstruction itself is delegated to:
//
//   hypereb_muscl_flux_x/y/z()
//
// in hyperEB_hydro_K.H.
//
// Both regular and EB faces use the same:
//
//   characteristic Van Leer limiting
//   final admissibility check
//
// as the Regular path.
// ======================================================

AMREX_GPU_DEVICE AMREX_FORCE_INLINE int
hypereb_eb_muscl_flux_x(
    int i, int j, int k,
    amrex::Array4<amrex::Real> const &fx,
    amrex::Array4<amrex::Real const> const &u,
    amrex::Array4<amrex::Real const> const &q,
    amrex::Array4<amrex::EBCellFlag const> const &flag,
    Parm const &parm) noexcept
{
    if (!hypereb_eb_face_open_x(i, j, k, flag))
    {
        hypereb_zero_eb_flux_x(i, j, k, fx);
        return hypereb_reconstruction_ok;
    }

    const bool left_stencil_ok =
        hypereb_eb_left_stencil_x(
            i, j, k, flag);

    const bool right_stencil_ok =
        hypereb_eb_right_stencil_x(
            i, j, k, flag);

    return hypereb_muscl_flux_x(
        i, j, k,
        fx,
        u,
        q,
        parm,
        left_stencil_ok,
        right_stencil_ok);
}


AMREX_GPU_DEVICE AMREX_FORCE_INLINE int
hypereb_eb_muscl_flux_y(
    int i, int j, int k,
    amrex::Array4<amrex::Real> const &fy,
    amrex::Array4<amrex::Real const> const &u,
    amrex::Array4<amrex::Real const> const &q,
    amrex::Array4<amrex::EBCellFlag const> const &flag,
    Parm const &parm) noexcept
{
    if (!hypereb_eb_face_open_y(i, j, k, flag))
    {
        hypereb_zero_eb_flux_y(i, j, k, fy);
        return hypereb_reconstruction_ok;
    }

    const bool left_stencil_ok =
        hypereb_eb_left_stencil_y(
            i, j, k, flag);

    const bool right_stencil_ok =
        hypereb_eb_right_stencil_y(
            i, j, k, flag);

    return hypereb_muscl_flux_y(
        i, j, k,
        fy,
        u,
        q,
        parm,
        left_stencil_ok,
        right_stencil_ok);
}


#if (AMREX_SPACEDIM == 3)

AMREX_GPU_DEVICE AMREX_FORCE_INLINE int
hypereb_eb_muscl_flux_z(
    int i, int j, int k,
    amrex::Array4<amrex::Real> const &fz,
    amrex::Array4<amrex::Real const> const &u,
    amrex::Array4<amrex::Real const> const &q,
    amrex::Array4<amrex::EBCellFlag const> const &flag,
    Parm const &parm) noexcept
{
    if (!hypereb_eb_face_open_z(i, j, k, flag))
    {
        hypereb_zero_eb_flux_z(i, j, k, fz);
        return hypereb_reconstruction_ok;
    }

    const bool left_stencil_ok =
        hypereb_eb_left_stencil_z(
            i, j, k, flag);

    const bool right_stencil_ok =
        hypereb_eb_right_stencil_z(
            i, j, k, flag);

    return hypereb_muscl_flux_z(
        i, j, k,
        fz,
        u,
        q,
        parm,
        left_stencil_ok,
        right_stencil_ok);
}

#endif


#endif
