#ifndef HYPEREB_HYDRO_K_H_
#define HYPEREB_HYDRO_K_H_

#include "hyperEB_index_macros.H"
#include "hyperEB_parm.H"
#include "hyperEB_eb_gradient_K.H"
#include <AMReX_EBCellFlag.H>
#include <AMReX_FArrayBox.H>
#include <cmath>
#include <cstdio>
#include <limits>

AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
hypereb_ctoprim(int i, int j, int k,
            amrex::Array4<amrex::Real const> const &u,
            amrex::Array4<amrex::Real> const &q,
            Parm const &parm) noexcept
{
    using amrex::Real;

    Real rho = u(i, j, k, URHO);
    if (!std::isfinite(rho) || !(rho >= parm.smallr))
    {
        std::printf("Invalid primitive input cell=(%d,%d,%d) rho=%.17g volume=%.17g\n",
                    i, j, k, double(rho), double(u(i,j,k,7)));
        std::fflush(stdout);
        amrex::Abort("Invalid density in hypereb_ctoprim.");
    }
    Real rhoinv = Real(1.0) / rho;
    AMREX_D_TERM(Real ux = u(i, j, k, UMX) * rhoinv;,
                                                    Real uy = u(i, j, k, UMY) * rhoinv;
                 ,
                 Real uz = u(i, j, k, UMZ) * rhoinv;);
#if (AMREX_SPACEDIM == 2)
    Real kineng = Real(0.5) * rho * (ux * ux + uy * uy);
#else
    Real kineng = Real(0.5) * rho * (ux * ux + uy * uy + uz * uz);
#endif
    Real ei = u(i, j, k, UEDEN) - kineng;
    Real p = (parm.eos_gamma - Real(1.0)) * ei;
    if (!std::isfinite(p) || !(p >= parm.smallp))
    {
        amrex::Abort("Invalid total-energy state in hypereb_ctoprim.");
    }

    // q(i,j,k,QEINT) is e, not (rho e)
    ei *= rhoinv;

    q(i, j, k, QRHO) = rho;
    AMREX_D_TERM(q(i, j, k, QU) = ux;,
                                     q(i, j, k, QV) = uy;
                 ,
                 q(i, j, k, QW) = uz;);
    q(i, j, k, QEINT) = ei;
    q(i, j, k, QPRES) = p;
    q(i, j, k, QCS) = std::sqrt(parm.eos_gamma * p * rhoinv);
    q(i, j, k, QTEMP) = ei / parm.cv;
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
hypereb_ctoprim(int i, int j, int k,
            amrex::Array4<amrex::Real const> const &u,
            amrex::Array4<amrex::Real> const &q,
            Parm const &parm,
            amrex::Array4<amrex::EBCellFlag const> const &flags) noexcept
{
    using amrex::Real;
    AMREX_ALWAYS_ASSERT(flags.contains(i, j, k));
    if (flags(i, j, k).isCovered())
    {
        // Scratch only: covered conservative storage is not a fluid state.
        // Initialize every primitive for whole-box coefficient/finite checks.
        for (int n = 0; n < NPRIM; ++n)
        {
            q(i, j, k, n) = Real(0.0);
        }
        const Real temperature = Real(300.0);
        const Real internal_energy = parm.cv * temperature;
        const Real pressure = (parm.eos_gamma - Real(1.0)) * internal_energy;
        q(i, j, k, QRHO) = Real(1.0);
        q(i, j, k, QEINT) = internal_energy;
        q(i, j, k, QPRES) = pressure;
        q(i, j, k, QCS) = std::sqrt(parm.eos_gamma * pressure);
        q(i, j, k, QTEMP) = temperature;
        return;
    }
    hypereb_ctoprim(i, j, k, u, q, parm);
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
hypereb_flux_to_dudt(int i, int j, int k, int n,
                 amrex::Array4<amrex::Real> const &dudt,
                 AMREX_D_DECL(amrex::Array4<amrex::Real const> const &fx,
                              amrex::Array4<amrex::Real const> const &fy,
                              amrex::Array4<amrex::Real const> const &fz),
                 amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dxinv) noexcept
{
#if (AMREX_SPACEDIM == 2)
    dudt(i, j, k, n) = dxinv[0] * (fx(i, j, k, n) - fx(i + 1, j, k, n)) + dxinv[1] * (fy(i, j, k, n) - fy(i, j + 1, k, n));
#else
    dudt(i, j, k, n) = dxinv[0] * (fx(i, j, k, n) - fx(i + 1, j, k, n)) + dxinv[1] * (fy(i, j, k, n) - fy(i, j + 1, k, n)) + dxinv[2] * (fz(i, j, k, n) - fz(i, j, k + 1, n));
#endif
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE amrex::Real
hypereb_limiter(amrex::Real dlft, amrex::Real drgt) noexcept
{
    using amrex::Real;
    if ((dlft <= Real(0.0) && drgt >= Real(0.0)) ||
        (dlft >= Real(0.0) && drgt <= Real(0.0)))
    {
        return Real(0.0);
    }

    // Scale the Van Leer harmonic mean to avoid overflowing the product.
    const Real scale = amrex::max(std::abs(dlft), std::abs(drgt));
    const Real a = dlft / scale;
    const Real b = drgt / scale;
    return scale * (Real(2.0) * a * b / (a + b));
}

// ======================================================
// MUSCL reconstruction support
//
// Status values returned by the face reconstruction path:
//   0 : high-order/normal state is valid
//   2 : unrecoverable/fatal reconstruction state
//
// ======================================================

constexpr int hypereb_reconstruction_ok = 0;
constexpr int hypereb_reconstruction_fatal = 2;

struct HyperEBLocalConsState
{
    amrex::Real rho;
    amrex::Real mn;
    amrex::Real mt1;
    amrex::Real mt2;
    amrex::Real eden;
};

struct HyperEBLocalPrimState
{
    amrex::Real rho;
    amrex::Real un;
    amrex::Real ut1;
    amrex::Real ut2;
    amrex::Real p;
};

struct HyperEBRoeState
{
    amrex::Real rho;
    amrex::Real un;
    amrex::Real ut1;
    amrex::Real ut2;
    amrex::Real H;
    amrex::Real a;
};

struct HyperEBCharacteristicJump
{
    amrex::Real acoustic_minus;
    amrex::Real contact;
    amrex::Real shear1;
    amrex::Real shear2;
    amrex::Real acoustic_plus;
};

AMREX_GPU_DEVICE AMREX_FORCE_INLINE bool
hypereb_cons_state_finite(HyperEBLocalConsState const &s) noexcept
{
    return std::isfinite(s.rho) &&
           std::isfinite(s.mn) &&
           std::isfinite(s.mt1) &&
           std::isfinite(s.mt2) &&
           std::isfinite(s.eden);
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE bool
hypereb_prim_state_finite(HyperEBLocalPrimState const &s) noexcept
{
    return std::isfinite(s.rho) &&
           std::isfinite(s.un) &&
           std::isfinite(s.ut1) &&
           std::isfinite(s.ut2) &&
           std::isfinite(s.p);
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE HyperEBLocalConsState
hypereb_load_local_cons(
    int i, int j, int k, int direction,
    amrex::Array4<amrex::Real const> const &u) noexcept
{
    HyperEBLocalConsState s{};
    s.rho = u(i, j, k, URHO);
    s.eden = u(i, j, k, UEDEN);

    if (direction == 0)
    {
        s.mn = u(i, j, k, UMX);
        s.mt1 = u(i, j, k, UMY);
#if (AMREX_SPACEDIM == 3)
        s.mt2 = u(i, j, k, UMZ);
#else
        s.mt2 = amrex::Real(0.0);
#endif
    }
    else if (direction == 1)
    {
        s.mn = u(i, j, k, UMY);
        s.mt1 = u(i, j, k, UMX);
#if (AMREX_SPACEDIM == 3)
        s.mt2 = u(i, j, k, UMZ);
#else
        s.mt2 = amrex::Real(0.0);
#endif
    }
#if (AMREX_SPACEDIM == 3)
    else
    {
        s.mn = u(i, j, k, UMZ);
        s.mt1 = u(i, j, k, UMX);
        s.mt2 = u(i, j, k, UMY);
    }
#endif

    return s;
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE bool
hypereb_load_local_prim_from_q(
    int i, int j, int k,
    int direction,
    amrex::Array4<amrex::Real const> const &q,
    Parm const &parm,
    HyperEBLocalPrimState &s) noexcept
{
    using amrex::Real;

    s.rho = q(i, j, k, QRHO);
    s.p = q(i, j, k, QPRES);

    if (direction == 0)
    {
        s.un = q(i, j, k, QU);
        s.ut1 = q(i, j, k, QV);
#if (AMREX_SPACEDIM == 3)
        s.ut2 = q(i, j, k, QW);
#else
        s.ut2 = Real(0.0);
#endif
    }
    else if (direction == 1)
    {
        s.un = q(i, j, k, QV);
        s.ut1 = q(i, j, k, QU);
#if (AMREX_SPACEDIM == 3)
        s.ut2 = q(i, j, k, QW);
#else
        s.ut2 = Real(0.0);
#endif
    }
#if (AMREX_SPACEDIM == 3)
    else
    {
        s.un = q(i, j, k, QW);
        s.ut1 = q(i, j, k, QU);
        s.ut2 = q(i, j, k, QV);
    }
#endif

    return hypereb_prim_state_finite(s) &&
           s.rho >= parm.smallr &&
           s.p >= parm.smallp;
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE bool
hypereb_cons_to_prim_local(
    HyperEBLocalConsState const &u,
    Parm const &parm,
    HyperEBLocalPrimState &q) noexcept
{
    using amrex::Real;

    if (!hypereb_cons_state_finite(u) || !(u.rho >= parm.smallr))
    {
        return false;
    }

    const Real rhoinv = Real(1.0) / u.rho;
    const Real un = u.mn * rhoinv;
    const Real ut1 = u.mt1 * rhoinv;
    const Real ut2 = u.mt2 * rhoinv;

    const Real kinetic =
        Real(0.5) * rhoinv *
        (u.mn * u.mn + u.mt1 * u.mt1 + u.mt2 * u.mt2);

    const Real rhoe = u.eden - kinetic;
    const Real p = (parm.eos_gamma - Real(1.0)) * rhoe;

    if (!std::isfinite(un) || !std::isfinite(ut1) ||
        !std::isfinite(ut2) || !std::isfinite(p) ||
        !(p >= parm.smallp))
    {
        return false;
    }

    q.rho = u.rho;
    q.un = un;
    q.ut1 = ut1;
    q.ut2 = ut2;
    q.p = p;
    return true;
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE HyperEBLocalConsState
hypereb_cons_difference(
    HyperEBLocalConsState const &a,
    HyperEBLocalConsState const &b) noexcept
{
    return HyperEBLocalConsState{
        a.rho - b.rho,
        a.mn - b.mn,
        a.mt1 - b.mt1,
        a.mt2 - b.mt2,
        a.eden - b.eden};
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE HyperEBLocalConsState
hypereb_cons_add_scaled(
    HyperEBLocalConsState const &base,
    HyperEBLocalConsState const &delta,
    amrex::Real scale) noexcept
{
    return HyperEBLocalConsState{
        base.rho + scale * delta.rho,
        base.mn + scale * delta.mn,
        base.mt1 + scale * delta.mt1,
        base.mt2 + scale * delta.mt2,
        base.eden + scale * delta.eden};
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE bool
hypereb_build_roe_state(
    HyperEBLocalConsState const &ul,
    HyperEBLocalConsState const &ur,
    HyperEBLocalPrimState const &ql,
    HyperEBLocalPrimState const &qr,
    Parm const &parm,
    HyperEBRoeState &roe) noexcept
{
    using amrex::Real;

    if (!(ql.rho > Real(0.0)) || !(qr.rho > Real(0.0)))
    {
        return false;
    }

    const Real sqrt_l = std::sqrt(ql.rho);
    const Real sqrt_r = std::sqrt(qr.rho);
    const Real denominator = sqrt_l + sqrt_r;

    if (!(denominator > Real(0.0)) || !std::isfinite(denominator))
    {
        return false;
    }

    const Real Hl = (ul.eden + ql.p) / ql.rho;
    const Real Hr = (ur.eden + qr.p) / qr.rho;

    roe.rho = sqrt_l * sqrt_r;
    roe.un = (sqrt_l * ql.un + sqrt_r * qr.un) / denominator;
    roe.ut1 = (sqrt_l * ql.ut1 + sqrt_r * qr.ut1) / denominator;
    roe.ut2 = (sqrt_l * ql.ut2 + sqrt_r * qr.ut2) / denominator;
    roe.H = (sqrt_l * Hl + sqrt_r * Hr) / denominator;

    const Real velocity2 =
        roe.un * roe.un + roe.ut1 * roe.ut1 + roe.ut2 * roe.ut2;
    const Real a2 =
        (parm.eos_gamma - Real(1.0)) *
        (roe.H - Real(0.5) * velocity2);

    if (!std::isfinite(roe.rho) || !std::isfinite(roe.un) ||
        !std::isfinite(roe.ut1) || !std::isfinite(roe.ut2) ||
        !std::isfinite(roe.H) || !std::isfinite(a2) ||
        !(roe.rho > Real(0.0)) || !(a2 > Real(0.0)))
    {
        return false;
    }

    roe.a = std::sqrt(a2);
    return std::isfinite(roe.a) && roe.a > Real(0.0);
}

// Project a conservative jump onto the five Euler characteristic fields
// of the Roe-averaged interface state.
AMREX_GPU_DEVICE AMREX_FORCE_INLINE bool
hypereb_project_characteristic_jump(
    HyperEBLocalConsState const &du,
    HyperEBRoeState const &roe,
    Parm const &parm,
    HyperEBCharacteristicJump &dw) noexcept
{
    using amrex::Real;

    const Real rho = roe.rho;
    const Real a = roe.a;
    const Real a2 = a * a;

    if (!(rho > Real(0.0)) || !(a2 > Real(0.0)) ||
        !std::isfinite(rho) || !std::isfinite(a2))
    {
        return false;
    }

    const Real dun = (du.mn - roe.un * du.rho) / rho;
    const Real dut1 = (du.mt1 - roe.ut1 * du.rho) / rho;
    const Real dut2 = (du.mt2 - roe.ut2 * du.rho) / rho;

    const Real velocity2 =
        roe.un * roe.un + roe.ut1 * roe.ut1 + roe.ut2 * roe.ut2;

    const Real dp =
        (parm.eos_gamma - Real(1.0)) *
        (du.eden - roe.un * du.mn - roe.ut1 * du.mt1 -
         roe.ut2 * du.mt2 + Real(0.5) * velocity2 * du.rho);

    dw.acoustic_minus = (dp - rho * a * dun) / (Real(2.0) * a2);
    dw.contact = du.rho - dp / a2;
    dw.shear1 = rho * dut1;
    dw.shear2 = rho * dut2;
    dw.acoustic_plus = (dp + rho * a * dun) / (Real(2.0) * a2);

    return std::isfinite(dw.acoustic_minus) &&
           std::isfinite(dw.contact) &&
           std::isfinite(dw.shear1) &&
           std::isfinite(dw.shear2) &&
           std::isfinite(dw.acoustic_plus);
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE HyperEBLocalConsState
hypereb_inverse_characteristic_jump(
    HyperEBCharacteristicJump const &dw,
    HyperEBRoeState const &roe) noexcept
{
    using amrex::Real;

    const Real v2 =
        roe.un * roe.un + roe.ut1 * roe.ut1 + roe.ut2 * roe.ut2;

    const Real am = dw.acoustic_minus;
    const Real ac = dw.contact;
    const Real s1 = dw.shear1;
    const Real s2 = dw.shear2;
    const Real ap = dw.acoustic_plus;

    HyperEBLocalConsState du{};
    du.rho = am + ac + ap;
    du.mn =
        am * (roe.un - roe.a) + ac * roe.un +
        ap * (roe.un + roe.a);
    du.mt1 =
        (am + ac + ap) * roe.ut1 + s1;
    du.mt2 =
        (am + ac + ap) * roe.ut2 + s2;
    du.eden =
        am * (roe.H - roe.un * roe.a) +
        ac * Real(0.5) * v2 +
        s1 * roe.ut1 + s2 * roe.ut2 +
        ap * (roe.H + roe.un * roe.a);

    return du;
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE bool
hypereb_characteristic_limited_slope(
    HyperEBLocalConsState const &um,
    HyperEBLocalConsState const &u0,
    HyperEBLocalConsState const &up,
    HyperEBRoeState const &roe,
    Parm const &parm,
    HyperEBLocalConsState &slope) noexcept
{
    const HyperEBLocalConsState dl = hypereb_cons_difference(u0, um);
    const HyperEBLocalConsState dr = hypereb_cons_difference(up, u0);

    HyperEBCharacteristicJump wl{};
    HyperEBCharacteristicJump wr{};

    if (!hypereb_project_characteristic_jump(dl, roe, parm, wl) ||
        !hypereb_project_characteristic_jump(dr, roe, parm, wr))
    {
        return false;
    }

    HyperEBCharacteristicJump limited{};
    limited.acoustic_minus = hypereb_limiter(
        wl.acoustic_minus, wr.acoustic_minus);
    limited.contact = hypereb_limiter(
        wl.contact, wr.contact);
    limited.shear1 = hypereb_limiter(
        wl.shear1, wr.shear1);
    limited.shear2 = hypereb_limiter(
        wl.shear2, wr.shear2);
    limited.acoustic_plus = hypereb_limiter(
        wl.acoustic_plus, wr.acoustic_plus);

    slope = hypereb_inverse_characteristic_jump(limited, roe);
    return hypereb_cons_state_finite(slope);
}

// Shared face-state reconstruction used by Regular and EB paths.
// For Regular faces both stencil flags are true.  EB may set either flag
// false when the corresponding outer neighbor is not geometrically
// connected; that side then uses the cell-centered state.
AMREX_GPU_DEVICE AMREX_FORCE_INLINE int
hypereb_reconstruct_muscl_face_states(
    int i, int j, int k,
    int direction,
    amrex::Array4<amrex::Real const> const &u,
    amrex::Array4<amrex::Real const> const &q,
    Parm const &parm,
    bool left_stencil_ok,
    bool right_stencil_ok,
    HyperEBLocalPrimState &ql_face,
    HyperEBLocalPrimState &qr_face) noexcept
{
    using amrex::Real;

    const int di = direction == 0 ? 1 : 0;
    const int dj = direction == 1 ? 1 : 0;
#if (AMREX_SPACEDIM == 3)
    const int dk = direction == 2 ? 1 : 0;
#else
    const int dk = 0;
#endif

    const int il = i - di;
    const int jl = j - dj;
    const int kl = k - dk;

    HyperEBLocalConsState ul =
        hypereb_load_local_cons(il, jl, kl, direction, u);
    HyperEBLocalConsState ur =
        hypereb_load_local_cons(i, j, k, direction, u);

    HyperEBLocalPrimState ql_cell{};
    HyperEBLocalPrimState qr_cell{};

    // --------------------------------------------------
    // Face-adjacent cell states
    //
    // Both primitive and total-energy states must be admissible.
    // --------------------------------------------------

    // A non-finite conservative state or invalid density is
    // not recoverable by reconstruction.
    if (!hypereb_cons_state_finite(ul) ||
        !hypereb_cons_state_finite(ur) ||
        !(ul.rho >= parm.smallr) ||
        !(ur.rho >= parm.smallr))
    {
        return hypereb_reconstruction_fatal;
    }

    // Load the canonical primitive states from q.
    if (!hypereb_load_local_prim_from_q(
            il, jl, kl, direction,
            q, parm, ql_cell) ||
        !hypereb_load_local_prim_from_q(
            i, j, k, direction,
            q, parm, qr_cell))
    {
        return hypereb_reconstruction_fatal;
    }

    // Check whether the conservative total-energy states
    // themselves are admissible.
    HyperEBLocalPrimState ql_cons{};
    HyperEBLocalPrimState qr_cons{};

    const bool left_cons_ok =
        hypereb_cons_to_prim_local(
            ul, parm, ql_cons);

    const bool right_cons_ok =
        hypereb_cons_to_prim_local(
            ur, parm, qr_cons);

    if (!left_cons_ok || !right_cons_ok)
    {
        return hypereb_reconstruction_fatal;
    }

    // --------------------------------------------------
    // Outer MUSCL stencil states
    //
    // Unavailable geometric stencils use the adjacent cell state.
    // Invalid available stencil states are fatal.
    // --------------------------------------------------

    bool use_left_stencil = left_stencil_ok;
    bool use_right_stencil = right_stencil_ok;

    HyperEBLocalConsState ull = ul;
    HyperEBLocalConsState urr = ur;

    HyperEBLocalPrimState qll = ql_cell;
    HyperEBLocalPrimState qrr = qr_cell;

    if (use_left_stencil)
    {
        const int ill = i - 2 * di;
        const int jll = j - 2 * dj;
        const int kll = k - 2 * dk;

        ull = hypereb_load_local_cons(
            ill, jll, kll,
            direction, u);

        HyperEBLocalPrimState qll_cons{};

        const bool outer_cons_ok =
            hypereb_cons_state_finite(ull) &&
            (ull.rho >= parm.smallr) &&
            hypereb_cons_to_prim_local(
                ull, parm, qll_cons);

        const bool outer_q_ok =
            hypereb_load_local_prim_from_q(
                ill, jll, kll,
                direction, q, parm, qll);

        if (!outer_cons_ok || !outer_q_ok)
        {
            return hypereb_reconstruction_fatal;
        }
    }

    if (use_right_stencil)
    {
        const int irr = i + di;
        const int jrr = j + dj;
        const int krr = k + dk;

        urr = hypereb_load_local_cons(
            irr, jrr, krr,
            direction, u);

        HyperEBLocalPrimState qrr_cons{};

        const bool outer_cons_ok =
            hypereb_cons_state_finite(urr) &&
            (urr.rho >= parm.smallr) &&
            hypereb_cons_to_prim_local(
                urr, parm, qrr_cons);

        const bool outer_q_ok =
            hypereb_load_local_prim_from_q(
                irr, jrr, krr,
                direction, q, parm, qrr);

        if (!outer_cons_ok || !outer_q_ok)
        {
            return hypereb_reconstruction_fatal;
        }
    }

    HyperEBLocalConsState ul_face = ul;
    HyperEBLocalConsState ur_face = ur;

    HyperEBRoeState roe{};
    if (!hypereb_build_roe_state(ul, ur, ql_cell, qr_cell, parm, roe))
    {
        return hypereb_reconstruction_fatal;
    }

    if (use_left_stencil)
    {
        HyperEBLocalConsState slope{};
        if (!hypereb_characteristic_limited_slope(
                ull, ul, ur, roe, parm, slope))
        {
            return hypereb_reconstruction_fatal;
        }
        ul_face = hypereb_cons_add_scaled(ul, slope, Real(0.5));
    }

    if (use_right_stencil)
    {
        HyperEBLocalConsState slope{};
        if (!hypereb_characteristic_limited_slope(
                ul, ur, urr, roe, parm, slope))
        {
            return hypereb_reconstruction_fatal;
        }
        ur_face = hypereb_cons_add_scaled(ur, slope, Real(-0.5));
    }

    // Invalid reconstructed states are fatal; no state repair is applied.
    if (!hypereb_cons_to_prim_local(ul_face, parm, ql_face) ||
        !hypereb_cons_to_prim_local(ur_face, parm, qr_face))
    {
        return hypereb_reconstruction_fatal;
    }

    return hypereb_reconstruction_ok;
}


AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
hll_flux(
    const amrex::Real gamma,
    const amrex::Real rl, const amrex::Real ul, const amrex::Real pl,
    const amrex::Real ut1l, const amrex::Real ut2l,
    const amrex::Real rr, const amrex::Real ur, const amrex::Real pr,
    const amrex::Real ut1r, const amrex::Real ut2r,
    amrex::Real &flxrho, amrex::Real &flxu,
    amrex::Real &flxut,
#if (AMREX_SPACEDIM == 3)
    amrex::Real &flxutt,
#endif
    amrex::Real &flxe) noexcept
{
    using amrex::Real;

#if (AMREX_SPACEDIM == 2)
    amrex::ignore_unused(ut2l, ut2r);
#endif

    const Real al = std::sqrt(gamma * pl / rl);
    const Real ar = std::sqrt(gamma * pr / rr);
    const Real sl = amrex::min(ul - al, ur - ar);
    const Real sr = amrex::max(ul + al, ur + ar);

#if (AMREX_SPACEDIM == 2)
    const Real eden_l = pl / (gamma - Real(1.0))
        + Real(0.5) * rl * (ul * ul + ut1l * ut1l);
    const Real eden_r = pr / (gamma - Real(1.0))
        + Real(0.5) * rr * (ur * ur + ut1r * ut1r);
#else
    const Real eden_l = pl / (gamma - Real(1.0))
        + Real(0.5) * rl * (ul * ul + ut1l * ut1l + ut2l * ut2l);
    const Real eden_r = pr / (gamma - Real(1.0))
        + Real(0.5) * rr * (ur * ur + ut1r * ut1r + ut2r * ut2r);
#endif

    const Real fl_rho = rl * ul;
    const Real fr_rho = rr * ur;
    const Real fl_u = rl * ul * ul + pl;
    const Real fr_u = rr * ur * ur + pr;
    const Real fl_ut = rl * ul * ut1l;
    const Real fr_ut = rr * ur * ut1r;
#if (AMREX_SPACEDIM == 3)
    const Real fl_utt = rl * ul * ut2l;
    const Real fr_utt = rr * ur * ut2r;
#endif
    const Real fl_e = ul * (eden_l + pl);
    const Real fr_e = ur * (eden_r + pr);

    if (sl >= Real(0.0))
    {
        flxrho = fl_rho;
        flxu = fl_u;
        flxut = fl_ut;
#if (AMREX_SPACEDIM == 3)
        flxutt = fl_utt;
#endif
        flxe = fl_e;
        return;
    }
    if (sr <= Real(0.0))
    {
        flxrho = fr_rho;
        flxu = fr_u;
        flxut = fr_ut;
#if (AMREX_SPACEDIM == 3)
        flxutt = fr_utt;
#endif
        flxe = fr_e;
        return;
    }

    const Real denominator = sr - sl;
    if (!(denominator > Real(0.0)) || !std::isfinite(denominator))
    {
        amrex::Abort("Invalid HLL intermediate state.");
        return;
    }

    const Real inverse = Real(1.0) / denominator;
    flxrho = (sr * fl_rho - sl * fr_rho
              + sl * sr * (rr - rl)) * inverse;
    flxu = (sr * fl_u - sl * fr_u
            + sl * sr * (rr * ur - rl * ul)) * inverse;
    flxut = (sr * fl_ut - sl * fr_ut
             + sl * sr * (rr * ut1r - rl * ut1l)) * inverse;
#if (AMREX_SPACEDIM == 3)
    flxutt = (sr * fl_utt - sl * fr_utt
              + sl * sr * (rr * ut2r - rl * ut2l)) * inverse;
#endif
    flxe = (sr * fl_e - sl * fr_e
            + sl * sr * (eden_r - eden_l)) * inverse;
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
hypereb_inviscid_flux(Parm const &parm,
                  amrex::Real rl, amrex::Real ul, amrex::Real pl,
                  amrex::Real ut1l, amrex::Real ut2l,
                  amrex::Real rr, amrex::Real ur, amrex::Real pr,
                  amrex::Real ut1r, amrex::Real ut2r,
                  amrex::Real &flxrho, amrex::Real &flxu,
                  amrex::Real &flxut,
#if (AMREX_SPACEDIM == 3)
                  amrex::Real &flxutt,
#endif
                  amrex::Real &flxe) noexcept
{
    hll_flux(
        parm.eos_gamma,
        rl, ul, pl, ut1l, ut2l,
        rr, ur, pr, ut1r, ut2r,
        flxrho, flxu, flxut,
#if (AMREX_SPACEDIM == 3)
        flxutt,
#endif
        flxe);
}

// ======================================================
// Second-order MUSCL face fluxes
//
// Return value:
//   0 normal/high-order flux
//   2 fatal reconstruction state (caller must report/abort)
//
// left_stencil_ok/right_stencil_ok allow the EB path to degrade one side
// normally to first order when an outer neighbor is not connected.
// ======================================================

AMREX_GPU_DEVICE AMREX_FORCE_INLINE int
hypereb_muscl_flux_x(
    int i, int j, int k,
    amrex::Array4<amrex::Real> const &fx,
    amrex::Array4<amrex::Real const> const &u,
    amrex::Array4<amrex::Real const> const &q,
    Parm const &parm,
    bool left_stencil_ok = true,
    bool right_stencil_ok = true) noexcept
{
    HyperEBLocalPrimState ql{};
    HyperEBLocalPrimState qr{};

    const int status = hypereb_reconstruct_muscl_face_states(
        i, j, k, 0, u, q, parm,
        left_stencil_ok, right_stencil_ok,
        ql, qr);

    if (status == hypereb_reconstruction_fatal)
    {
        fx(i, j, k, URHO) = amrex::Real(0.0);
        fx(i, j, k, UMX) = amrex::Real(0.0);
        fx(i, j, k, UMY) = amrex::Real(0.0);
#if (AMREX_SPACEDIM == 3)
        fx(i, j, k, UMZ) = amrex::Real(0.0);
#endif
        fx(i, j, k, UEDEN) = amrex::Real(0.0);
        return status;
    }


    hypereb_inviscid_flux(
        parm,
        ql.rho, ql.un, ql.p, ql.ut1, ql.ut2,
        qr.rho, qr.un, qr.p, qr.ut1, qr.ut2,
        fx(i, j, k, URHO),
        fx(i, j, k, UMX),
        fx(i, j, k, UMY),
#if (AMREX_SPACEDIM == 3)
        fx(i, j, k, UMZ),
#endif
        fx(i, j, k, UEDEN));

    return status;
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE int
hypereb_muscl_flux_y(
    int i, int j, int k,
    amrex::Array4<amrex::Real> const &fy,
    amrex::Array4<amrex::Real const> const &u,
    amrex::Array4<amrex::Real const> const &q,
    Parm const &parm,
    bool left_stencil_ok = true,
    bool right_stencil_ok = true) noexcept
{
    HyperEBLocalPrimState ql{};
    HyperEBLocalPrimState qr{};

    const int status = hypereb_reconstruct_muscl_face_states(
        i, j, k, 1, u, q, parm,
        left_stencil_ok, right_stencil_ok,
        ql, qr);

    if (status == hypereb_reconstruction_fatal)
    {
        fy(i, j, k, URHO) = amrex::Real(0.0);
        fy(i, j, k, UMY) = amrex::Real(0.0);
        fy(i, j, k, UMX) = amrex::Real(0.0);
#if (AMREX_SPACEDIM == 3)
        fy(i, j, k, UMZ) = amrex::Real(0.0);
#endif
        fy(i, j, k, UEDEN) = amrex::Real(0.0);
        return status;
    }


    hypereb_inviscid_flux(
        parm,
        ql.rho, ql.un, ql.p, ql.ut1, ql.ut2,
        qr.rho, qr.un, qr.p, qr.ut1, qr.ut2,
        fy(i, j, k, URHO),
        fy(i, j, k, UMY),
        fy(i, j, k, UMX),
#if (AMREX_SPACEDIM == 3)
        fy(i, j, k, UMZ),
#endif
        fy(i, j, k, UEDEN));

    return status;
}

#if (AMREX_SPACEDIM == 3)
AMREX_GPU_DEVICE AMREX_FORCE_INLINE int
hypereb_muscl_flux_z(
    int i, int j, int k,
    amrex::Array4<amrex::Real> const &fz,
    amrex::Array4<amrex::Real const> const &u,
    amrex::Array4<amrex::Real const> const &q,
    Parm const &parm,
    bool left_stencil_ok = true,
    bool right_stencil_ok = true) noexcept
{
    HyperEBLocalPrimState ql{};
    HyperEBLocalPrimState qr{};

    const int status = hypereb_reconstruct_muscl_face_states(
        i, j, k, 2, u, q, parm,
        left_stencil_ok, right_stencil_ok,
        ql, qr);

    if (status == hypereb_reconstruction_fatal)
    {
        fz(i, j, k, URHO) = amrex::Real(0.0);
        fz(i, j, k, UMZ) = amrex::Real(0.0);
        fz(i, j, k, UMX) = amrex::Real(0.0);
        fz(i, j, k, UMY) = amrex::Real(0.0);
        fz(i, j, k, UEDEN) = amrex::Real(0.0);
        return status;
    }


    hypereb_inviscid_flux(
        parm,
        ql.rho, ql.un, ql.p, ql.ut1, ql.ut2,
        qr.rho, qr.un, qr.p, qr.ut1, qr.ut2,
        fz(i, j, k, URHO),
        fz(i, j, k, UMZ),
        fz(i, j, k, UMX),
        fz(i, j, k, UMY),
        fz(i, j, k, UEDEN));

    return status;
}
#endif

#if (AMREX_SPACEDIM == 3)

struct HyperEBWallPrimitiveState
{
    amrex::Real density;
    amrex::GpuArray<amrex::Real,3> velocity;
    amrex::Real pressure;
};

AMREX_GPU_DEVICE AMREX_FORCE_INLINE bool
reconstruct_connected_hyp_wall_state(
    int i, int j, int k,
    amrex::Array4<amrex::Real> const& q,
    amrex::Array4<amrex::Real const> const& volume_fraction,
    amrex::Array4<amrex::Real const> const& cell_centroid,
    amrex::Array4<amrex::Real const> const& boundary_centroid,
    amrex::Array4<amrex::EBCellFlag const> const& flags,
    Parm const& parm,
    HyperEBWallPrimitiveState& state) noexcept
{
    using amrex::Real;

    state.density=q(i,j,k,QRHO);
    state.velocity={q(i,j,k,QU),q(i,j,k,QV),q(i,j,k,QW)};
    state.pressure=q(i,j,k,QPRES);
    if (!std::isfinite(state.density) || !std::isfinite(state.pressure)
        || !std::isfinite(state.velocity[0])
        || !std::isfinite(state.velocity[1])
        || !std::isfinite(state.velocity[2])
        || !(state.density >= parm.smallr) || !(state.pressure >= parm.smallp)) {
        amrex::Abort("Invalid central EB wall state.");
    }
    if (!boundary_centroid.contains(i,j,k)
        || !cell_centroid.contains(i,j,k)) {
        return false;
    }

    constexpr int primitive_components[5]={QRHO,QU,QV,QW,QPRES};
    Real candidate[5]{};
    for (int component=0; component<5; ++component) {
        if (!hypereb::kernels::eb_gradient::
                connected_wall_weighted_least_squares_value(
                    i,j,k,primitive_components[component],q,
                    volume_fraction,cell_centroid,boundary_centroid,flags,
                    parm.cell_size,candidate[component])) {
            return false;
        }
        if (!std::isfinite(candidate[component])) {
            amrex::Abort("Non-finite reconstructed EB wall state.");
        }
    }

    if (!(candidate[0] >= parm.smallr) || !(candidate[4] >= parm.smallp)) {
        amrex::Abort("Invalid reconstructed EB wall state.");
    }
    state.density=candidate[0];
    for (int direction=0; direction<3; ++direction) {
        state.velocity[direction]=candidate[direction+1];
    }
    state.pressure=candidate[4];
    return true;
}

#endif

AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
hypereb_wall_velocity(int, int, int,
                  amrex::Array4<amrex::Real const> const &,
                  Parm const &parm,
                  amrex::GpuArray<amrex::Real, 3> &velocity) noexcept
{
    velocity = parm.wall_translation;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
hypereb_symmetric_wall_riemann(
    amrex::Real gamma, amrex::Real rho, amrex::Real pressure,
    amrex::Real relative_normal_velocity,
    amrex::Real& star_density, amrex::Real& star_pressure) noexcept
{
    using amrex::Real;
    if (!std::isfinite(gamma) || !(gamma > Real(1.0)) ||
        !std::isfinite(rho) || !(rho > Real(0.0)) ||
        !std::isfinite(pressure) || !(pressure > Real(0.0)) ||
        !std::isfinite(relative_normal_velocity)) {
        amrex::Abort("Invalid input to symmetric wall Riemann solve.");
    }

    const Real sound_speed = std::sqrt(gamma * pressure / rho);
    const Real mach = relative_normal_velocity / sound_speed;
    const Real gm1 = gamma - Real(1.0);
    if (relative_normal_velocity <= Real(0.0)) {
        // The reflected rarefactions meet at zero wall-frame velocity.
        const Real factor = Real(1.0) + Real(0.5) * gm1 * mach;
        if (factor <= Real(0.0)) {
            // Analytic vacuum, not a pressure/density floor.
            star_density = Real(0.0);
            star_pressure = Real(0.0);
            return;
        }
        const Real ratio = std::pow(factor, Real(2.0) * gamma / gm1);
        star_pressure = pressure * ratio;
        star_density = rho * std::pow(ratio, Real(1.0) / gamma);
    } else {
        // Positive root of the perfect-gas shock relation f(p*) = u_rel.
        const Real gp1 = gamma + Real(1.0);
        const Real mach_squared = mach * mach;
        const Real ratio = Real(1.0) + gamma * gp1 * mach_squared / Real(4.0)
            + gamma * mach * std::sqrt(
                Real(1.0) + gp1 * gp1 * mach_squared / Real(16.0));
        const Real b = gm1 / gp1;
        star_pressure = pressure * ratio;
        star_density = rho * (ratio + b) / (b * ratio + Real(1.0));
    }
    if (!std::isfinite(star_density) || !std::isfinite(star_pressure)) {
        amrex::Abort("Non-finite symmetric wall Riemann solution.");
    }
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
compute_hyp_wallflux(int i, int j, int k,
                     const amrex::Real rho,
                     AMREX_D_DECL(const amrex::Real u, const amrex::Real v, const amrex::Real w),
                     const amrex::Real p, const amrex::Real axm, const amrex::Real axp,
                     const amrex::Real aym, const amrex::Real ayp,
#if (AMREX_SPACEDIM == 3)
                     const amrex::Real azm, const amrex::Real azp,
#endif
                     amrex::Array4<amrex::Real const> const &bcent,
                     amrex::GpuArray<amrex::Real, NEQNS> &fluxw, Parm const &parm,
                     amrex::Real* interface_pressure_output=nullptr) noexcept
{
    using amrex::Real;

    const Real area_x = axm - axp;
    const Real area_y = aym - ayp;
    const Real area_z = azm - azp;
    const Real area_norm = std::sqrt(
        area_x * area_x + area_y * area_y + area_z * area_z);
    const Real inverse_area_norm = Real(1.0) / area_norm;
    const Real normal_x = area_x * inverse_area_norm;
    const Real normal_y = area_y * inverse_area_norm;
    const Real normal_z = area_z * inverse_area_norm;
    const Real fluid_normal_velocity = u * normal_x + v * normal_y + w * normal_z;

    amrex::GpuArray<Real, 3> wall_velocity;
    hypereb_wall_velocity(i, j, k, bcent, parm, wall_velocity);
    const Real wall_normal_velocity = wall_velocity[0] * normal_x + wall_velocity[1] * normal_y + wall_velocity[2] * normal_z;

    Real godunov_density;
    Real riemann_pressure;
    const Real relative_normal_velocity =
        fluid_normal_velocity - wall_normal_velocity;
    hypereb_symmetric_wall_riemann(
        parm.eos_gamma, rho, p, relative_normal_velocity,
        godunov_density, riemann_pressure);
    if (interface_pressure_output != nullptr) {
        *interface_pressure_output=riemann_pressure;
    }

    const Real interface_u = u - fluid_normal_velocity * normal_x + wall_normal_velocity * normal_x;
    const Real interface_v = v - fluid_normal_velocity * normal_y + wall_normal_velocity * normal_y;
    const Real interface_w = w - fluid_normal_velocity * normal_z + wall_normal_velocity * normal_z;
    const Real velocity_dot_area = interface_u * area_x + interface_v * area_y + interface_w * area_z;

    // Interface-state transport through the moving control surface.
    const Real swept_density = godunov_density;
    const Real swept_mass_flux=swept_density*velocity_dot_area;
    fluxw[URHO] = swept_mass_flux;
    fluxw[UMX] = area_x * riemann_pressure+interface_u*swept_mass_flux;
    fluxw[UMY] = area_y * riemann_pressure+interface_v*swept_mass_flux;
    fluxw[UMZ] = area_z * riemann_pressure+interface_w*swept_mass_flux;
    fluxw[UEDEN] = (riemann_pressure
        + riemann_pressure / (parm.eos_gamma - Real(1.0))
        + Real(0.5) * swept_density * (interface_u * interface_u
            + interface_v * interface_v + interface_w * interface_w))
        * velocity_dot_area;
}

#endif
