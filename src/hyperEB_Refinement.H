#ifndef HYPEREB_HYPEREB_REFINEMENT_H_
#define HYPEREB_HYPEREB_REFINEMENT_H_

#include <AMReX_Array4.H>
#include <AMReX_Box.H>
#include <AMReX_EBCellFlag.H>
#include <algorithm>
#include <cmath>

namespace hypereb::refinement
{
// Relative jump, independent of pressure units; never sample through a wall.
inline amrex::Real pressure_jump(
    const amrex::IntVect& cell,
    const amrex::Array4<const amrex::Real>& pressure,
    const amrex::Array4<const amrex::EBCellFlag>& flags,
    const amrex::Box& available_domain,
    amrex::Real pressure_scale_floor = 0.0)
{
    const auto& flag = flags(cell);
    if (!(flag.isRegular() || flag.isSingleValued())) return 0.0;
    const auto pc = pressure(cell);
    if (!(pc > 0.0) || !std::isfinite(pc)) return 0.0;
    amrex::Real sensor = 0.0;
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        for (int sign : {-1, 1})
        {
            amrex::IntVect offset(0);
            offset[d] = sign;
            const auto neighbor = cell + offset;
            if (!available_domain.contains(neighbor) || !flag.isConnected(offset))
                continue;
            const auto& nf = flags(neighbor);
            if (!(nf.isRegular() || nf.isSingleValued())) continue;
            const auto pn = pressure(neighbor);
            if (!(pn > 0.0) || !std::isfinite(pn)) continue;
            const amrex::Real scale = std::max(
                pressure_scale_floor, std::max(pn, pc));
            sensor = std::max(sensor, std::abs(pn - pc) / scale);
        }
    }
    return sensor;
}

}
#endif
