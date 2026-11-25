#ifndef HYPEREB_HYPEREB_MOVING_H_
#define HYPEREB_HYPEREB_MOVING_H_

#include <AMReX_Array4.H>
#include <AMReX_Box.H>
#include <AMReX_EBCellFlag.H>
#include <AMReX_GpuContainers.H>

namespace hypereb::kernels {

void zero_out_solid_cells(
    const amrex::Box& box,
    const amrex::Array4<amrex::Real>& state,
    int components,
    const amrex::Array4<const amrex::Real>& volume_fraction,
    amrex::Real ambient_density);

void prepare_uncovered_cells(
    const amrex::Box& box,
    const amrex::Array4<const amrex::Real>& old_volume_fraction,
    const amrex::Array4<const amrex::Real>& new_volume_fraction,
    const amrex::Array4<const amrex::EBCellFlag>& new_flags,
    const amrex::Array4<const amrex::Real>& new_cell_centroid,
    const amrex::Array4<amrex::Real>& state,
    const amrex::Array4<amrex::Real>& conservative_adjustment,
    const amrex::Array4<amrex::Real>& donor_tag,
    const amrex::GpuArray<amrex::Real,3>& cell_size,
    amrex::Real gamma,
    amrex::Real cv,
    int level,
    amrex::Real time,
    int moving_step,
    int components);

void apply_uncovered_cells(
    const amrex::Box& box,
    const amrex::Array4<amrex::Real>& state,
    const amrex::Array4<const amrex::Real>& conservative_adjustment,
    const amrex::Array4<const amrex::Real>& donor_tag,
    int components);

} // namespace hypereb::kernels

#endif
