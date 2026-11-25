// Source attribution and license obligations are recorded in THIRD_PARTY_NOTICES.md.
#ifndef HYPEREB_HYPEREB_EB_H_
#define HYPEREB_HYPEREB_EB_H_

#include <hyperEB_RegularGrid.H>

#include <AMReX_Array4.H>
#include <AMReX_BCRec.H>
#include <AMReX_EBCellFlag.H>
#include <AMReX_Geometry.H>
#include <AMReX_IntVect.H>
#include <AMReX_MultiFab.H>

#include <array>
#include <limits>

namespace hypereb::kernels {

namespace detail {

// Face-flux storage extends two cells in the face-normal
// direction and three cells in each transverse direction.
inline amrex::Box eb_face_flux_box(const amrex::Box& cell_box, int direction)
{
    amrex::IntVect grow_vector(AMREX_D_DECL(3, 3, 3));
    grow_vector[direction] = 2;
    return amrex::surroundingNodes(amrex::grow(cell_box, grow_vector), direction);
}

} // namespace detail

struct EBGeometryViews {
    amrex::Array4<const amrex::EBCellFlag> flags;
    amrex::Array4<const amrex::Real> volume_fraction;
    amrex::Array4<const amrex::Real> cell_centroid;
    amrex::Array4<const amrex::Real> boundary_centroid;
    std::array<amrex::Array4<const amrex::Real>, 3> area_fraction;
    std::array<amrex::Array4<const amrex::Real>, 3> face_centroid;
};

struct EBCouplingViews {
    int as_coarse = 0;
    amrex::Array4<amrex::Real> coarse_mass_change;
    amrex::Array4<const int> coarse_flag;
    int as_fine = 0;
    amrex::Array4<amrex::Real> fine_mass_change;
    amrex::Array4<const int> level_mask;
    int level_mask_not_covered = 2;
};

struct MovingWallCflEstimate {
    amrex::Real normal_dt = std::numeric_limits<amrex::Real>::max();
    amrex::Real full_dt = std::numeric_limits<amrex::Real>::max();
    amrex::IntVect limiting_cell{AMREX_D_DECL(0, 0, 0)};
    amrex::Real volume_fraction = 0.0;
    amrex::Real sound_speed = 0.0;
    amrex::GpuArray<amrex::Real, 3> fluid_velocity{0.0, 0.0, 0.0};
    amrex::GpuArray<amrex::Real, 3> wall_velocity{0.0, 0.0, 0.0};
    amrex::GpuArray<amrex::Real, 3> eb_normal{0.0, 0.0, 0.0};
    amrex::Real normal_relative_speed = 0.0;
    amrex::Real full_relative_speed = 0.0;
};

MovingWallCflEstimate estimate_moving_wall_cfl(
    const amrex::MultiFab& state,
    const amrex::Geometry& geometry,
    const RegularPhysics& physics,
    amrex::Real time,
    amrex::Real cfl,
    bool collect_details);

enum EBSurfaceLoadComponent {
    SurfacePressure = 0,
    SurfaceTemperature,
    WallTemperature,
    WallHeatFlux,
    HeatTransfer,
    SurfaceNormalX,
    SurfaceNormalY,
    SurfaceNormalZ,
    PressureForceX,
    PressureForceY,
    PressureForceZ,
    ViscousForceX,
    ViscousForceY,
    ViscousForceZ,
    BoundaryArea,
    NumSurfaceLoadComponents
};

void compute_eb_rhs(
    const amrex::Box& box,
    const amrex::Array4<amrex::Real>& derivative,
    const amrex::Array4<const amrex::Real>& state,
    const std::array<amrex::Array4<amrex::Real>, 3>& flux,
    const amrex::Geometry& geometry,
    const RegularPhysics& physics,
    const EBGeometryViews& eb,
    const EBCouplingViews& coupling,
    const amrex::BCRec* boundary_conditions,
    amrex::Real stage_dt,
    amrex::Real reflux_weight,
    amrex::Real time);

void compute_eb_surface_loads(
    const amrex::Box& box,
    const amrex::Array4<amrex::Real>& loads,
    const amrex::Array4<const amrex::Real>& state,
    const amrex::Geometry& geometry,
    const RegularPhysics& physics,
    const EBGeometryViews& eb,
    amrex::Real time);

} // namespace hypereb::kernels

#endif
