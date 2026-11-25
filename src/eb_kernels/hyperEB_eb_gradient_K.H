#ifndef HYPEREB_HYPEREB_EB_GRADIENT_K_H_
#define HYPEREB_HYPEREB_EB_GRADIENT_K_H_

#include "hyperEB_index_macros.H"

#include <AMReX.H>
#include <AMReX_Array4.H>
#include <AMReX_EBCellFlag.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_GpuQualifiers.H>

#include <cmath>
#include <limits>

namespace hypereb::kernels::eb_gradient {


AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool solve_3x3(
    amrex::Real matrix[3][3], amrex::Real right_hand_side[3],
    amrex::Real solution[3]) noexcept
{
    using amrex::Real;

    Real matrix_scale=Real(0.0);
    for (int row=0; row<3; ++row) {
        if (!std::isfinite(right_hand_side[row])) {
            amrex::Abort("Non-finite wall WLS right-hand side.");
        }
        for (int column=0; column<3; ++column) {
            if (!std::isfinite(matrix[row][column])) {
                amrex::Abort("Non-finite wall WLS matrix.");
            }
            matrix_scale=amrex::max(matrix_scale,std::abs(matrix[row][column]));
        }
    }
    if (matrix_scale==Real(0.0)) {
        return false;
    }

    const Real pivot_tolerance=matrix_scale*Real(1.e-12);
    for (int pivot_column=0; pivot_column<3; ++pivot_column) {
        int pivot_row=pivot_column;
        Real pivot_magnitude=std::abs(matrix[pivot_row][pivot_column]);
        for (int row=pivot_column+1; row<3; ++row) {
            const Real magnitude=std::abs(matrix[row][pivot_column]);
            if (magnitude>pivot_magnitude) {
                pivot_magnitude=magnitude;
                pivot_row=row;
            }
        }
        if (pivot_magnitude<=pivot_tolerance) {
            return false;
        }
        if (pivot_row!=pivot_column) {
            for (int column=0; column<3; ++column) {
                const Real temporary=matrix[pivot_column][column];
                matrix[pivot_column][column]=matrix[pivot_row][column];
                matrix[pivot_row][column]=temporary;
            }
            const Real temporary=right_hand_side[pivot_column];
            right_hand_side[pivot_column]=right_hand_side[pivot_row];
            right_hand_side[pivot_row]=temporary;
        }
        for (int row=pivot_column+1; row<3; ++row) {
            const Real factor=matrix[row][pivot_column]
                /matrix[pivot_column][pivot_column];
            matrix[row][pivot_column]=Real(0.0);
            for (int column=pivot_column+1; column<3; ++column) {
                matrix[row][column] -= factor*matrix[pivot_column][column];
                if (!std::isfinite(matrix[row][column])) {
                    amrex::Abort("Non-finite wall WLS elimination.");
                }
            }
            right_hand_side[row] -= factor*right_hand_side[pivot_column];
            if (!std::isfinite(right_hand_side[row])) {
                amrex::Abort("Non-finite wall WLS elimination.");
            }
        }
    }

    solution[0]=solution[1]=solution[2]=Real(0.0);
    for (int row=2; row>=0; --row) {
        Real value=right_hand_side[row];
        for (int column=row+1; column<3; ++column) {
            value -= matrix[row][column]*solution[column];
        }
        solution[row]=value/matrix[row][row];
    }
    if (!std::isfinite(solution[0]) || !std::isfinite(solution[1])
        || !std::isfinite(solution[2])) {
        amrex::Abort("Non-finite wall WLS solution.");
    }
    return true;
}

// Fit an affine primitive field to the cut cell and its directly connected
// fluid neighbors, then evaluate the fit at the true EB face centroid.  The
// constant term is fitted rather than anchored to the cut-cell value.  This
// distinction is important for a vanishing cut cell: redistribution can leave
// its cell average noticeably noisier than the surrounding full-fluid cells,
// and anchoring a wall extrapolation to that value transfers the small-cell
// error directly into the surface pressure.  Fluid-volume weighting makes the
// operator approach the connected full-cell fit continuously as vfrac -> 0.
template <typename ValueType>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool connected_wall_weighted_least_squares_value(
    int i, int j, int k, int component,
    const amrex::Array4<ValueType>& values,
    const amrex::Array4<const amrex::Real>& volume_fraction,
    const amrex::Array4<const amrex::Real>& cell_centroid,
    const amrex::Array4<const amrex::Real>& boundary_centroid,
    const amrex::Array4<const amrex::EBCellFlag>& flags,
    const amrex::GpuArray<amrex::Real,3>& cell_size,
    amrex::Real& wall_value) noexcept
{
    using amrex::Real;

    if (!values.contains(i,j,k) || !volume_fraction.contains(i,j,k)
        || !cell_centroid.contains(i,j,k)
        || !boundary_centroid.contains(i,j,k) || !flags.contains(i,j,k)
        || volume_fraction(i,j,k)<=Real(0.0)
        || flags(i,j,k).isCovered()) {
        return false;
    }

    constexpr int dimensions=3;
    Real total_weight=Real(0.0);
    Real displacement_mean[3]{};
    Real value_mean=Real(0.0);
    int sample_count=0;
    const Real distance_floor=Real(0.25)*amrex::min(
        cell_size[0]*cell_size[0],cell_size[1]*cell_size[1]);

    const auto accumulate_mean = [&](int sample_i, int sample_j,
                                     int sample_k) noexcept {
        if (!values.contains(sample_i,sample_j,sample_k)
            || !volume_fraction.contains(sample_i,sample_j,sample_k)
            || !cell_centroid.contains(sample_i,sample_j,sample_k)
            || !flags.contains(sample_i,sample_j,sample_k)
            || volume_fraction(sample_i,sample_j,sample_k)<=Real(0.0)
            || flags(sample_i,sample_j,sample_k).isCovered()) {
            return;
        }
        const int offset[3]={sample_i-i,sample_j-j,sample_k-k};
        if ((sample_i!=i || sample_j!=j || sample_k!=k)
            && (!flags(i,j,k).isConnected(offset[0],offset[1],offset[2])
                || !flags(sample_i,sample_j,sample_k).isConnected(
                    -offset[0],-offset[1],-offset[2]))) {
            return;
        }
        Real displacement[3]{};
        Real distance_squared=Real(0.0);
        for (int direction=0; direction<3; ++direction) {
            displacement[direction]=(
                Real(offset[direction])
                +cell_centroid(sample_i,sample_j,sample_k,direction)
                -boundary_centroid(i,j,k,direction))*cell_size[direction];
            distance_squared += displacement[direction]
                *displacement[direction];
        }
        const Real sample=values(sample_i,sample_j,sample_k,component);
        if (!std::isfinite(sample)) {
            amrex::Abort("Non-finite connected wall WLS sample.");
        }
        const Real weight=volume_fraction(sample_i,sample_j,sample_k)
            /amrex::max(distance_squared,distance_floor);
        if (!std::isfinite(distance_squared) || !std::isfinite(weight)) {
            amrex::Abort("Non-finite connected wall WLS geometry.");
        }
        total_weight += weight;
        for (int direction=0; direction<3; ++direction) {
            displacement_mean[direction] += weight*displacement[direction];
        }
        value_mean += weight*sample;
        ++sample_count;
    };

    for (int kk=k-1; kk<=k+1; ++kk) {
        for (int jj=j-1; jj<=j+1; ++jj) {
            for (int ii=i-1; ii<=i+1; ++ii) {
                accumulate_mean(ii,jj,kk);
            }
        }
    }
    if (!std::isfinite(total_weight) || !std::isfinite(value_mean)) {
        amrex::Abort("Non-finite connected wall WLS mean.");
    }
    for (int direction=0; direction<3; ++direction) {
        if (!std::isfinite(displacement_mean[direction])) {
            amrex::Abort("Non-finite connected wall WLS displacement mean.");
        }
    }
    if (!(total_weight>Real(0.0)) || sample_count<dimensions+1) {
        return false;
    }
    for (int direction=0; direction<3; ++direction) {
        displacement_mean[direction] /= total_weight;
    }
    value_mean /= total_weight;

    Real matrix[3][3]{};
    Real right_hand_side[3]{};
    const auto accumulate_fit = [&](int sample_i, int sample_j,
                                    int sample_k) noexcept {
        if (!values.contains(sample_i,sample_j,sample_k)
            || !volume_fraction.contains(sample_i,sample_j,sample_k)
            || !cell_centroid.contains(sample_i,sample_j,sample_k)
            || !flags.contains(sample_i,sample_j,sample_k)
            || volume_fraction(sample_i,sample_j,sample_k)<=Real(0.0)
            || flags(sample_i,sample_j,sample_k).isCovered()) {
            return;
        }
        const int offset[3]={sample_i-i,sample_j-j,sample_k-k};
        if ((sample_i!=i || sample_j!=j || sample_k!=k)
            && (!flags(i,j,k).isConnected(offset[0],offset[1],offset[2])
                || !flags(sample_i,sample_j,sample_k).isConnected(
                    -offset[0],-offset[1],-offset[2]))) {
            return;
        }
        Real centered[3]{};
        Real distance_squared=Real(0.0);
        for (int direction=0; direction<3; ++direction) {
            Real displacement=(Real(offset[direction])
                +cell_centroid(sample_i,sample_j,sample_k,direction)
                -boundary_centroid(i,j,k,direction))*cell_size[direction];
            distance_squared += displacement*displacement;
            centered[direction]=displacement-displacement_mean[direction];
        }
        const Real sample=values(sample_i,sample_j,sample_k,component);
        if (!std::isfinite(sample)) {
            amrex::Abort("Non-finite connected wall WLS sample.");
        }
        const Real weight=volume_fraction(sample_i,sample_j,sample_k)
            /amrex::max(distance_squared,distance_floor);
        if (!std::isfinite(distance_squared) || !std::isfinite(weight)) {
            amrex::Abort("Non-finite connected wall WLS geometry.");
        }
        for (int row=0; row<dimensions; ++row) {
            for (int column=0; column<dimensions; ++column) {
                matrix[row][column] += weight*centered[row]*centered[column];
            }
            right_hand_side[row] += weight*(sample-value_mean)*centered[row];
        }
    };
    for (int kk=k-1; kk<=k+1; ++kk) {
        for (int jj=j-1; jj<=j+1; ++jj) {
            for (int ii=i-1; ii<=i+1; ++ii) {
                accumulate_fit(ii,jj,kk);
            }
        }
    }

    Real gradient[3]{};
    if (!solve_3x3(matrix,right_hand_side,gradient)) {
        return false;
    }

    wall_value=value_mean;
    for (int direction=0; direction<dimensions; ++direction) {
        wall_value -= gradient[direction]*displacement_mean[direction];
    }
    if (!std::isfinite(wall_value)) {
        amrex::Abort("Non-finite connected wall WLS extrapolation.");
    }
    return true;
}

} // namespace hypereb::kernels::eb_gradient

#endif
