// Source attribution and license obligations are recorded in THIRD_PARTY_NOTICES.md.
#include <hyperEB_MovingBoundary.H>
#include <eb_kernels/hyperEB_eb_gradient_K.H>
#include <eb_kernels/hyperEB_index_macros.H>

#include <AMReX.H>

#include <cmath>
#include <limits>
#include <sstream>

namespace hypereb::kernels {

void zero_out_solid_cells(
    const amrex::Box& box,
    const amrex::Array4<amrex::Real>& state,
    int components,
    const amrex::Array4<const amrex::Real>& volume_fraction,
    amrex::Real ambient_density)
{
    const auto low = amrex::lbound(box);
    const auto high = amrex::ubound(box);
    for (int k = low.z; k <= high.z; ++k) {
        for (int j = low.y; j <= high.y; ++j) {
            for (int i = low.x; i <= high.x; ++i) {
                if (volume_fraction(i, j, k) == 0.0) {
                    state(i, j, k, 0) = ambient_density;
                    for (int component = 1; component < components; ++component) {
                        state(i, j, k, component) = 0.0;
                    }
                }
            }
        }
    }
}

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
    int components)
{
    // Only owned targets are applied; their ghost states are filled by the caller.
    const auto low = amrex::lbound(box);
    const auto high = amrex::ubound(box);
    for (int k = low.z; k <= high.z; ++k) {
        for (int j = low.y; j <= high.y; ++j) {
            for (int i = low.x; i <= high.x; ++i) {
                const amrex::Real old_target_fraction =
                    old_volume_fraction(i, j, k);
                const amrex::Real new_target_fraction =
                    new_volume_fraction(i, j, k);
                if (!(new_target_fraction > old_target_fraction)
                    || !(new_target_fraction > 0.0)) {
                    continue;
                }
                // Partially cut cells retain the state advanced by the wall flux.
                if (old_target_fraction > 0.0) {
                    continue;
                }

                // The one-sided affine fit needs a second donor layer.
                constexpr int reconstruction_radius = 2;
                // The previous volume-fraction gradient points from the old
                // solid into the old fluid.  A freshly exposed cell must be
                // reconstructed from that one-sided exterior half-space.
                amrex::Real old_fluid_normal[3]{};
                for (int direction=0; direction<3; ++direction) {
                    const int di=direction==0;
                    const int dj=direction==1;
                    const int dk=direction==2;
                    old_fluid_normal[direction]=0.5*(
                        old_volume_fraction(i+di,j+dj,k+dk)
                        -old_volume_fraction(i-di,j-dj,k-dk));
                }
                const amrex::Real old_normal_squared=
                    old_fluid_normal[0]*old_fluid_normal[0]
                    +old_fluid_normal[1]*old_fluid_normal[1]
                    +old_fluid_normal[2]*old_fluid_normal[2];
                const auto in_old_fluid_half_space = [&] (
                    int donor_i,int donor_j,int donor_k) noexcept {
                    if (!(old_normal_squared>1.e-24)) { return true; }
                    const int offset[3]={donor_i-i,donor_j-j,donor_k-k};
                    amrex::Real projection=0.0;
                    for (int direction=0; direction<3; ++direction) {
                        projection += old_fluid_normal[direction]
                            *static_cast<amrex::Real>(offset[direction]);
                    }
                    return projection>0.0;
                };
                const auto current_fluid = [&](int cell_i, int cell_j,
                                                int cell_k) noexcept {
                    return new_volume_fraction.contains(cell_i,cell_j,cell_k)
                        && new_flags.contains(cell_i,cell_j,cell_k)
                        && new_volume_fraction(cell_i,cell_j,cell_k)>0.0
                        && !new_flags(cell_i,cell_j,cell_k).isCovered();
                };
                const auto mutually_connected = [&](int from_i, int from_j,
                                                      int from_k, int to_i,
                                                      int to_j,
                                                      int to_k) noexcept {
                    const int offset_i=to_i-from_i;
                    const int offset_j=to_j-from_j;
                    const int offset_k=to_k-from_k;
                    if ((offset_i==0 && offset_j==0 && offset_k==0)
                        || std::abs(offset_i)>1 || std::abs(offset_j)>1
                        || std::abs(offset_k)>1
                        || !current_fluid(from_i,from_j,from_k)
                        || !current_fluid(to_i,to_j,to_k)) {
                        return false;
                    }
                    return new_flags(from_i,from_j,from_k).isConnected(
                               offset_i,offset_j,offset_k)
                        && new_flags(to_i,to_j,to_k).isConnected(
                               -offset_i,-offset_j,-offset_k);
                };
                const auto connected_to_target = [&](int donor_i, int donor_j,
                                                       int donor_k) noexcept {
                    const int offset_i=donor_i-i;
                    const int offset_j=donor_j-j;
                    const int offset_k=donor_k-k;
                    if (std::abs(offset_i)<=1 && std::abs(offset_j)<=1
                        && std::abs(offset_k)<=1) {
                        return mutually_connected(
                            i,j,k,donor_i,donor_j,donor_k);
                    }

                    // Radius-two donors are accepted only through an
                    // explicit two-edge path in the current fluid graph.
                    // This provides the second normal layer required by a
                    // one-sided affine fit without crossing a solid or gap.
                    for (int step_k=-1; step_k<=1; ++step_k) {
                        for (int step_j=-1; step_j<=1; ++step_j) {
                            for (int step_i=-1; step_i<=1; ++step_i) {
                                if (step_i==0 && step_j==0 && step_k==0) {
                                    continue;
                                }
                                const int middle_i=i+step_i;
                                const int middle_j=j+step_j;
                                const int middle_k=k+step_k;
                                if (!mutually_connected(
                                        i,j,k,middle_i,middle_j,middle_k)) {
                                    continue;
                                }
                                if (std::abs(donor_i-middle_i)<=1
                                    && std::abs(donor_j-middle_j)<=1
                                    && std::abs(donor_k-middle_k)<=1
                                    && mutually_connected(
                                        middle_i,middle_j,middle_k,
                                        donor_i,donor_j,donor_k)) {
                                    return true;
                                }
                            }
                        }
                    }
                    return false;
                };
                const auto connected_donor = [&](int donor_i, int donor_j,
                                                  int donor_k) noexcept {
                    if (!old_volume_fraction.contains(donor_i,donor_j,donor_k)
                        || !new_volume_fraction.contains(donor_i,donor_j,donor_k)
                        || !new_flags.contains(donor_i,donor_j,donor_k)
                        || !state.contains(donor_i,donor_j,donor_k)
                        || !new_cell_centroid.contains(donor_i,donor_j,donor_k)) {
                        return false;
                    }
                    const amrex::Real old_fraction=
                        old_volume_fraction(donor_i,donor_j,donor_k);
                    if (!(old_fraction>0.0 && std::isfinite(old_fraction))
                        || !(new_volume_fraction(donor_i,donor_j,donor_k)>0.0)
                        || new_flags(donor_i,donor_j,donor_k).isCovered()
                        || !in_old_fluid_half_space(
                            donor_i,donor_j,donor_k)) {
                        return false;
                    }
                    return connected_to_target(donor_i,donor_j,donor_k);
                };

                amrex::Real donor_volume = 0.0;
                for (int kk = -reconstruction_radius;
                     kk <= reconstruction_radius; ++kk) {
                    for (int jj = -reconstruction_radius;
                         jj <= reconstruction_radius; ++jj) {
                        for (int ii = -reconstruction_radius;
                             ii <= reconstruction_radius; ++ii) {
                            if (ii==0 && jj==0 && kk==0) { continue; }
                            const int donor_i=i+ii;
                            const int donor_j=j+jj;
                            const int donor_k=k+kk;
                            if (!connected_donor(donor_i,donor_j,donor_k)) {
                                continue;
                            }
                            const amrex::Real old_fraction=
                                old_volume_fraction(donor_i,donor_j,donor_k);
                            donor_volume += old_fraction;
                        }
                    }
                }

                if (!(donor_volume > 0.0 && std::isfinite(donor_volume))) {
                    int finite_count = 0;
                    int nonzero_count = 0;
                    int positive_count = 0;
                    amrex::Real minimum_fraction =
                        std::numeric_limits<amrex::Real>::max();
                    amrex::Real maximum_fraction =
                        std::numeric_limits<amrex::Real>::lowest();
                    for (int kk = -1; kk <= 1; ++kk) {
                        for (int jj = -1; jj <= 1; ++jj) {
                            for (int ii = -1; ii <= 1; ++ii) {
                                if (!old_volume_fraction.contains(i+ii,j+jj,k+kk)) {
                                    continue;
                                }
                                const amrex::Real fraction = old_volume_fraction(
                                    i + ii, j + jj, k + kk);
                                if (std::isfinite(fraction)) {
                                    ++finite_count;
                                    minimum_fraction =
                                        amrex::min(minimum_fraction, fraction);
                                    maximum_fraction =
                                        amrex::max(maximum_fraction, fraction);
                                }
                                if (fraction != 0.0) {
                                    ++nonzero_count;
                                }
                                if (fraction > 0.0) {
                                    ++positive_count;
                                }
                            }
                        }
                    }

                    std::ostringstream message;
                    message << "Uncovered-cell radius-2 donor invariant failed: level="
                            << level << ", time=" << time
                            << ", moving_step=" << moving_step << ", cell=("
                            << i << ',' << j << ',' << k << "), valid_box="
                            << box << ", new_volfrac="
                            << new_volume_fraction(i, j, k)
                            << ", old_center="
                            << old_volume_fraction(i, j, k)
                            << ", radius1_finite=" << finite_count << "/27"
                            << ", radius1_nonzero=" << nonzero_count
                            << ", radius1_positive=" << positive_count
                            << ", radius1_min=" << minimum_fraction
                            << ", radius1_max=" << maximum_fraction
                            << ", old_volfrac_sums=[";
                    for (int diagnostic_radius = 1; diagnostic_radius <= 4;
                         ++diagnostic_radius) {
                        amrex::Real diagnostic_sum = 0.0;
                        for (int kk = -diagnostic_radius; kk <= diagnostic_radius; ++kk) {
                            for (int jj = -diagnostic_radius; jj <= diagnostic_radius; ++jj) {
                                for (int ii = -diagnostic_radius;
                                     ii <= diagnostic_radius; ++ii) {
                                    if (!old_volume_fraction.contains(i+ii,j+jj,k+kk)) {
                                        continue;
                                    }
                                    diagnostic_sum += old_volume_fraction(
                                        i + ii, j + jj, k + kk);
                                }
                            }
                        }
                        if (diagnostic_radius > 1) {
                            message << ',';
                        }
                        message << diagnostic_sum;
                    }
                    message << "]. The connected uncovered-cell reconstruction "
                               "requires an old-fluid/current-fluid donor "
                               "connected to the fresh cell.";
                    amrex::Abort(message.str());
                }

                // Weighted affine regression about the connected donor cloud.
                amrex::Real regression_weight=0.0;
                amrex::Real displacement_mean[3]{};
                amrex::Real regression_mean[5]{};
                const amrex::Real target_centroid[3]={
                    new_cell_centroid(i,j,k,0),
                    new_cell_centroid(i,j,k,1),
                    new_cell_centroid(i,j,k,2)};
                for (int kk = -reconstruction_radius;
                     kk <= reconstruction_radius; ++kk) {
                    for (int jj=-reconstruction_radius;
                         jj<=reconstruction_radius; ++jj) {
                        for (int ii=-reconstruction_radius;
                             ii<=reconstruction_radius; ++ii) {
                            if (ii==0 && jj==0 && kk==0) { continue; }
                            const int donor_i=i+ii;
                            const int donor_j=j+jj;
                            const int donor_k=k+kk;
                            if (!connected_donor(donor_i,donor_j,donor_k)) {
                                continue;
                            }
                            amrex::Real displacement[3]{};
                            amrex::Real distance_squared=0.0;
                            const int offset[3]={ii,jj,kk};
                            for (int direction=0; direction<3; ++direction) {
                                displacement[direction]=(
                                    static_cast<amrex::Real>(offset[direction])
                                    +new_cell_centroid(
                                        donor_i,donor_j,donor_k,direction)
                                    -target_centroid[direction])
                                    *cell_size[direction];
                                distance_squared += displacement[direction]
                                    *displacement[direction];
                            }
                            if (!(distance_squared>0.0)) { continue; }
                            const amrex::Real weight=
                                old_volume_fraction(donor_i,donor_j,donor_k)
                                /distance_squared;
                            regression_weight += weight;
                            for (int direction=0; direction<3; ++direction) {
                                displacement_mean[direction] +=
                                    weight*displacement[direction];
                            }
                            for (int component=0; component<5; ++component) {
                                regression_mean[component] += weight
                                    *state(donor_i,donor_j,donor_k,component);
                            }
                        }
                    }
                }

                bool fitted=regression_weight>0.0
                    && std::isfinite(regression_weight);
                if (fitted) {
                    for (int direction=0; direction<3; ++direction) {
                        displacement_mean[direction] /= regression_weight;
                    }
                    for (int component=0; component<5; ++component) {
                        regression_mean[component] /= regression_weight;
                    }
                }

                amrex::Real matrix[3][3]{};
                amrex::Real right_hand_side[5][3]{};
                if (fitted) {
                    for (int kk = -reconstruction_radius;
                         kk <= reconstruction_radius; ++kk) {
                        for (int jj = -reconstruction_radius;
                             jj <= reconstruction_radius; ++jj) {
                            for (int ii = -reconstruction_radius;
                                 ii <= reconstruction_radius; ++ii) {
                                if (ii==0 && jj==0 && kk==0) { continue; }
                                const int donor_i = i + ii;
                                const int donor_j = j + jj;
                                const int donor_k = k + kk;
                                if (!connected_donor(
                                        donor_i,donor_j,donor_k)) {
                                    continue;
                                }
                                amrex::Real centered[3]{};
                                amrex::Real distance_squared=0.0;
                                const int offset[3]={ii,jj,kk};
                                for (int direction=0; direction<3; ++direction) {
                                    amrex::Real displacement=(
                                        static_cast<amrex::Real>(offset[direction])
                                        +new_cell_centroid(
                                            donor_i,donor_j,donor_k,direction)
                                        -target_centroid[direction])
                                        *cell_size[direction];
                                    centered[direction]=displacement
                                        -displacement_mean[direction];
                                    distance_squared += displacement*displacement;
                                }
                                if (!(distance_squared>0.0)) { continue; }
                                const amrex::Real weight=
                                    old_volume_fraction(donor_i,donor_j,donor_k)
                                    /distance_squared;
                                for (int row=0; row<3; ++row) {
                                    for (int column=0; column<3;
                                         ++column) {
                                        matrix[row][column] += weight
                                            *centered[row]*centered[column];
                                    }
                                    for (int component=0; component<5;
                                         ++component) {
                                        right_hand_side[component][row] += weight
                                            *(state(donor_i,donor_j,donor_k,component)
                                              -regression_mean[component])
                                            *centered[row];
                                    }
                                }
                            }
                        }
                    }
                }

                amrex::Real reconstructed[5]{};
                if (fitted) {
                    for (int component=0; component<5; ++component) {
                        amrex::Real gradient[3]{};
                        amrex::Real matrix_copy[3][3];
                        amrex::Real rhs_copy[3];
                        for (int row=0; row<3; ++row) {
                            rhs_copy[row]=right_hand_side[component][row];
                            for (int column=0; column<3; ++column) {
                                matrix_copy[row][column]=matrix[row][column];
                            }
                        }
                        fitted=hypereb::kernels::eb_gradient::solve_3x3(
                            matrix_copy,rhs_copy,gradient);
                        if (!fitted) { break; }
                        reconstructed[component]=regression_mean[component];
                        for (int direction=0; direction<3; ++direction) {
                            reconstructed[component] -= gradient[direction]
                                *displacement_mean[direction];
                        }
                    }
                }
                if (!fitted) {
                    amrex::Abort("Uncovered-cell affine reconstruction failed");
                }

                for (int component=0; component<5; ++component) {
                    if (!std::isfinite(reconstructed[component])) {
                        amrex::Abort("Uncovered-cell state is not finite");
                    }
                }
                const amrex::Real rho=reconstructed[URHO];
                if (!(rho>0.0)) {
                    amrex::Abort("Uncovered-cell density is not positive");
                }
                const amrex::Real mx=reconstructed[UMX];
                const amrex::Real my=reconstructed[UMY];
                const amrex::Real mz=reconstructed[UMZ];
                const amrex::Real internal_energy=reconstructed[UEDEN]
                    -0.5*(mx*mx+my*my+mz*mz)/rho;
                const amrex::Real pressure=(gamma-1.0)*internal_energy;
                if (!(internal_energy>0.0 && std::isfinite(internal_energy)
                      && pressure>0.0 && std::isfinite(pressure))) {
                    amrex::Abort("Uncovered-cell internal energy or pressure is invalid");
                }
                for (int component=0; component<5; ++component) {
                    conservative_adjustment(i,j,k,component)=reconstructed[component];
                }
                if (components>UEINT) {
                    conservative_adjustment(i,j,k,UEINT)=internal_energy;
                }
                if (components>UTEMP) {
                    const amrex::Real temperature=internal_energy/(rho*cv);
                    if (!(temperature>0.0 && std::isfinite(temperature))) {
                        amrex::Abort("Uncovered-cell temperature is invalid");
                    }
                    conservative_adjustment(i,j,k,UTEMP)=temperature;
                }
                // A positive tag identifies a validated affine reconstruction.
                donor_tag(i, j, k) = 2.0;
            }
        }
    }
}

void apply_uncovered_cells(
    const amrex::Box& box,
    const amrex::Array4<amrex::Real>& state,
    const amrex::Array4<const amrex::Real>& conservative_adjustment,
    const amrex::Array4<const amrex::Real>& donor_tag,
    int components)
{
    const auto low = amrex::lbound(box);
    const auto high = amrex::ubound(box);
    for (int k = low.z; k <= high.z; ++k) {
        for (int j = low.y; j <= high.y; ++j) {
            for (int i = low.x; i <= high.x; ++i) {
                if (donor_tag(i, j, k) > 0.0) {
                    for (int component = 0; component < components; ++component) {
                        if (component == 7) { continue; }
                        state(i, j, k, component)
                            = conservative_adjustment(i, j, k, component);
                    }
                }
            }
        }
    }
}

} // namespace hypereb::kernels
