#include <hyperEB.H>
#include <hyperEB_EmbeddedBoundary.H>
#include <hyperEB_RegularGrid.H>
#include <hyperEB_SurfaceWriter.H>

#include <eb_kernels/hyperEB_diffusion_K.H>
#include <eb_kernels/hyperEB_hydro_K.H>
#include <eb_kernels/hyperEB_parm.H>

#include <AMReX_EBFArrayBox.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <sstream>

using namespace amrex;

namespace {

constexpr int num_residual_components = 5;

constexpr std::array<const char*, num_residual_components> residual_names{
    "density", "xmom", "ymom", "zmom", "rho_E"};

enum PhysicalWallField {
    WallX = 0,
    WallY,
    WallZ,
    WallNormalX,
    WallNormalY,
    WallNormalZ,
    WallArea,
    WallPressure,
    WallTemperature,
    WallHeatFlux,
    WallPressureTractionX,
    WallPressureTractionY,
    WallPressureTractionZ,
    WallViscousTractionX,
    WallViscousTractionY,
    WallViscousTractionZ,
    WallCp,
    WallCfX,
    WallCfY,
    WallCfZ,
    WallCfMagnitude,
    WallYPlus,
    WallUPlus,
    WallDensity,
    WallMolecularViscosity,
    WallFrictionVelocity,
    WallAmrLevel,
    WallDirection,
    WallSide,
    WallNoSlip,
    NumPhysicalWallFields
};

constexpr std::array<const char*, NumPhysicalWallFields> physical_wall_names{
    "x", "y", "z", "normal_x", "normal_y", "normal_z", "area",
    "pressure", "temperature", "wall_heat_flux",
    "pressure_traction_x", "pressure_traction_y", "pressure_traction_z",
    "viscous_traction_x", "viscous_traction_y", "viscous_traction_z",
    "cp", "cf_x", "cf_y", "cf_z", "cf_magnitude", "y_plus", "u_plus",
    "density", "molecular_viscosity", "friction_velocity", "amr_level",
    "boundary_direction", "boundary_side", "no_slip"};

Parm make_diagnostic_parameters (const hypereb::kernels::RegularPhysics& physics)
{
    Parm parameters;
    parameters.eos_gamma = physics.gamma;
    parameters.cv = physics.cv;
    parameters.cp = physics.cp;
    parameters.Pr = physics.prandtl;
    parameters.C_S = physics.sutherland_coefficient;
    parameters.T_S = physics.sutherland_temperature;
    parameters.viscosity_power_law = physics.viscosity_power_law;
    parameters.viscosity_reference_mu = physics.viscosity_reference_mu;
    parameters.viscosity_reference_temperature = physics.viscosity_reference_temperature;
    parameters.viscosity_power_exponent = physics.viscosity_power_exponent;
    parameters.const_visc_mu = physics.constant_mu;
    parameters.const_visc_ki = physics.constant_xi;
    parameters.const_lambda = physics.constant_lambda;
    return parameters;
}

std::vector<Real> gather_rows (const std::vector<Real>& local)
{
    const int root = ParallelDescriptor::IOProcessorNumber();
    const int local_count = static_cast<int>(local.size());
    const std::vector<int> counts = ParallelDescriptor::Gather(local_count, root);
    std::vector<int> offsets;
    std::vector<Real> global;
    if (ParallelDescriptor::IOProcessor()) {
        offsets.resize(counts.size());
        int total = 0;
        for (std::size_t rank = 0; rank < counts.size(); ++rank) {
            offsets[rank] = total;
            total += counts[rank];
        }
        global.resize(total);
    }
    ParallelDescriptor::Gatherv(
        local.empty() ? nullptr : local.data(), local_count,
        global.empty() ? nullptr : global.data(), counts, offsets, root);
    return global;
}

struct ResidualSums {
    std::array<Real, num_residual_components> rhs_l1{};
    std::array<Real, num_residual_components> rhs_l2{};
    std::array<Real, num_residual_components> rhs_linf{};
    std::array<Real, num_residual_components> state_l1{};
    std::array<Real, num_residual_components> state_l2{};
    std::array<Real, num_residual_components> state_linf{};
    Real fluid_volume = 0.0;
};

void add_sums (ResidualSums& destination, const ResidualSums& source)
{
    destination.fluid_volume += source.fluid_volume;
    for (int n = 0; n < num_residual_components; ++n) {
        destination.rhs_l1[n] += source.rhs_l1[n];
        destination.rhs_l2[n] += source.rhs_l2[n];
        destination.rhs_linf[n] = std::max(destination.rhs_linf[n], source.rhs_linf[n]);
        destination.state_l1[n] += source.state_l1[n];
        destination.state_l2[n] += source.state_l2[n];
        destination.state_linf[n] = std::max(destination.state_linf[n], source.state_linf[n]);
    }
}

void write_header (std::ostream& stream)
{
    stream << "step,time,dt,fluid_volume,residual_max";
    for (const char* name : residual_names) {
        stream << ',' << name << "_rel_l1"
               << ',' << name << "_rel_l2"
               << ',' << name << "_rel_linf"
               << ',' << name << "_rhs_l2"
               << ',' << name << "_rhs_linf";
    }
    stream << '\n';
}

void write_force_header (std::ostream& stream)
{
    stream << "step,time,surface_area,pressure_fx,pressure_fy,pressure_fz,"
              "viscous_fx,viscous_fy,viscous_fz,total_fx,total_fy,total_fz,"
              "pressure_mx,pressure_my,pressure_mz,viscous_mx,viscous_my,"
              "viscous_mz,total_mx,total_my,total_mz,cd,cl,cs,cmx,cmy,cmz,"
              "heat_transfer\n";
}

Real dot_product (const std::array<Real,3>& left,
                  const std::array<Real,3>& right)
{
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2];
}

} // namespace

void
HyperEB::recordResidualRhs (const MultiFab& rhs)
{
    if (residual_int <= 0) {
        return;
    }

    if (!residual_rhs || residual_rhs->boxArray() != rhs.boxArray() ||
        residual_rhs->DistributionMap() != rhs.DistributionMap()) {
        residual_rhs = std::make_unique<MultiFab>(
            rhs.boxArray(), rhs.DistributionMap(), num_residual_components, 0,
            MFInfo(), rhs.Factory());
    }

    MultiFab::Copy(*residual_rhs, rhs, 0, 0, num_residual_components, 0);
    residual_rhs_valid = true;
}

void
HyperEB::initializeResidualHistory (bool restart)
{
    if (residual_int <= 0 || residual_file_initialized) {
        return;
    }

    if (ParallelDescriptor::IOProcessor()) {
        const std::filesystem::path path(residual_file);
        if (path.has_parent_path()) {
            std::filesystem::create_directories(path.parent_path());
        }

        bool append = restart && std::filesystem::exists(path) &&
                      std::filesystem::file_size(path) > 0;
        std::ofstream stream(path, append ? std::ios::app : std::ios::trunc);
        if (!stream) {
            amrex::Abort("Unable to open residual history file: " + residual_file);
        }
        if (!append) {
            write_header(stream);
        }
    }

    residual_file_initialized = true;
}

void
HyperEB::writeResidualHistory (Real time) const
{
    ResidualSums sums;

    for (int lev = 0; lev <= parent->finestLevel(); ++lev) {
        const HyperEB& hypereb_level = dynamic_cast<const HyperEB&>(parent->getLevel(lev));
        if (!hypereb_level.residual_rhs_valid || !hypereb_level.residual_rhs) {
            if (ParallelDescriptor::IOProcessor()) {
                amrex::Print() << "hyperEB residual skipped: level " << lev
                               << " has no completed time-integrator residual state\n";
            }
            return;
        }

        const MultiFab& rhs = *hypereb_level.residual_rhs;
        const MultiFab& state_new = hypereb_level.get_new_data(State_Type);
        const MultiFab& volume_fraction = hypereb_level.volFrac();
        const auto dx = hypereb_level.geom.CellSizeArray();
        const Real cell_volume = dx[0] * dx[1] * dx[2];

        std::unique_ptr<iMultiFab> fine_mask;
        if (lev < parent->finestLevel()) {
            const HyperEB& fine_level = dynamic_cast<const HyperEB&>(parent->getLevel(lev + 1));
            fine_mask = std::make_unique<iMultiFab>(makeFineMask(
                rhs, fine_level.boxArray(), parent->refRatio(lev), 1, 0));
        }

#ifdef _OPENMP
#pragma omp parallel
#endif
        {
            ResidualSums thread_sums;

            for (MFIter mfi(rhs, true); mfi.isValid(); ++mfi) {
                const Box& box = mfi.tilebox();
                const auto rhs_array = rhs.const_array(mfi);
                const auto state_array = state_new.const_array(mfi);
                const auto vfrac_array = volume_fraction.const_array(mfi);
                Array4<const int> mask_array;
                if (fine_mask) {
                    mask_array = fine_mask->const_array(mfi);
                }

                for (BoxIterator iterator(box); iterator.ok(); ++iterator) {
                    const IntVect& cell = iterator();
                    if (fine_mask && mask_array(cell) == 0) {
                        continue;
                    }

                    const Real weight = cell_volume * vfrac_array(cell);
                    if (!(weight > 0.0)) {
                        continue;
                    }

                    thread_sums.fluid_volume += weight;
                    for (int n = 0; n < num_residual_components; ++n) {
                        const Real rhs_value = std::abs(rhs_array(cell, n));
                        const Real state_value = std::abs(state_array(cell, n));
                        thread_sums.rhs_l1[n] += weight * rhs_value;
                        thread_sums.rhs_l2[n] += weight * rhs_value * rhs_value;
                        thread_sums.rhs_linf[n] = std::max(thread_sums.rhs_linf[n], rhs_value);
                        thread_sums.state_l1[n] += weight * state_value;
                        thread_sums.state_l2[n] += weight * state_value * state_value;
                        thread_sums.state_linf[n] = std::max(thread_sums.state_linf[n], state_value);
                    }
                }
            }

#ifdef _OPENMP
#pragma omp critical(zeb_residual_reduction)
#endif
            add_sums(sums, thread_sums);
        }
    }

    ParallelDescriptor::ReduceRealSum(sums.rhs_l1.data(), num_residual_components);
    ParallelDescriptor::ReduceRealSum(sums.rhs_l2.data(), num_residual_components);
    ParallelDescriptor::ReduceRealMax(sums.rhs_linf.data(), num_residual_components);
    ParallelDescriptor::ReduceRealSum(sums.state_l1.data(), num_residual_components);
    ParallelDescriptor::ReduceRealSum(sums.state_l2.data(), num_residual_components);
    ParallelDescriptor::ReduceRealMax(sums.state_linf.data(), num_residual_components);
    ParallelDescriptor::ReduceRealSum(sums.fluid_volume);

    const Real dt = parent->dtLevel(0);
    std::array<Real, num_residual_components> relative_l1{};
    std::array<Real, num_residual_components> relative_l2{};
    std::array<Real, num_residual_components> relative_linf{};
    std::array<Real, num_residual_components> rhs_l2{};
    Real residual_max = 0.0;

    for (int n = 0; n < num_residual_components; ++n) {
        const Real l1_floor = residual_reference_floor * sums.fluid_volume;
        const Real l2_floor = residual_reference_floor * residual_reference_floor *
                              sums.fluid_volume;
        relative_l1[n] = dt * sums.rhs_l1[n] /
                         std::max(sums.state_l1[n], l1_floor);
        relative_l2[n] = dt * std::sqrt(sums.rhs_l2[n] /
                         std::max(sums.state_l2[n], l2_floor));
        relative_linf[n] = dt * sums.rhs_linf[n] /
                           std::max(sums.state_linf[n], residual_reference_floor);
        rhs_l2[n] = std::sqrt(sums.rhs_l2[n] / sums.fluid_volume);
        residual_max = std::max(residual_max, relative_l2[n]);
    }

    if (ParallelDescriptor::IOProcessor()) {
        if (!residual_file_initialized) {
            const_cast<HyperEB*>(this)->initializeResidualHistory(false);
        }
        std::ofstream stream(residual_file, std::ios::app);
        if (!stream) {
            amrex::Abort("Unable to append residual history file: " + residual_file);
        }
        stream << std::setprecision(17)
               << parent->levelSteps(0) << ',' << time << ',' << dt << ','
               << sums.fluid_volume << ',' << residual_max;
        for (int n = 0; n < num_residual_components; ++n) {
            stream << ',' << relative_l1[n]
                   << ',' << relative_l2[n]
                   << ',' << relative_linf[n]
                   << ',' << rhs_l2[n]
                   << ',' << sums.rhs_linf[n];
        }
        stream << '\n';

        amrex::Print().SetPrecision(10)
            << "[hyperEB residual] step=" << parent->levelSteps(0)
            << " time=" << time << " max_rel_l2=" << residual_max
            << " rho_abs_max=" << sums.state_linf[0]
            << " rhoE_abs_max=" << sums.state_linf[4] << '\n';
    }
}

void
HyperEB::initializePressureForceHistory (bool restart)
{
    if (force_int <= 0 || force_file_initialized) {
        return;
    }

    if (ParallelDescriptor::IOProcessor()) {
        const std::filesystem::path path(force_file);
        if (path.has_parent_path()) {
            std::filesystem::create_directories(path.parent_path());
        }

        const bool append = restart && std::filesystem::exists(path) &&
                            std::filesystem::file_size(path) > 0;
        if (append) {
            std::ifstream previous(path);
            std::string header;
            std::getline(previous, header);
            if (!header.empty() && header.back() == '\r') header.pop_back();
            std::ostringstream expected;
            write_force_header(expected);
            std::string expected_header = expected.str();
            expected_header.pop_back();
            if (!previous || header != expected_header) {
                amrex::Abort("Incompatible aerodynamic history header: " +
                             force_file + ". Select a new diagnostics.force_file.");
            }
        }
        std::ofstream stream(path, append ? std::ios::app : std::ios::trunc);
        if (!stream) {
            amrex::Abort("Unable to open pressure-force history file: " + force_file);
        }
        if (!append) {
            write_force_header(stream);
        }
    }

    force_file_initialized = true;
}

void
HyperEB::collectPhysicalWallData (
    Real time,
    std::array<Real,3>& pressure_force,
    std::array<Real,3>& viscous_force,
    std::array<Real,3>& pressure_moment,
    std::array<Real,3>& viscous_moment,
    Real& surface_area,
    Real& heat_transfer,
    std::vector<Real>* samples) const
{
    const auto physics = hypereb::kernels::regular_physics();
    const Parm parameters = make_diagnostic_parameters(physics);
    const Real dynamic_pressure = Real(0.5) * force_reference_density
        * force_reference_velocity * force_reference_velocity;
    const Real nan = std::numeric_limits<Real>::quiet_NaN();

    for (int lev = 0; lev <= parent->finestLevel(); ++lev) {
        const HyperEB& hypereb_level = dynamic_cast<const HyperEB&>(parent->getLevel(lev));
        const MultiFab& state_new = hypereb_level.get_new_data(State_Type);
        MultiFab state_border(hypereb_level.grids, hypereb_level.dmap, NUM_STATE,
                              NUM_GROW, MFInfo(), hypereb_level.Factory());
        FillPatch(const_cast<HyperEB&>(hypereb_level), state_border, NUM_GROW, time,
                  State_Type, 0, NUM_STATE);

        const Box domain = hypereb_level.geom.Domain();
        const auto dx = hypereb_level.geom.CellSizeArray();
        const auto inverse_spacing = hypereb_level.geom.InvCellSizeArray();
        const auto prob_lo = hypereb_level.geom.ProbLoArray();
        const auto prob_hi = hypereb_level.geom.ProbHiArray();

        std::unique_ptr<iMultiFab> fine_mask;
        if (lev < parent->finestLevel()) {
            const HyperEB& fine_level = dynamic_cast<const HyperEB&>(parent->getLevel(lev + 1));
            fine_mask = std::make_unique<iMultiFab>(makeFineMask(
                state_new, fine_level.boxArray(), parent->refRatio(lev), 1, 0));
        }

        for (MFIter mfi(state_new); mfi.isValid(); ++mfi) {
            const Box& valid_box = mfi.validbox();
            const Box primitive_box = amrex::grow(valid_box, 2);
            FArrayBox primitive(primitive_box, NPRIM);
            const auto q = primitive.array();
            const auto state = state_border[mfi].const_array();
            const auto& eb_factory = dynamic_cast<const EBFArrayBoxFactory&>(state_new.Factory());
            const auto cell_flags = eb_factory.getMultiEBCellFlagFab().const_array(mfi);
            amrex::ParallelFor(primitive_box,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                if (cell_flags(i,j,k).isCovered()) {
                    for (int n = 0; n < NPRIM; ++n) { q(i,j,k,n) = 0.0; }
                    q(i,j,k,QRHO) = 1.0;
                    q(i,j,k,QPRES) = 1.0;
                    q(i,j,k,QEINT) = 1.0 / (parameters.eos_gamma - 1.0);
                    q(i,j,k,QCS) = std::sqrt(parameters.eos_gamma);
                    q(i,j,k,QTEMP) = q(i,j,k,QEINT) / parameters.cv;
                    return;
                }
                hypereb_ctoprim(i,j,k,state,q,parameters);
            });

            const Box coefficient_box = amrex::grow(valid_box, 1);
            FArrayBox coefficient_storage(coefficient_box, NCOEF);
            const auto coefficient = coefficient_storage.array();
            if (physics.use_constant_viscosity) {
                amrex::ParallelFor(coefficient_box,
                [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    hypereb_constcoef(i,j,k,coefficient,parameters);
                });
            } else {
                amrex::ParallelFor(coefficient_box,
                [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    hypereb_diffcoef(i,j,k,q,coefficient,parameters);
                });
            }

            const auto constant_q = primitive.const_array();
            const auto constant_coefficient = coefficient_storage.const_array();
            Array4<const int> mask;
            if (fine_mask) {
                mask = fine_mask->const_array(mfi);
            }

            for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
                for (int side_index = 0; side_index < 2; ++side_index) {
                    const bool low_side = side_index == 0;
                    const int boundary_type = low_side
                        ? phys_bc.lo(direction) : phys_bc.hi(direction);
                    const bool wall_boundary =
                        boundary_type == PhysBCType::slipwall
                        || boundary_type == PhysBCType::noslipwall;
                    if (!wall_boundary) {
                        continue;
                    }
                    if ((low_side && valid_box.smallEnd(direction) != domain.smallEnd(direction))
                        || (!low_side && valid_box.bigEnd(direction) != domain.bigEnd(direction))) {
                        continue;
                    }

                    Box face_box = amrex::surroundingNodes(valid_box, direction);
                    const int face_index = low_side ? domain.smallEnd(direction)
                                                    : domain.bigEnd(direction) + 1;
                    face_box.setSmall(direction, face_index);
                    face_box.setBig(direction, face_index);
                    FArrayBox diffusive_storage(face_box, NEQNS);
                    diffusive_storage.setVal(0.0);
                    const auto diffusive = diffusive_storage.array();
                    if (physics.do_diffusion) {
                        if (direction == 0) {
                            amrex::ParallelFor(face_box,
                            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                            {
                                hypereb_diff_x(i,j,k,constant_q,constant_coefficient,
                                           inverse_spacing,diffusive);
                            });
                        } else if (direction == 1) {
                            amrex::ParallelFor(face_box,
                            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                            {
                                hypereb_diff_y(i,j,k,constant_q,constant_coefficient,
                                           inverse_spacing,diffusive);
                            });
                        } else {
                            amrex::ParallelFor(face_box,
                            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                            {
                                hypereb_diff_z(i,j,k,constant_q,constant_coefficient,
                                           inverse_spacing,diffusive);
                            });
                        }
                    }

#ifdef AMREX_USE_GPU
                    Gpu::streamSynchronize();
#endif
                    const auto diffusive_values = diffusive_storage.const_array();
                    const Real side_sign = low_side ? Real(-1.0) : Real(1.0);
                    const Real face_area = dx[(direction + 1) % 3]
                        * dx[(direction + 2) % 3];

                    for (BoxIterator iterator(face_box); iterator.ok(); ++iterator) {
                        const IntVect& face = iterator();
                        IntVect cell = face;
                        if (!low_side) {
                            cell[direction] -= 1;
                        }
                        if (fine_mask && mask(cell) == 0) {
                            continue;
                        }

                        std::array<Real,3> position{};
                        for (int coordinate = 0; coordinate < 3; ++coordinate) {
                            position[coordinate] = coordinate == direction
                                ? (low_side ? prob_lo[coordinate] : prob_hi[coordinate])
                                : prob_lo[coordinate]
                                  + (static_cast<Real>(face[coordinate]) + Real(0.5))
                                    * dx[coordinate];
                        }
                        const bool no_slip =
                            boundary_type == PhysBCType::noslipwall;

                        const Real pressure = constant_q(cell,QPRES);
                        const Real temperature = physics.wall_thermal_boundary
                            == hypereb::kernels::EBThermalBoundary::isothermal && no_slip
                            ? physics.wall_temperature : constant_q(cell,QTEMP);
                        const Real density = constant_q(cell,QRHO);
                        const Real local_temperature = amrex::max(
                            constant_q(cell,QTEMP), Real(1.0e-30));
                        const Real molecular_mu = physics.use_constant_viscosity
                            ? amrex::max(physics.constant_mu, Real(0.0))
                            : hypereb_variable_viscosity(local_temperature, parameters);
                        std::array<Real,3> normal{};
                        normal[direction] = side_sign;
                        std::array<Real,3> cell_pressure_force{};
                        std::array<Real,3> cell_viscous_force{};
                        cell_pressure_force[direction] = side_sign
                            * (pressure - force_pressure_offset) * face_area;
                        if (physics.do_diffusion) {
                            cell_viscous_force[0] = side_sign
                                * diffusive_values(face,UMX) * face_area;
                            cell_viscous_force[1] = side_sign
                                * diffusive_values(face,UMY) * face_area;
                            cell_viscous_force[2] = side_sign
                                * diffusive_values(face,UMZ) * face_area;
                        }

                        const std::array<Real,3> arm{
                            position[0] - force_reference_point[0],
                            position[1] - force_reference_point[1],
                            position[2] - force_reference_point[2]};
                        for (int component = 0; component < 3; ++component) {
                            pressure_force[component] += cell_pressure_force[component];
                            viscous_force[component] += cell_viscous_force[component];
                        }
                        pressure_moment[0] += arm[1]*cell_pressure_force[2]
                            - arm[2]*cell_pressure_force[1];
                        pressure_moment[1] += arm[2]*cell_pressure_force[0]
                            - arm[0]*cell_pressure_force[2];
                        pressure_moment[2] += arm[0]*cell_pressure_force[1]
                            - arm[1]*cell_pressure_force[0];
                        viscous_moment[0] += arm[1]*cell_viscous_force[2]
                            - arm[2]*cell_viscous_force[1];
                        viscous_moment[1] += arm[2]*cell_viscous_force[0]
                            - arm[0]*cell_viscous_force[2];
                        viscous_moment[2] += arm[0]*cell_viscous_force[1]
                            - arm[1]*cell_viscous_force[0];
                        surface_area += face_area;

                        int left_i=face[0], left_j=face[1], left_k=face[2];
                        int right_i=face[0], right_j=face[1], right_k=face[2];
                        if (direction == 0) { --left_i; }
                        if (direction == 1) { --left_j; }
                        if (direction == 2) { --left_k; }
                        const Real temperature_gradient =
                            (constant_q(right_i,right_j,right_k,QTEMP)
                             - constant_q(left_i,left_j,left_k,QTEMP))
                            * inverse_spacing[direction];
                        const Real conductivity = Real(0.5)
                            * (constant_coefficient(right_i,right_j,right_k,CLAM)
                               + constant_coefficient(left_i,left_j,left_k,CLAM));
                        const Real heat_flux_to_wall = side_sign
                            * conductivity * temperature_gradient;
                        heat_transfer += heat_flux_to_wall * face_area;

                        if (!samples) {
                            continue;
                        }
                        std::array<Real,3> pressure_traction{};
                        std::array<Real,3> viscous_traction{};
                        for (int component = 0; component < 3; ++component) {
                            pressure_traction[component] =
                                cell_pressure_force[component] / face_area;
                            viscous_traction[component] =
                                cell_viscous_force[component] / face_area;
                        }
                        const Real normal_viscous = dot_product(viscous_traction,normal);
                        std::array<Real,3> shear{};
                        Real shear_squared = 0.0;
                        for (int component = 0; component < 3; ++component) {
                            shear[component] = viscous_traction[component]
                                - normal_viscous * normal[component];
                            shear_squared += shear[component] * shear[component];
                        }
                        const Real shear_magnitude = std::sqrt(shear_squared);
                        const Real friction_velocity = no_slip && density > 0.0
                            ? std::sqrt(shear_magnitude / density) : Real(0.0);
                        const Real first_cell_distance = Real(0.5)*dx[direction];
                        Real tangential_speed_squared = 0.0;
                        const std::array<Real,3> velocity{
                            constant_q(cell,QU),constant_q(cell,QV),constant_q(cell,QW)};
                        const Real normal_velocity = dot_product(velocity,normal);
                        for (int component = 0; component < 3; ++component) {
                            const Real tangential_velocity = velocity[component]
                                - normal_velocity*normal[component];
                            tangential_speed_squared += tangential_velocity*tangential_velocity;
                        }
                        const Real y_plus = no_slip && molecular_mu > 0.0
                            ? density*friction_velocity*first_cell_distance/molecular_mu
                            : nan;
                        const Real u_plus = no_slip && friction_velocity > 0.0
                            ? std::sqrt(tangential_speed_squared)/friction_velocity : nan;

                        const std::size_t offset = samples->size();
                        samples->resize(offset + NumPhysicalWallFields, Real(0.0));
                        auto set = [&] (int field, Real value) {
                            (*samples)[offset + static_cast<std::size_t>(field)] = value;
                        };
                        set(WallX,position[0]); set(WallY,position[1]); set(WallZ,position[2]);
                        set(WallNormalX,normal[0]); set(WallNormalY,normal[1]); set(WallNormalZ,normal[2]);
                        set(WallArea,face_area); set(WallPressure,pressure);
                        set(WallTemperature,temperature); set(WallHeatFlux,heat_flux_to_wall);
                        for (int component = 0; component < 3; ++component) {
                            set(WallPressureTractionX+component,pressure_traction[component]);
                            set(WallViscousTractionX+component,viscous_traction[component]);
                            set(WallCfX+component,dynamic_pressure > 0.0
                                ? shear[component]/dynamic_pressure : nan);
                        }
                        set(WallCp,dynamic_pressure > 0.0
                            ? (pressure-force_reference_pressure)/dynamic_pressure : nan);
                        set(WallCfMagnitude,dynamic_pressure > 0.0
                            ? shear_magnitude/dynamic_pressure : nan);
                        set(WallYPlus,y_plus); set(WallUPlus,u_plus);
                        set(WallDensity,density); set(WallMolecularViscosity,molecular_mu);
                        set(WallFrictionVelocity,friction_velocity);
                        set(WallAmrLevel,static_cast<Real>(lev));
                        set(WallDirection,static_cast<Real>(direction));
                        set(WallSide,side_sign); set(WallNoSlip,no_slip ? Real(1.0) : Real(0.0));
                    }
                }
            }
        }
    }
}

void
HyperEB::writePressureForceHistory (Real time)
{
    std::array<Real,3> pressure_force{};
    std::array<Real,3> viscous_force{};
    std::array<Real,3> pressure_moment{};
    std::array<Real,3> viscous_moment{};
    Real surface_area = 0.0;
    Real heat_transfer = 0.0;
    const auto physics = hypereb::kernels::regular_physics();

    collectPhysicalWallData(
        time, pressure_force, viscous_force, pressure_moment, viscous_moment,
        surface_area, heat_transfer, nullptr);

    for (int lev = 0; lev <= parent->finestLevel(); ++lev) {
        HyperEB& hypereb_level = dynamic_cast<HyperEB&>(parent->getLevel(lev));
        const MultiFab& state_new = hypereb_level.get_new_data(State_Type);
        MultiFab state_border(hypereb_level.grids, hypereb_level.dmap, NUM_STATE,
                              NUM_GROW, MFInfo(), hypereb_level.Factory());
        FillPatch(hypereb_level, state_border, NUM_GROW, time,
                  State_Type, 0, NUM_STATE);
        MultiFab surface_loads(hypereb_level.grids, hypereb_level.dmap,
                               hypereb::kernels::NumSurfaceLoadComponents, 0);

        const auto& factory = dynamic_cast<const EBFArrayBoxFactory&>(state_new.Factory());
        const auto& flags = factory.getMultiEBCellFlagFab();
        const auto& boundary_centroid = factory.getBndryCent();
        const auto area_fraction = factory.getAreaFrac();
        const auto face_centroid = factory.getFaceCent();
        const auto dx = hypereb_level.geom.CellSizeArray();
        const auto prob_lo = hypereb_level.geom.ProbLoArray();

        std::unique_ptr<iMultiFab> fine_mask;
        if (lev < parent->finestLevel()) {
            const HyperEB& fine_level = dynamic_cast<const HyperEB&>(parent->getLevel(lev + 1));
            fine_mask = std::make_unique<iMultiFab>(makeFineMask(
                state_new, fine_level.boxArray(), parent->refRatio(lev), 1, 0));
        }

#ifdef _OPENMP
#pragma omp parallel
#endif
        {
            std::array<Real,3> thread_pressure_force{};
            std::array<Real,3> thread_viscous_force{};
            std::array<Real,3> thread_pressure_moment{};
            std::array<Real,3> thread_viscous_moment{};
            Real thread_area = 0.0;
            Real thread_heat_transfer = 0.0;

            for (MFIter mfi(state_new, true); mfi.isValid(); ++mfi) {
                const Box& box = mfi.tilebox();
                const hypereb::kernels::EBGeometryViews eb_views{
                    flags[mfi].const_array(),
                    hypereb_level.volFrac()[mfi].const_array(),
                    factory.getCentroid()[mfi].const_array(),
                    boundary_centroid[mfi].const_array(),
                    {area_fraction[0]->operator[](mfi).const_array(),
                     area_fraction[1]->operator[](mfi).const_array(),
                     area_fraction[2]->operator[](mfi).const_array()},
                    {face_centroid[0]->operator[](mfi).const_array(),
                     face_centroid[1]->operator[](mfi).const_array(),
                     face_centroid[2]->operator[](mfi).const_array()}};
                hypereb::kernels::compute_eb_surface_loads(
                    box, surface_loads[mfi].array(), state_border[mfi].const_array(),
                    hypereb_level.geom, physics, eb_views, time);

                const auto load_array = surface_loads.const_array(mfi);
                const auto flag_array = flags.const_array(mfi);
                const auto bcent_array = boundary_centroid.const_array(mfi);
                const auto area_x_array = area_fraction[0]->const_array(mfi);
                const auto area_y_array = area_fraction[1]->const_array(mfi);
                const auto area_z_array = area_fraction[2]->const_array(mfi);
                Array4<const int> mask_array;
                if (fine_mask) {
                    mask_array = fine_mask->const_array(mfi);
                }

                for (BoxIterator iterator(box); iterator.ok(); ++iterator) {
                    const IntVect& cell = iterator();
                    if (!flag_array(cell).isSingleValued() ||
                        (fine_mask && mask_array(cell) == 0)) {
                        continue;
                    }

                    const int i = cell[0];
                    const int j = cell[1];
                    const int k = cell[2];
                    std::array<Real,3> cell_pressure_force{
                        load_array(cell,hypereb::kernels::PressureForceX),
                        load_array(cell,hypereb::kernels::PressureForceY),
                        load_array(cell,hypereb::kernels::PressureForceZ)};
                    std::array<Real,3> cell_viscous_force{
                        load_array(cell,hypereb::kernels::ViscousForceX),
                        load_array(cell,hypereb::kernels::ViscousForceY),
                        load_array(cell,hypereb::kernels::ViscousForceZ)};

                    if (force_pressure_offset != 0.0) {
                        cell_pressure_force[0] -= force_pressure_offset
                            * (area_x_array(i,j,k) - area_x_array(i+1,j,k))
                            * dx[1] * dx[2];
                        cell_pressure_force[1] -= force_pressure_offset
                            * (area_y_array(i,j,k) - area_y_array(i,j+1,k))
                            * dx[0] * dx[2];
                        cell_pressure_force[2] -= force_pressure_offset
                            * (area_z_array(i,j,k) - area_z_array(i,j,k+1))
                            * dx[0] * dx[1];
                    }

                    const std::array<Real,3> position{
                        prob_lo[0] + (static_cast<Real>(i) + 0.5 + bcent_array(cell,0)) * dx[0],
                        prob_lo[1] + (static_cast<Real>(j) + 0.5 + bcent_array(cell,1)) * dx[1],
                        prob_lo[2] + (static_cast<Real>(k) + 0.5 + bcent_array(cell,2)) * dx[2]};
                    const std::array<Real,3> arm{
                        position[0] - force_reference_point[0],
                        position[1] - force_reference_point[1],
                        position[2] - force_reference_point[2]};

                    for (int direction = 0; direction < 3; ++direction) {
                        thread_pressure_force[direction] += cell_pressure_force[direction];
                        thread_viscous_force[direction] += cell_viscous_force[direction];
                    }
                    thread_pressure_moment[0] += arm[1] * cell_pressure_force[2] - arm[2] * cell_pressure_force[1];
                    thread_pressure_moment[1] += arm[2] * cell_pressure_force[0] - arm[0] * cell_pressure_force[2];
                    thread_pressure_moment[2] += arm[0] * cell_pressure_force[1] - arm[1] * cell_pressure_force[0];
                    thread_viscous_moment[0] += arm[1] * cell_viscous_force[2] - arm[2] * cell_viscous_force[1];
                    thread_viscous_moment[1] += arm[2] * cell_viscous_force[0] - arm[0] * cell_viscous_force[2];
                    thread_viscous_moment[2] += arm[0] * cell_viscous_force[1] - arm[1] * cell_viscous_force[0];
                    thread_area += load_array(cell,hypereb::kernels::BoundaryArea);
                    thread_heat_transfer +=
                        load_array(cell,hypereb::kernels::HeatTransfer);
                }
            }

#ifdef _OPENMP
#pragma omp critical(zeb_surface_force_reduction)
#endif
            {
                surface_area += thread_area;
                heat_transfer += thread_heat_transfer;
                for (int direction = 0; direction < 3; ++direction) {
                    pressure_force[direction] += thread_pressure_force[direction];
                    viscous_force[direction] += thread_viscous_force[direction];
                    pressure_moment[direction] += thread_pressure_moment[direction];
                    viscous_moment[direction] += thread_viscous_moment[direction];
                }
            }
        }
    }

    ParallelDescriptor::ReduceRealSum(pressure_force.data(), 3);
    ParallelDescriptor::ReduceRealSum(viscous_force.data(), 3);
    ParallelDescriptor::ReduceRealSum(pressure_moment.data(), 3);
    ParallelDescriptor::ReduceRealSum(viscous_moment.data(), 3);
    ParallelDescriptor::ReduceRealSum(surface_area);
    ParallelDescriptor::ReduceRealSum(heat_transfer);

    std::array<Real,3> total_force{};
    std::array<Real,3> total_moment{};
    for (int direction = 0; direction < 3; ++direction) {
        total_force[direction] = pressure_force[direction] + viscous_force[direction];
        total_moment[direction] = pressure_moment[direction] + viscous_moment[direction];
    }

    const Real dynamic_pressure = 0.5 * force_reference_density *
                                  force_reference_velocity * force_reference_velocity;
    const Real force_scale = dynamic_pressure * force_reference_area;
    const Real moment_scale = force_scale * force_reference_length;
    const Real nan = std::numeric_limits<Real>::quiet_NaN();
    Real cd = nan;
    Real cl = nan;
    Real cs = nan;
    std::array<Real,3> cm{nan, nan, nan};
    if (force_scale > 0.0) {
        cd = dot_product(total_force, force_drag_direction) / force_scale;
        cl = dot_product(total_force, force_lift_direction) / force_scale;
        cs = dot_product(total_force, force_side_direction) / force_scale;
        for (int direction = 0; direction < 3; ++direction) {
            cm[direction] = total_moment[direction] / moment_scale;
        }
    }

    if (ParallelDescriptor::IOProcessor()) {
        if (!force_file_initialized) {
            initializePressureForceHistory(false);
        }
        std::ofstream stream(force_file, std::ios::app);
        if (!stream) {
            amrex::Abort("Unable to append aerodynamic history file: " + force_file);
        }
        stream << std::setprecision(17)
               << parent->levelSteps(0) << ',' << time << ',' << surface_area;
        for (Real value : pressure_force) stream << ',' << value;
        for (Real value : viscous_force) stream << ',' << value;
        for (Real value : total_force) stream << ',' << value;
        for (Real value : pressure_moment) stream << ',' << value;
        for (Real value : viscous_moment) stream << ',' << value;
        for (Real value : total_moment) stream << ',' << value;
        stream << ',' << cd << ',' << cl << ',' << cs;
        for (Real value : cm) stream << ',' << value;
        stream << ',' << heat_transfer << '\n';

        amrex::Print().SetPrecision(10)
            << "[hyperEB aerodynamic load] step=" << parent->levelSteps(0)
            << " Fp=(" << pressure_force[0] << ',' << pressure_force[1] << ','
            << pressure_force[2] << ") Fv=(" << viscous_force[0] << ','
            << viscous_force[1] << ',' << viscous_force[2] << ") Qwall="
            << heat_transfer << '\n';
    }
}

void
HyperEB::writeEBSurfaceData (Real time)
{
    enum SurfaceField {
        FieldPressure = 0,
        FieldTemperature,
        FieldWallTemperature,
        FieldWallHeatFlux,
        FieldHeatFluxX,
        FieldHeatFluxY,
        FieldHeatFluxZ,
        FieldHeatTransfer,
        FieldNormalX,
        FieldNormalY,
        FieldNormalZ,
        FieldPressureTractionX,
        FieldPressureTractionY,
        FieldPressureTractionZ,
        FieldViscousTractionX,
        FieldViscousTractionY,
        FieldViscousTractionZ,
        FieldCp,
        FieldCfX,
        FieldCfY,
        FieldCfZ,
        FieldCfMagnitude,
        FieldBoundaryArea,
        FieldAmrLevel,
        FieldVolumeFraction,
        NumSurfaceFields
    };

    const std::vector<std::string> field_names{
        "pressure", "temperature", "wall_temperature", "wall_heat_flux",
        "heat_flux_x", "heat_flux_y", "heat_flux_z", "heat_transfer",
        "normal_x", "normal_y", "normal_z",
        "pressure_traction_x", "pressure_traction_y", "pressure_traction_z",
        "viscous_traction_x", "viscous_traction_y", "viscous_traction_z",
        "cp", "cf_x", "cf_y", "cf_z", "cf_magnitude", "boundary_area",
        "amr_level", "volume_fraction"};
    const auto physics = hypereb::kernels::regular_physics();
    const Real dynamic_pressure = 0.5 * force_reference_density *
                                  force_reference_velocity * force_reference_velocity;
    const Real nan = std::numeric_limits<Real>::quiet_NaN();
    hypereb::SurfaceWriter writer;
    writer.SetCellDataNames(field_names);

    for (int lev = 0; lev <= parent->finestLevel(); ++lev) {
        HyperEB& hypereb_level = dynamic_cast<HyperEB&>(parent->getLevel(lev));
        const MultiFab& state_new = hypereb_level.get_new_data(State_Type);
        MultiFab state_border(hypereb_level.grids, hypereb_level.dmap, NUM_STATE,
                              NUM_GROW, MFInfo(), hypereb_level.Factory());
        FillPatch(hypereb_level, state_border, NUM_GROW, time,
                  State_Type, 0, NUM_STATE);
        MultiFab surface_loads(hypereb_level.grids, hypereb_level.dmap,
                               hypereb::kernels::NumSurfaceLoadComponents, 0);
        MultiFab surface_fields(hypereb_level.grids, hypereb_level.dmap,
                                NumSurfaceFields, 0);

        const auto& factory = dynamic_cast<const EBFArrayBoxFactory&>(state_new.Factory());
        const auto& flags = factory.getMultiEBCellFlagFab();
        const auto& boundary_centroid = factory.getBndryCent();
        const auto area_fraction = factory.getAreaFrac();
        const auto face_centroid = factory.getFaceCent();
        std::unique_ptr<iMultiFab> fine_mask;
        if (lev < parent->finestLevel()) {
            const HyperEB& fine_level = dynamic_cast<const HyperEB&>(parent->getLevel(lev + 1));
            fine_mask = std::make_unique<iMultiFab>(makeFineMask(
                state_new, fine_level.boxArray(), parent->refRatio(lev), 1, 0));
        }

#ifdef _OPENMP
#pragma omp parallel
#endif
        for (MFIter mfi(state_new, true); mfi.isValid(); ++mfi) {
            const Box& box = mfi.tilebox();
            const hypereb::kernels::EBGeometryViews eb_views{
                flags[mfi].const_array(),
                hypereb_level.volFrac()[mfi].const_array(),
                factory.getCentroid()[mfi].const_array(),
                boundary_centroid[mfi].const_array(),
                {area_fraction[0]->operator[](mfi).const_array(),
                 area_fraction[1]->operator[](mfi).const_array(),
                 area_fraction[2]->operator[](mfi).const_array()},
                {face_centroid[0]->operator[](mfi).const_array(),
                 face_centroid[1]->operator[](mfi).const_array(),
                 face_centroid[2]->operator[](mfi).const_array()}};
            hypereb::kernels::compute_eb_surface_loads(
                box, surface_loads[mfi].array(), state_border[mfi].const_array(),
                hypereb_level.geom, physics, eb_views, time);

            const auto loads = surface_loads.const_array(mfi);
            const auto fields = surface_fields.array(mfi);
            const auto volume_fraction = hypereb_level.volFrac()[mfi].const_array();
            amrex::ParallelFor(box,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                const Real area = loads(i,j,k,hypereb::kernels::BoundaryArea);
                fields(i,j,k,FieldPressure) = loads(i,j,k,hypereb::kernels::SurfacePressure);
                fields(i,j,k,FieldTemperature) = loads(i,j,k,hypereb::kernels::SurfaceTemperature);
                fields(i,j,k,FieldWallTemperature) =
                    loads(i,j,k,hypereb::kernels::WallTemperature);
                fields(i,j,k,FieldWallHeatFlux) =
                    loads(i,j,k,hypereb::kernels::WallHeatFlux);
                fields(i,j,k,FieldHeatTransfer) =
                    loads(i,j,k,hypereb::kernels::HeatTransfer);
                fields(i,j,k,FieldNormalX) = loads(i,j,k,hypereb::kernels::SurfaceNormalX);
                fields(i,j,k,FieldNormalY) = loads(i,j,k,hypereb::kernels::SurfaceNormalY);
                fields(i,j,k,FieldNormalZ) = loads(i,j,k,hypereb::kernels::SurfaceNormalZ);
                fields(i,j,k,FieldBoundaryArea) = area;
                fields(i,j,k,FieldAmrLevel) = static_cast<Real>(lev);
                fields(i,j,k,FieldVolumeFraction) = volume_fraction(i,j,k);

                const Real inverse_area = area > 0.0 ? 1.0 / area : 0.0;
                const std::array<Real,3> normal{
                    loads(i,j,k,hypereb::kernels::SurfaceNormalX),
                    loads(i,j,k,hypereb::kernels::SurfaceNormalY),
                    loads(i,j,k,hypereb::kernels::SurfaceNormalZ)};
                for (int direction = 0; direction < 3; ++direction) {
                    fields(i,j,k,FieldHeatFluxX + direction) =
                        fields(i,j,k,FieldWallHeatFlux) * normal[direction];
                }
                std::array<Real,3> viscous_traction{};
                for (int direction = 0; direction < 3; ++direction) {
                    fields(i,j,k,FieldPressureTractionX + direction) =
                        loads(i,j,k,hypereb::kernels::PressureForceX + direction) * inverse_area;
                    viscous_traction[direction] =
                        loads(i,j,k,hypereb::kernels::ViscousForceX + direction) * inverse_area;
                    fields(i,j,k,FieldViscousTractionX + direction) = viscous_traction[direction];
                }

                if (dynamic_pressure > 0.0) {
                    fields(i,j,k,FieldCp) =
                        (fields(i,j,k,FieldPressure) - force_reference_pressure)
                        / dynamic_pressure;
                    const Real normal_traction = viscous_traction[0] * normal[0]
                        + viscous_traction[1] * normal[1]
                        + viscous_traction[2] * normal[2];
                    Real cf_squared = 0.0;
                    for (int direction = 0; direction < 3; ++direction) {
                        const Real cf = (viscous_traction[direction]
                            - normal_traction * normal[direction]) / dynamic_pressure;
                        fields(i,j,k,FieldCfX + direction) = cf;
                        cf_squared += cf * cf;
                    }
                    fields(i,j,k,FieldCfMagnitude) = std::sqrt(cf_squared);
                } else {
                    fields(i,j,k,FieldCp) = nan;
                    fields(i,j,k,FieldCfX) = nan;
                    fields(i,j,k,FieldCfY) = nan;
                    fields(i,j,k,FieldCfZ) = nan;
                    fields(i,j,k,FieldCfMagnitude) = nan;
                }
            });
        }

#ifdef AMREX_USE_GPU
        Gpu::streamSynchronize();
#endif
        const Real* dx = hypereb_level.geom.CellSize();
        const Real* prob_lo = hypereb_level.geom.ProbLo();
        for (MFIter mfi(state_new); mfi.isValid(); ++mfi) {
            const Box& box = mfi.validbox();
            if (flags[mfi].getType(box) == FabType::regular ||
                flags[mfi].getType(box) == FabType::covered) {
                continue;
            }
            Array4<const int> mask_array;
            const bool use_mask = static_cast<bool>(fine_mask);
            if (fine_mask) {
                mask_array = fine_mask->const_array(mfi);
            }
            writer.EBToPolygon(
                prob_lo, dx, box, flags[mfi].const_array(),
                boundary_centroid[mfi].const_array(),
                area_fraction[0]->operator[](mfi).const_array(),
                area_fraction[1]->operator[](mfi).const_array(),
                area_fraction[2]->operator[](mfi).const_array(),
                surface_fields.const_array(mfi), mask_array, use_mask);
        }
    }

    std::ostringstream directory_name;
    directory_name << surface_prefix << std::setw(5) << std::setfill('0')
                   << parent->levelSteps(0);
    const std::string directory = directory_name.str();
    writer.WriteEBVTP(ParallelDescriptor::MyProc(), directory, "eb");
    if (ParallelDescriptor::IOProcessor()) {
        hypereb::SurfaceWriter::WritePVTP(ParallelDescriptor::NProcs(), field_names,
                           directory, "eb");
        amrex::Print() << "[hyperEB EB surface] file=" << directory << "/eb.pvtp\n";
    }
    ParallelDescriptor::Barrier();
    writePhysicalWallSurfaceData(time, directory);
}

void
HyperEB::writePhysicalWallSurfaceData (Real time, const std::string& directory)
{
    std::array<Real,3> pressure_force{};
    std::array<Real,3> viscous_force{};
    std::array<Real,3> pressure_moment{};
    std::array<Real,3> viscous_moment{};
    Real surface_area = 0.0;
    Real heat_transfer = 0.0;
    std::vector<Real> local_samples;
    collectPhysicalWallData(
        time, pressure_force, viscous_force, pressure_moment, viscous_moment,
        surface_area, heat_transfer, &local_samples);
    std::vector<Real> samples = gather_rows(local_samples);

    if (ParallelDescriptor::IOProcessor()) {
        std::filesystem::create_directories(directory);
        const std::filesystem::path wall_file =
            std::filesystem::path(directory) / "physical_wall.csv";
        std::ofstream wall_stream(wall_file, std::ios::trunc);
        if (!wall_stream) {
            amrex::Abort("Unable to write physical-wall surface file: "
                         + wall_file.string());
        }
        for (std::size_t field = 0; field < physical_wall_names.size(); ++field) {
            if (field != 0) wall_stream << ',';
            wall_stream << physical_wall_names[field];
        }
        wall_stream << '\n' << std::setprecision(17);
        const std::size_t row_count = samples.size() / NumPhysicalWallFields;
        std::vector<std::size_t> order(row_count);
        for (std::size_t row = 0; row < row_count; ++row) order[row] = row;
        std::sort(order.begin(),order.end(),[&] (std::size_t left, std::size_t right) {
            const Real* a = samples.data() + left*NumPhysicalWallFields;
            const Real* b = samples.data() + right*NumPhysicalWallFields;
            if (a[WallDirection] != b[WallDirection]) return a[WallDirection] < b[WallDirection];
            if (a[WallSide] != b[WallSide]) return a[WallSide] < b[WallSide];
            if (a[WallX] != b[WallX]) return a[WallX] < b[WallX];
            if (a[WallY] != b[WallY]) return a[WallY] < b[WallY];
            return a[WallZ] < b[WallZ];
        });
        for (std::size_t row : order) {
            const Real* values = samples.data() + row*NumPhysicalWallFields;
            for (int field = 0; field < NumPhysicalWallFields; ++field) {
                if (field != 0) wall_stream << ',';
                wall_stream << values[field];
            }
            wall_stream << '\n';
        }
        amrex::Print() << "[hyperEB physical wall] file=" << wall_file.string() << '\n';
    }
    ParallelDescriptor::Barrier();
}
