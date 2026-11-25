// Source attribution and license obligations are recorded in THIRD_PARTY_NOTICES.md.
#include <hyperEB_EmbeddedBoundary.H>

using namespace amrex;

#include <eb_kernels/hyperEB_hydro_K.H>
#include <eb_kernels/hyperEB_hydro_eb_K.H>
#include <eb_kernels/hyperEB_diffusion_K.H>
#include <eb_kernels/hyperEB_diffusion_eb_K.H>
#include <eb_kernels/hyperEB_divop_K.H>
#include <eb_kernels/hyperEB_parm.H>

#include <AMReX_FArrayBox.H>
#include <AMReX_BoxIterator.H>
#include <AMReX_EBFArrayBox.H>
#include <AMReX_EB_Redistribution.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_GpuAtomic.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>

#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

namespace hypereb::kernels
{

    namespace
    {

        AMREX_GPU_DEVICE AMREX_FORCE_INLINE
        void eb_ctoprim(
            int i, int j, int k,
            const amrex::Array4<const amrex::Real> &state,
            const amrex::Array4<amrex::Real> &q,
            const amrex::Array4<const amrex::EBCellFlag> &flags,
            const Parm &parameters) noexcept
        {
            if (!flags(i, j, k).isCovered())
            {
                hypereb_ctoprim(i, j, k, state, q, parameters);
                return;
            }

            // Covered cells have no fluid state. Initialize scratch only;
            // open-face and connected-stencil kernels exclude these cells.
            for (int n = 0; n < NPRIM; ++n)
            {
                q(i, j, k, n) = 0.0;
            }
            q(i, j, k, QRHO) = 1.0;
            q(i, j, k, QPRES) = 1.0;
            q(i, j, k, QEINT) = 1.0 / (parameters.eos_gamma - 1.0);
            q(i, j, k, QCS) = std::sqrt(parameters.eos_gamma);
            q(i, j, k, QTEMP) = q(i, j, k, QEINT) / parameters.cv;
        }

        struct StageStateProbeConfig
        {
            bool enabled = false;
            std::vector<amrex::IntVect> cells;
        };

        const StageStateProbeConfig &stage_state_probe_config()
        {
            static const StageStateProbeConfig config = []
            {
                StageStateProbeConfig result;
                amrex::ParmParse diagnostics("diagnostics");
                int enabled = 0;
                diagnostics.query("stage_state_probe", enabled);
                result.enabled = enabled != 0;
                if (!result.enabled)
                {
                    return result;
                }

                amrex::Vector<int> indices;
                diagnostics.getarr("stage_state_probe_cells", indices);
                if (indices.empty() || indices.size() % AMREX_SPACEDIM != 0)
                {
                    amrex::Abort(
                        "diagnostics.stage_state_probe_cells must contain "
                        "one or more complete cell indices");
                }
                for (int offset = 0;
                     offset < static_cast<int>(indices.size());
                     offset += AMREX_SPACEDIM)
                {
                    result.cells.emplace_back(
                        AMREX_D_DECL(indices[offset], indices[offset + 1],
                                     indices[offset + 2]));
                }
                return result;
            }();
            return config;
        }

        bool eb_geometry_snapshot_enabled() noexcept
        {
            static const bool enabled = []
            {
                int value = 0;
                amrex::ParmParse diagnostics("diagnostics");
                diagnostics.query("eb_geometry_snapshot", value);
                return value != 0;
            }();
            return enabled;
        }

        struct SnapshotCellLess
        {
            bool operator()(const amrex::IntVect &a,
                            const amrex::IntVect &b) const noexcept
            {
                for (int d = 0; d < AMREX_SPACEDIM; ++d)
                {
                    if (a[d] != b[d])
                    {
                        return a[d] < b[d];
                    }
                }
                return false;
            }
        };

        struct SnapshotFace
        {
            int direction = 0;
            amrex::IntVect origin{AMREX_D_DECL(0, 0, 0)};
            amrex::Real area_fraction = 0.0;
            amrex::Real area = 0.0;
            bool open = false;
            bool eb_adjacent = false;
            std::array<amrex::Real, AMREX_SPACEDIM> centroid{};
            std::set<amrex::IntVect, SnapshotCellLess> member_cells;
            std::map<amrex::IntVect, int, SnapshotCellLess> member_sides;
            std::set<std::string> supports;
        };

        void write_json_real(std::ostream &out, amrex::Real value)
        {
            if (std::isfinite(value))
            {
                out << std::setprecision(17) << value;
            }
            else
            {
                out << "null";
            }
        }

        void write_json_cell(std::ostream &out, const amrex::IntVect &cell)
        {
            out << '[' << cell[0] << ',' << cell[1] << ',' << cell[2] << ']';
        }

        template <typename Array>
        void write_json_state(
            std::ostream &out,
            const Array &state,
            const amrex::IntVect &cell,
            bool valid)
        {
            out << '[';
            for (int n = 0; n < NCONS; ++n)
            {
                if (n != 0)
                {
                    out << ',';
                }
                write_json_real(out, valid ? state(cell[0], cell[1], cell[2], n)
                                           : std::numeric_limits<amrex::Real>::quiet_NaN());
            }
            out << ']';
        }

        void write_json_derived(
            std::ostream &out,
            const std::array<amrex::Real, NCONS> &state,
            const RegularPhysics &physics)
        {
            const amrex::Real rho = state[URHO];
            const amrex::Real velocity_squared = rho > 0.0
                ? (state[UMX] * state[UMX] + state[UMY] * state[UMY] +
                   state[UMZ] * state[UMZ]) / (rho * rho)
                : std::numeric_limits<amrex::Real>::quiet_NaN();
            const amrex::Real internal = rho > 0.0
                ? state[UEDEN] - 0.5 * rho * velocity_squared
                : std::numeric_limits<amrex::Real>::quiet_NaN();
            const amrex::Real pressure = (physics.gamma - 1.0) * internal;
            const amrex::Real temperature = rho > 0.0 && physics.cv > 0.0
                ? internal / (rho * physics.cv)
                : std::numeric_limits<amrex::Real>::quiet_NaN();
            const amrex::Real sound_speed = rho > 0.0 && pressure > 0.0
                ? std::sqrt(physics.gamma * pressure / rho)
                : std::numeric_limits<amrex::Real>::quiet_NaN();

            out << "{\"rho\":";
            write_json_real(out, rho);
            out << ",\"velocity\":[";
            write_json_real(out, rho > 0.0 ? state[UMX] / rho
                                           : std::numeric_limits<amrex::Real>::quiet_NaN());
            out << ',';
            write_json_real(out, rho > 0.0 ? state[UMY] / rho
                                           : std::numeric_limits<amrex::Real>::quiet_NaN());
            out << ',';
            write_json_real(out, rho > 0.0 ? state[UMZ] / rho
                                           : std::numeric_limits<amrex::Real>::quiet_NaN());
            out << "],\"pressure\":";
            write_json_real(out, pressure);
            out << ",\"temperature\":";
            write_json_real(out, temperature);
            out << ",\"sound_speed\":";
            write_json_real(out, sound_speed);
            out << '}';
        }

        void write_eb_geometry_snapshot(
            const amrex::Box &box,
            const amrex::IntVect &failure_cell,
            const amrex::Array4<const amrex::Real> &state,
            const amrex::Array4<const amrex::Real> &raw_rhs,
            const amrex::Array4<const amrex::Real> &stabilized_rhs,
            const EBGeometryViews &eb,
            const amrex::Geometry &geometry,
            const RegularPhysics &physics,
            amrex::Real stage_dt,
            amrex::Real time)
        {
            static std::mutex snapshot_mutex;
            static bool snapshot_written = false;
            std::lock_guard<std::mutex> lock(snapshot_mutex);
            if (snapshot_written)
            {
                return;
            }
            snapshot_written = true;
#ifdef AMREX_USE_GPU
            amrex::Gpu::streamSynchronize();
#endif
            if (!amrex::ParallelDescriptor::IOProcessor())
            {
                return;
            }

            const amrex::Box state_box = amrex::grow(box, 5);
            const amrex::Box rhs_box = amrex::grow(box, 3);
            const amrex::Box data_box = amrex::grow(box, 5);
            amrex::IArrayBox tracker(data_box, 8);
            amrex::FArrayBox nrs(data_box, 1);
            amrex::FArrayBox alpha(amrex::grow(box, 4), 2);
            amrex::FArrayBox neighborhood_volume(data_box, 1);
            amrex::FArrayBox neighborhood_centroid(data_box, AMREX_SPACEDIM);
            amrex::MakeITracker(
                box,
                AMREX_D_DECL(eb.area_fraction[0], eb.area_fraction[1],
                             eb.area_fraction[2]),
                eb.volume_fraction, tracker.array(), geometry,
                physics.srd_target_volfrac);
            amrex::MakeStateRedistUtils(
                box, eb.flags, eb.volume_fraction, eb.cell_centroid,
                tracker.const_array(), nrs.array(), alpha.array(),
                neighborhood_volume.array(), neighborhood_centroid.array(),
                geometry, physics.srd_target_volfrac);
#ifdef AMREX_USE_GPU
            amrex::Gpu::streamSynchronize();
#endif

            static constexpr std::array<int, 27> imap{
                0,-1,0,1,-1,1,-1,0,1,-1,0,1,-1,0,1,-1,0,1,-1,0,1,-1,0,1,-1,0,1};
            static constexpr std::array<int, 27> jmap{
                0,-1,-1,-1,0,0,1,1,1,-1,-1,-1,0,0,0,1,1,1,-1,-1,-1,0,0,0,1,1,1};
            static constexpr std::array<int, 27> kmap{
                0,0,0,0,0,0,0,0,0,-1,-1,-1,-1,-1,-1,-1,-1,-1,1,1,1,1,1,1,1,1,1};
            const auto itr = tracker.const_array();
            const auto nr = nrs.const_array();
            const auto al = alpha.const_array();
            const auto nbhd = neighborhood_volume.const_array();
            auto is_fluid = [&](const amrex::IntVect &cell)
            {
                return data_box.contains(cell) && geometry.Domain().contains(cell) &&
                       eb.volume_fraction(cell[0], cell[1], cell[2]) > 0.0 &&
                       !eb.flags(cell[0], cell[1], cell[2]).isCovered();
            };
            auto connected = [&](const amrex::IntVect &source,
                                 const amrex::IntVect &target,
                                 int direction,
                                 int sign)
            {
                const auto &source_flag = eb.flags(source[0], source[1], source[2]);
                const auto &target_flag = eb.flags(target[0], target[1], target[2]);
                const int sx = direction == 0 ? sign : 0;
                const int sy = direction == 1 ? sign : 0;
                const int sz = direction == 2 ? sign : 0;
                return source_flag.isConnected(sx, sy, sz) &&
                       target_flag.isConnected(-sx, -sy, -sz);
            };

            std::vector<std::set<amrex::IntVect, SnapshotCellLess>> supports(4);
            supports[0].insert(failure_cell);
            if (box.contains(failure_cell))
            {
                for (int member = 1; member <= itr(failure_cell[0], failure_cell[1], failure_cell[2], 0); ++member)
                {
                    const int code = itr(failure_cell[0], failure_cell[1], failure_cell[2], member);
                    const amrex::IntVect cell(AMREX_D_DECL(
                        failure_cell[0] + imap[code],
                        failure_cell[1] + jmap[code],
                        failure_cell[2] + kmap[code]));
                    if (is_fluid(cell))
                    {
                        supports[0].insert(cell);
                    }
                }
            }
            for (int ring = 1; ring < 4; ++ring)
            {
                supports[ring] = supports[ring - 1];
                const auto frontier = supports[ring - 1];
                for (const auto &source : frontier)
                {
                    for (int direction = 0; direction < AMREX_SPACEDIM; ++direction)
                    {
                        for (int sign : {-1, 1})
                        {
                            amrex::IntVect target = source;
                            target[direction] += sign;
                            if (is_fluid(target) && connected(source, target, direction, sign))
                            {
                                supports[ring].insert(target);
                            }
                        }
                    }
                }
            }

            std::map<std::tuple<int, int, int, int>, SnapshotFace> faces;
            const auto spacing = geometry.CellSizeArray();
            for (int ring = 0; ring < 4; ++ring)
            {
                const std::string support_name = "N" + std::to_string(ring);
                for (const auto &member : supports[ring])
                {
                    for (int direction = 0; direction < AMREX_SPACEDIM; ++direction)
                    {
                        const amrex::Box face_box = detail::eb_face_flux_box(box, direction);
                        for (int side : {-1, 1})
                        {
                            amrex::IntVect origin = member;
                            if (side > 0)
                            {
                                origin[direction] += 1;
                            }
                            if (!face_box.contains(origin))
                            {
                                continue;
                            }
                            const auto key = std::make_tuple(
                                direction, origin[0], origin[1], origin[2]);
                            auto found = faces.find(key);
                            if (found == faces.end())
                            {
                                SnapshotFace face;
                                face.direction = direction;
                                face.origin = origin;
                                face.area_fraction = eb.area_fraction[direction](
                                    origin[0], origin[1], origin[2]);
                                int transverse_one = (direction + 1) % AMREX_SPACEDIM;
                                int transverse_two = (direction + 2) % AMREX_SPACEDIM;
                                face.area = face.area_fraction * spacing[transverse_one] *
                                            spacing[transverse_two];
                                face.open = face.area_fraction > 0.0;
                                for (int d = 0; d < AMREX_SPACEDIM; ++d)
                                {
                                    face.centroid[d] = eb.face_centroid[direction](
                                        origin[0], origin[1], origin[2], d);
                                }
                                const amrex::IntVect lower = origin - amrex::BASISV(direction);
                                const amrex::IntVect upper = origin;
                                face.eb_adjacent =
                                    (data_box.contains(lower) && !eb.flags(lower[0], lower[1], lower[2]).isRegular()) ||
                                    (data_box.contains(upper) && !eb.flags(upper[0], upper[1], upper[2]).isRegular());
                                found = faces.emplace(key, std::move(face)).first;
                            }
                            found->second.member_cells.insert(member);
                            found->second.member_sides[member] = side;
                            found->second.supports.insert(support_name);
                        }
                    }
                }
            }

            auto state_at = [&](const amrex::IntVect &cell, bool stabilized)
            {
                std::array<amrex::Real, NCONS> result{};
                for (int n = 0; n < NCONS; ++n)
                {
                    const amrex::Real rhs_value =
                        rhs_box.contains(cell)
                            ? (stabilized ? stabilized_rhs(cell[0], cell[1], cell[2], n)
                                          : raw_rhs(cell[0], cell[1], cell[2], n))
                            : 0.0;
                    result[n] = state_box.contains(cell)
                        ? state(cell[0], cell[1], cell[2], n) + stage_dt * rhs_value
                        : std::numeric_limits<amrex::Real>::quiet_NaN();
                }
                return result;
            };

            auto write_json_array = [](
                std::ostream &out,
                const std::array<amrex::Real, NCONS> &values)
            {
                out << '[';
                for (int n = 0; n < NCONS; ++n)
                {
                    if (n != 0)
                    {
                        out << ',';
                    }
                    write_json_real(out, values[n]);
                }
                out << ']';
            };

            std::ofstream output("eb_geometry_snapshot.json", std::ios::trunc);
            if (!output)
            {
                amrex::Abort("diagnostics.eb_geometry_snapshot could not open eb_geometry_snapshot.json");
            }
            output << std::setprecision(17);
            output << "{\"schema\":\"eb_geometry_snapshot_v1\",\"time\":";
            write_json_real(output, time);
            output << ",\"stage_dt\":";
            write_json_real(output, stage_dt);
            output << ",\"cell_size\":[";
            for (int d = 0; d < AMREX_SPACEDIM; ++d)
            {
                if (d != 0) output << ',';
                write_json_real(output, spacing[d]);
            }
            output << "],\"failure_cell\":";
            write_json_cell(output, failure_cell);
            output << ",\"failure_stage_state\":{\"state\":";
            write_json_state(output, state, failure_cell, state_box.contains(failure_cell));
            output << ",\"qstar\":";
            write_json_array(output, state_at(failure_cell, false));
            output << ",\"post_stabilization\":";
            write_json_array(output, state_at(failure_cell, true));
            output << "},\"supports\":[";
            for (int ring = 0; ring < 4; ++ring)
            {
                if (ring != 0) output << ',';
                const auto &members = supports[ring];
                amrex::Real native_effective_volume = ring == 0 && box.contains(failure_cell)
                    ? nbhd(failure_cell[0], failure_cell[1], failure_cell[2])
                    : 0.0;
                amrex::Real member_volume_sum = 0.0;
                for (const auto &member : members)
                {
                    member_volume_sum += eb.volume_fraction(member[0], member[1], member[2]);
                }
                output << "{\"support\":\"N" << ring << "\",\"member_count\":"
                       << members.size() << ",\"native_effective_volume_fraction\":";
                write_json_real(output, native_effective_volume);
                output << ",\"member_volume_fraction_sum\":";
                write_json_real(output, member_volume_sum);
                output << ",\"members\":[";
                bool first_member = true;
                for (const auto &member : members)
                {
                    if (!first_member) output << ',';
                    first_member = false;
                    const auto qstar = state_at(member, false);
                    const auto post = state_at(member, true);
                    amrex::Real weight = 1.0;
                    if (ring == 0 && member == failure_cell && box.contains(failure_cell))
                    {
                        weight = al(failure_cell[0], failure_cell[1], failure_cell[2], 0);
                    }
                    else if (ring == 0 && box.contains(failure_cell))
                    {
                        const Real nrs_value = nr(member[0], member[1], member[2]);
                        weight = nrs_value > 0.0
                            ? al(failure_cell[0], failure_cell[1], failure_cell[2], 1) / nrs_value
                            : 0.0;
                    }
                    output << "{\"cell\":";
                    write_json_cell(output, member);
                    output << ",\"vfrac\":";
                    write_json_real(output, eb.volume_fraction(member[0], member[1], member[2]));
                    output << ",\"volume\":";
                    write_json_real(output, eb.volume_fraction(member[0], member[1], member[2]) *
                                              spacing[0] * spacing[1] * spacing[2]);
                    output << ",\"weight\":";
                    write_json_real(output, weight);
                    output << ",\"qstar\": [";
                    for (int n = 0; n < NCONS; ++n)
                    {
                        if (n != 0) output << ',';
                        write_json_real(output, qstar[n]);
                    }
                    output << "],\"qstar_primitive\":";
                    write_json_derived(output, qstar, physics);
                    output << ",\"post_stabilization\": [";
                    for (int n = 0; n < NCONS; ++n)
                    {
                        if (n != 0) output << ',';
                        write_json_real(output, post[n]);
                    }
                    output << "]}";
                }
                output << "]}";
            }
            output << "],\"faces\":[";
            bool first_face = true;
            for (const auto &entry : faces)
            {
                if (!first_face) output << ',';
                first_face = false;
                const SnapshotFace &face = entry.second;
                output << "{\"id\":\"d" << face.direction << ':'
                       << face.origin[0] << ':' << face.origin[1] << ':'
                       << face.origin[2] << "\",\"direction\":"
                       << face.direction << ",\"origin\":";
                write_json_cell(output, face.origin);
                output << ",\"area\":";
                write_json_real(output, face.area);
                output << ",\"area_fraction\":";
                write_json_real(output, face.area_fraction);
                output << ",\"open_area\":";
                write_json_real(output, face.area);
                output << ",\"open\":" << (face.open ? "true" : "false")
                       << ",\"eb_adjacent\":" << (face.eb_adjacent ? "true" : "false")
                       << ",\"centroid\":[";
                for (int d = 0; d < AMREX_SPACEDIM; ++d)
                {
                    if (d != 0) output << ',';
                    write_json_real(output, face.centroid[d]);
                }
                output << "],\"member_cells\":[";
                bool first_cell = true;
                for (const auto &member : face.member_cells)
                {
                    if (!first_cell) output << ',';
                    first_cell = false;
                    write_json_cell(output, member);
                }
                output << "],\"member_mapping\":[";
                bool first_mapping = true;
                for (const auto &member_side : face.member_sides)
                {
                    if (!first_mapping) output << ',';
                    first_mapping = false;
                    output << "{\"cell\":";
                    write_json_cell(output, member_side.first);
                    output << ",\"side\":" << member_side.second << "}";
                }
                output << "],\"supports\":[";
                bool first_support = true;
                for (const auto &support : face.supports)
                {
                    if (!first_support) output << ',';
                    first_support = false;
                    output << '"' << support << '"';
                }
                output << "]}";
            }
            output << "]}\n";
        }

        void report_stage_state_probe(
            const amrex::Box &box,
            const amrex::Array4<const amrex::Real> &state,
            const amrex::Array4<const amrex::Real> &raw_rhs,
            const amrex::Array4<const amrex::Real> &stabilized_rhs,
            const amrex::Array4<const amrex::Real> &volume_fraction,
            const RegularPhysics &physics,
            amrex::Real stage_dt,
            amrex::Real time)
        {
            const auto &config = stage_state_probe_config();
            if (!config.enabled)
            {
                return;
            }

            constexpr int RawRho = 0;
            constexpr int RawEint = 1;
            constexpr int RawPressure = 2;
            constexpr int RawTemperature = 3;
            constexpr int RawInvalid = 4;
            constexpr int PostRho = 5;
            constexpr int PostEint = 6;
            constexpr int PostPressure = 7;
            constexpr int PostTemperature = 8;
            constexpr int PostInvalid = 9;
            constexpr int VolumeFraction = 10;
            constexpr int NumValues = 11;

            for (const auto &cell : config.cells)
            {
                if (!box.contains(cell))
                {
                    continue;
                }

                amrex::Gpu::DeviceVector<amrex::Real> device_values(NumValues);
                auto *const values = device_values.data();
                const int i = cell[0];
                const int j = cell[1];
                const int k = cell[2];
                const amrex::Real gamma_minus_one = physics.gamma - 1.0;
                const amrex::Real cv = physics.cv;

                amrex::ParallelFor(
                    1,
                    [=] AMREX_GPU_DEVICE(int) noexcept
                    {
                        auto candidate = [=] AMREX_GPU_DEVICE(
                                             const amrex::Array4<const amrex::Real> &rhs,
                                             int rho_slot, int eint_slot,
                                             int pressure_slot, int temperature_slot,
                                             int invalid_slot) noexcept
                        {
                            const amrex::Real rho =
                                state(i, j, k, URHO) + stage_dt * rhs(i, j, k, URHO);
                            const amrex::Real mx =
                                state(i, j, k, UMX) + stage_dt * rhs(i, j, k, UMX);
                            const amrex::Real my =
                                state(i, j, k, UMY) + stage_dt * rhs(i, j, k, UMY);
                            const amrex::Real mz =
                                state(i, j, k, UMZ) + stage_dt * rhs(i, j, k, UMZ);
                            const amrex::Real total_energy =
                                state(i, j, k, UEDEN) + stage_dt * rhs(i, j, k, UEDEN);
                            const amrex::Real kinetic = rho > 0.0
                                ? 0.5 * (mx * mx + my * my + mz * mz) / rho
                                : 0.0;
                            const amrex::Real internal_energy = total_energy - kinetic;
                            const amrex::Real pressure = gamma_minus_one * internal_energy;
                            const amrex::Real temperature = rho > 0.0
                                ? internal_energy / (rho * cv)
                                : -1.0;
                            const bool invalid =
                                !std::isfinite(rho) || !std::isfinite(internal_energy) ||
                                !(rho > 0.0) || !(internal_energy > 0.0);
                            values[rho_slot] = rho;
                            values[eint_slot] = internal_energy;
                            values[pressure_slot] = pressure;
                            values[temperature_slot] = temperature;
                            values[invalid_slot] = invalid ? 1.0 : 0.0;
                        };

                        candidate(raw_rhs, RawRho, RawEint, RawPressure,
                                  RawTemperature, RawInvalid);
                        candidate(stabilized_rhs, PostRho, PostEint, PostPressure,
                                  PostTemperature, PostInvalid);
                        values[VolumeFraction] = volume_fraction(i, j, k);
                    });

                amrex::Gpu::streamSynchronize();
                amrex::Vector<amrex::Real> host_values(NumValues);
                amrex::Gpu::copy(
                    amrex::Gpu::deviceToHost,
                    device_values.begin(), device_values.end(),
                    host_values.begin());

#ifdef _OPENMP
#pragma omp critical(zeb_stage_state_probe)
#endif
                {
                    amrex::Print().SetPrecision(17)
                        << "[hyperEB stage state probe] method=weighted_state"
                        << " time=" << time
                        << " stage_dt=" << stage_dt
                        << " cell=(" << i << ',' << j << ',' << k << ')'
                        << " vfrac=" << host_values[VolumeFraction]
                        << " raw_rho=" << host_values[RawRho]
                        << " raw_rhoe=" << host_values[RawEint]
                        << " raw_p=" << host_values[RawPressure]
                        << " raw_T=" << host_values[RawTemperature]
                        << " raw_invalid=" << static_cast<int>(host_values[RawInvalid])
                        << " post_rho=" << host_values[PostRho]
                        << " post_rhoe=" << host_values[PostEint]
                        << " post_p=" << host_values[PostPressure]
                        << " post_T=" << host_values[PostTemperature]
                        << " post_invalid=" << static_cast<int>(host_values[PostInvalid])
                        << '\n';
                }
            }
        }

        Parm make_eb_parameters(const amrex::Geometry &geometry,
                                const RegularPhysics &physics,
                                amrex::Real time)
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
            // A stale constant coefficient in an input file must not
            // override the selected variable law at an isothermal EB wall.
            parameters.const_lambda = physics.use_constant_viscosity
                ? physics.constant_lambda : -1.0;
            parameters.eb_thermal_boundary = physics.eb_thermal_boundary == EBThermalBoundary::isothermal
                                                 ? Parm::thermal_isothermal
                                                 : Parm::thermal_adiabatic;
            parameters.eb_wall_temperature = physics.eb_wall_temperature;
            parameters.eb_noslip_segment = physics.eb_noslip_segment;
            parameters.eb_noslip_origin = physics.eb_noslip_origin;
            parameters.eb_noslip_tangent = physics.eb_noslip_tangent;
            parameters.eb_noslip_s_lo = physics.eb_noslip_s_lo;
            parameters.eb_noslip_s_hi = physics.eb_noslip_s_hi;
            parameters.prob_lo = physics.problem_low;
            parameters.cell_size = geometry.CellSizeArray();
            parameters.wall_translation = physics.wall_translation;
            for (int direction = 0; direction < 3; ++direction)
            {
                parameters.wall_translation[direction] += physics.wall_acceleration[direction] * time;
            }
            return parameters;
        }

        void require_stabilized_candidate_admissible(
            const amrex::Box &box,
            const amrex::Array4<const amrex::Real> &state,
            const amrex::Array4<const amrex::Real> &raw_rhs,
            const amrex::Array4<const amrex::Real> &stabilized_rhs,
            const EBGeometryViews &eb,
            const amrex::Geometry &geometry,
            const RegularPhysics &physics,
            amrex::Real stage_dt,
            amrex::Real time)
        {
            amrex::Gpu::DeviceScalar<int> failure_count(0);
            amrex::Gpu::DeviceScalar<int> failure_i(0);
            amrex::Gpu::DeviceScalar<int> failure_j(0);
            amrex::Gpu::DeviceScalar<int> failure_k(0);
            amrex::Gpu::DeviceScalar<amrex::Real> failure_rho(0.0);
            amrex::Gpu::DeviceScalar<amrex::Real> failure_mx(0.0);
            amrex::Gpu::DeviceScalar<amrex::Real> failure_my(0.0);
            amrex::Gpu::DeviceScalar<amrex::Real> failure_mz(0.0);
            amrex::Gpu::DeviceScalar<amrex::Real> failure_eden(0.0);
            amrex::Gpu::DeviceScalar<amrex::Real> failure_eint(0.0);
            amrex::Gpu::DeviceScalar<amrex::Real> failure_raw_eint(0.0);
            amrex::Gpu::DeviceScalar<amrex::Real> failure_raw_rho(0.0);
            amrex::Gpu::DeviceScalar<amrex::Real> failure_vfrac(0.0);
            amrex::Gpu::DeviceScalar<int> failure_raw_invalid(0);

            int *const count = failure_count.dataPtr();
            int *const bad_i = failure_i.dataPtr();
            int *const bad_j = failure_j.dataPtr();
            int *const bad_k = failure_k.dataPtr();
            amrex::Real *const bad_rho = failure_rho.dataPtr();
            amrex::Real *const bad_mx = failure_mx.dataPtr();
            amrex::Real *const bad_my = failure_my.dataPtr();
            amrex::Real *const bad_mz = failure_mz.dataPtr();
            amrex::Real *const bad_eden = failure_eden.dataPtr();
            amrex::Real *const bad_eint = failure_eint.dataPtr();
            amrex::Real *const bad_raw_eint = failure_raw_eint.dataPtr();
            amrex::Real *const bad_raw_rho = failure_raw_rho.dataPtr();
            amrex::Real *const bad_vfrac = failure_vfrac.dataPtr();
            int *const bad_raw_invalid = failure_raw_invalid.dataPtr();
            const auto flags = eb.flags;
            const auto volume_fraction = eb.volume_fraction;

            amrex::ParallelFor(
                box,
                [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                {
                    if (flags(i, j, k).isCovered())
                    {
                        return;
                    }

                    const amrex::Real rho = state(i, j, k, URHO) + stage_dt * stabilized_rhs(i, j, k, URHO);
                    const amrex::Real mx = state(i, j, k, UMX) + stage_dt * stabilized_rhs(i, j, k, UMX);
                    const amrex::Real my = state(i, j, k, UMY) + stage_dt * stabilized_rhs(i, j, k, UMY);
                    const amrex::Real mz = state(i, j, k, UMZ) + stage_dt * stabilized_rhs(i, j, k, UMZ);
                    const amrex::Real eden = state(i, j, k, UEDEN) + stage_dt * stabilized_rhs(i, j, k, UEDEN);
                    const amrex::Real kinetic = rho > 0.0
                                                    ? 0.5 * (mx * mx + my * my + mz * mz) / rho
                                                    : 0.0;
                    const amrex::Real eint = eden - kinetic;
                    const bool invalid =
                        !std::isfinite(rho) || !std::isfinite(mx) ||
                        !std::isfinite(my) || !std::isfinite(mz) ||
                        !std::isfinite(eden) || !std::isfinite(eint) ||
                        !(rho > 0.0) || !(eint > 0.0);

                    if (invalid &&
                        amrex::HostDevice::Atomic::FetchAdd(count, 1) == 0)
                    {
                        const amrex::Real raw_rho = state(i, j, k, URHO) + stage_dt * raw_rhs(i, j, k, URHO);
                        const amrex::Real raw_mx = state(i, j, k, UMX) + stage_dt * raw_rhs(i, j, k, UMX);
                        const amrex::Real raw_my = state(i, j, k, UMY) + stage_dt * raw_rhs(i, j, k, UMY);
                        const amrex::Real raw_mz = state(i, j, k, UMZ) + stage_dt * raw_rhs(i, j, k, UMZ);
                        const amrex::Real raw_eden = state(i, j, k, UEDEN) + stage_dt * raw_rhs(i, j, k, UEDEN);
                        const amrex::Real raw_kinetic = raw_rho > 0.0
                                                          ? 0.5 * (raw_mx * raw_mx + raw_my * raw_my + raw_mz * raw_mz) / raw_rho
                                                          : 0.0;
                        const amrex::Real raw_eint = raw_eden - raw_kinetic;
                        const bool raw_invalid =
                            !std::isfinite(raw_rho) || !std::isfinite(raw_mx) ||
                            !std::isfinite(raw_my) || !std::isfinite(raw_mz) ||
                            !std::isfinite(raw_eden) || !std::isfinite(raw_eint) ||
                            !(raw_rho > 0.0) || !(raw_eint > 0.0);

                        *bad_i = i;
                        *bad_j = j;
                        *bad_k = k;
                        *bad_rho = rho;
                        *bad_mx = mx;
                        *bad_my = my;
                        *bad_mz = mz;
                        *bad_eden = eden;
                        *bad_eint = eint;
                        *bad_raw_eint = raw_eint;
                        *bad_raw_rho = raw_rho;
                        *bad_vfrac = volume_fraction(i, j, k);
                        *bad_raw_invalid = raw_invalid ? 1 : 0;
                    }
                });

            amrex::Gpu::streamSynchronize();
            if (failure_count.dataValue() != 0)
            {
                if (eb_geometry_snapshot_enabled())
                {
                    write_eb_geometry_snapshot(
                        box,
                        amrex::IntVect(AMREX_D_DECL(
                            failure_i.dataValue(), failure_j.dataValue(),
                            failure_k.dataValue())),
                        state,
                        raw_rhs,
                        stabilized_rhs,
                        eb,
                        geometry,
                        physics,
                        stage_dt,
                        time);
                }
                std::ostringstream message;
                message << "Invalid EB forward-Euler candidate after WSR"
                        << ": cell=(" << failure_i.dataValue() << ','
                        << failure_j.dataValue() << ',' << failure_k.dataValue()
                        << ") rho=" << failure_rho.dataValue()
                        << " momentum=(" << failure_mx.dataValue() << ','
                        << failure_my.dataValue() << ',' << failure_mz.dataValue()
                        << ") rhoE=" << failure_eden.dataValue()
                        << " rhoe_total=" << failure_eint.dataValue()
                        << " raw_candidate_invalid=" << failure_raw_invalid.dataValue()
                        << " raw_rho=" << failure_raw_rho.dataValue()
                        << " raw_rhoe_total=" << failure_raw_eint.dataValue()
                        << " raw_p=" << (physics.gamma - 1.0) * failure_raw_eint.dataValue()
                        << " post_p=" << (physics.gamma - 1.0) * failure_eint.dataValue()
                        << " volfrac=" << failure_vfrac.dataValue()
                        << " stage_dt=" << stage_dt;
                amrex::Abort(message.str());
            }
        }

    } // namespace

    MovingWallCflEstimate estimate_moving_wall_cfl(
        const amrex::MultiFab &state,
        const amrex::Geometry &geometry,
        const RegularPhysics &physics,
        amrex::Real time,
        amrex::Real cfl,
        bool collect_details)
    {
        constexpr int NormalDt = 0;
        constexpr int FullDt = 1;
        constexpr int VolumeFraction = 2;
        constexpr int SoundSpeed = 3;
        constexpr int FluidVelocity = 4;
        constexpr int WallVelocity = 7;
        constexpr int EbNormal = 10;
        constexpr int NormalRelativeSpeed = 13;
        constexpr int FullRelativeSpeed = 14;
        constexpr int NumMetrics = 15;
        constexpr amrex::Real small_density = 1.0e-19;
        constexpr amrex::Real small_pressure = 1.0e-10;

        const auto &factory =
            dynamic_cast<const amrex::EBFArrayBoxFactory &>(state.Factory());
        const auto &flags = factory.getMultiEBCellFlagFab();
        const auto &volume_fraction = factory.getVolFrac();
        const auto &boundary_centroid = factory.getBndryCent();
        const auto area_fraction = factory.getAreaFrac();
        const auto parameters = make_eb_parameters(geometry, physics, time);
        const auto dx = geometry.CellSizeArray();
        const amrex::Real h = amrex::min(dx[0], amrex::min(dx[1], dx[2]));

        const int metric_components = collect_details ? NumMetrics : 2;
        amrex::MultiFab metrics(
            state.boxArray(), state.DistributionMap(), metric_components, 0);
        metrics.setVal(0.0);
        metrics.setVal(std::numeric_limits<amrex::Real>::max(), NormalDt, 2, 0);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (amrex::MFIter mfi(state, amrex::TilingIfNotGPU());
             mfi.isValid(); ++mfi)
        {
            const amrex::Box &box = mfi.tilebox();
            const auto tile_type = flags[mfi].getType(box);
            if (tile_type == amrex::FabType::regular ||
                tile_type == amrex::FabType::covered)
            {
                continue;
            }

            const auto s = state.const_array(mfi);
            const auto flag = flags.const_array(mfi);
            const auto vfrac = volume_fraction.const_array(mfi);
            const auto bcent = boundary_centroid.const_array(mfi);
            const auto apx = area_fraction[0]->const_array(mfi);
            const auto apy = area_fraction[1]->const_array(mfi);
            const auto apz = area_fraction[2]->const_array(mfi);
            const auto output = metrics.array(mfi);
            const amrex::Real gamma = physics.gamma;

            amrex::ParallelFor(
                box,
                [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                {
                    if (!flag(i, j, k).isSingleValued())
                    {
                        return;
                    }

                    amrex::GpuArray<amrex::Real, 3> wall_velocity;
                    hypereb_wall_velocity(
                        i, j, k, bcent, parameters, wall_velocity);
                    const amrex::Real wall_speed = std::sqrt(
                        wall_velocity[0] * wall_velocity[0] +
                        wall_velocity[1] * wall_velocity[1] +
                        wall_velocity[2] * wall_velocity[2]);
                    if (!(wall_speed > 0.0))
                    {
                        return;
                    }

                    const amrex::Real area_x = apx(i, j, k) - apx(i + 1, j, k);
                    const amrex::Real area_y = apy(i, j, k) - apy(i, j + 1, k);
                    const amrex::Real area_z = apz(i, j, k) - apz(i, j, k + 1);
                    const amrex::Real area_norm = std::sqrt(
                        area_x * area_x + area_y * area_y + area_z * area_z);
                    if (!(area_norm > 0.0))
                    {
                        return;
                    }

                    const amrex::Real rho = amrex::max(
                        s(i, j, k, URHO), small_density);
                    const amrex::Real rho_inverse = 1.0 / rho;
                    const amrex::Real pressure = amrex::max(
                        (gamma - 1.0) * s(i, j, k, UEINT), small_pressure);
                    const amrex::Real sound_speed =
                        std::sqrt(gamma * pressure * rho_inverse);
                    const amrex::Real u = s(i, j, k, UMX) * rho_inverse;
                    const amrex::Real v = s(i, j, k, UMY) * rho_inverse;
                    const amrex::Real w = s(i, j, k, UMZ) * rho_inverse;
                    const amrex::Real normal_x = area_x / area_norm;
                    const amrex::Real normal_y = area_y / area_norm;
                    const amrex::Real normal_z = area_z / area_norm;
                    const amrex::Real relative_x = u - wall_velocity[0];
                    const amrex::Real relative_y = v - wall_velocity[1];
                    const amrex::Real relative_z = w - wall_velocity[2];
                    const amrex::Real normal_relative_speed = std::abs(
                        relative_x * normal_x + relative_y * normal_y +
                        relative_z * normal_z);
                    const amrex::Real full_relative_speed = std::sqrt(
                        relative_x * relative_x + relative_y * relative_y +
                        relative_z * relative_z);

                    output(i, j, k, NormalDt) =
                        cfl * h / (sound_speed + normal_relative_speed);
                    output(i, j, k, FullDt) =
                        cfl * h / (sound_speed + full_relative_speed);
                    if (collect_details)
                    {
                        output(i, j, k, VolumeFraction) = vfrac(i, j, k);
                        output(i, j, k, SoundSpeed) = sound_speed;
                        output(i, j, k, FluidVelocity + 0) = u;
                        output(i, j, k, FluidVelocity + 1) = v;
                        output(i, j, k, FluidVelocity + 2) = w;
                        output(i, j, k, WallVelocity + 0) = wall_velocity[0];
                        output(i, j, k, WallVelocity + 1) = wall_velocity[1];
                        output(i, j, k, WallVelocity + 2) = wall_velocity[2];
                        output(i, j, k, EbNormal + 0) = normal_x;
                        output(i, j, k, EbNormal + 1) = normal_y;
                        output(i, j, k, EbNormal + 2) = normal_z;
                        output(i, j, k, NormalRelativeSpeed) =
                            normal_relative_speed;
                        output(i, j, k, FullRelativeSpeed) = full_relative_speed;
                    }
                });
        }

        MovingWallCflEstimate result;
        result.normal_dt = metrics.min(NormalDt);
        result.full_dt = metrics.min(FullDt);
        if (!collect_details ||
            result.normal_dt == std::numeric_limits<amrex::Real>::max())
        {
            return result;
        }

        result.limiting_cell = metrics.minIndex(NormalDt);
        const amrex::Box selected(result.limiting_cell, result.limiting_cell);
        result.volume_fraction = metrics.min(selected, VolumeFraction);
        result.sound_speed = metrics.min(selected, SoundSpeed);
        for (int direction = 0; direction < 3; ++direction)
        {
            result.fluid_velocity[direction] =
                metrics.min(selected, FluidVelocity + direction);
            result.wall_velocity[direction] =
                metrics.min(selected, WallVelocity + direction);
            result.eb_normal[direction] =
                metrics.min(selected, EbNormal + direction);
        }
        result.normal_relative_speed =
            metrics.min(selected, NormalRelativeSpeed);
        result.full_relative_speed =
            metrics.min(selected, FullRelativeSpeed);
        return result;
    }

#ifdef HYPEREB_ENABLE_RUNTIME_CHECKS
    namespace
    {

        void require_finite(const amrex::FArrayBox &values, const amrex::Box &box,
                            int components, const char *stage)
        {
            amrex::Gpu::streamSynchronize();
            amrex::IntVect location;
            const bool has_nan = values.contains_nan(box, 0, components, location);
            const bool has_inf = !has_nan && values.contains_inf(box, 0, components, location);
            if (has_nan || has_inf)
            {
                std::ostringstream message;
                message << "Non-finite EB value after " << stage << " at " << location;
                amrex::Abort(message.str());
            }
        }

    } // namespace
#endif

    void compute_eb_rhs(
        const amrex::Box &box,
        const amrex::Array4<amrex::Real> &derivative,
        const amrex::Array4<const amrex::Real> &state,
        const std::array<amrex::Array4<amrex::Real>, 3> &output_flux,
        const amrex::Geometry &geometry,
        const RegularPhysics &physics,
        const EBGeometryViews &eb,
        const EBCouplingViews &coupling,
        const amrex::BCRec *boundary_conditions,
        amrex::Real stage_dt,
        amrex::Real reflux_weight,
        amrex::Real time)
    {
        BL_PROFILE("hypereb::kernels::compute_eb_rhs");

        const amrex::Box box_grown_1 = amrex::grow(box, 1);
        const amrex::Box box_grown_2 = amrex::grow(box, 2);
        const amrex::Box box_grown_3 = amrex::grow(box, 3);
        const amrex::Box box_grown_5 = amrex::grow(box, 5);
        const auto inverse_spacing = geometry.InvCellSizeArray();

        const Parm parameters = make_eb_parameters(geometry, physics, time);
        amrex::FArrayBox conservative_divergence(box_grown_3, NCONS);
        conservative_divergence.setVal<amrex::RunOn::Device>(0.0);

        amrex::FArrayBox primitive(box_grown_5, NPRIM);
        amrex::FArrayBox diffusion_coefficient;
        if (physics.do_diffusion)
        {
            diffusion_coefficient.resize(box_grown_3, NCOEF);
        }

        std::array<amrex::FArrayBox, 3> face_flux;
        for (int direction = 0; direction < 3; ++direction)
        {
            face_flux[direction].resize(detail::eb_face_flux_box(box, direction), NCONS);
            face_flux[direction].setVal<amrex::RunOn::Device>(0.0);

            const amrex::Box output_box = amrex::surroundingNodes(box, direction);
            const auto output = output_flux[direction];
            amrex::ParallelFor(output_box, NCONS,
                               [=] AMREX_GPU_DEVICE(int i, int j, int k, int component) noexcept
                               {
                                   output(i, j, k, component) = 0.0;
                               });
        }

        if (coupling.as_fine != 0)
        {
            const auto fine_mass_change = coupling.fine_mass_change;
            amrex::ParallelFor(box_grown_1, NCONS,
                               [=] AMREX_GPU_DEVICE(int i, int j, int k, int component) noexcept
                               {
                                   fine_mass_change(i, j, k, component) = 0.0;
                               });
        }

        const auto q = primitive.array();
        amrex::ParallelFor(box_grown_5,
                           [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                           {
                               eb_ctoprim(i, j, k, state, q, eb.flags, parameters);
                           });

#ifdef HYPEREB_ENABLE_RUNTIME_CHECKS
        require_finite(primitive, box_grown_5, NPRIM, "conservative-to-primitive");
#endif

        if (physics.do_diffusion)
        {
            const auto coefficients = diffusion_coefficient.array();
            if (physics.use_constant_viscosity)
            {
                amrex::ParallelFor(box_grown_3,
                                   [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                                   {
                                       hypereb_constcoef_eb(i, j, k, coefficients, parameters);
                                   });
            }
            else
            {
                amrex::ParallelFor(box_grown_3,
                                   [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                                   {
                                       hypereb_diffcoef_eb(
                                           i, j, k,
                                           state, q,
                                           eb.flags,
                                           eb.volume_fraction,
                                           coefficients,
                                           parameters);
                                   });
            }
        }

        // Characteristic MUSCL is fixed; EB kernels handle geometric stencils.
        const amrex::GpuArray<amrex::Real, 3>
            transverse_weights{0.0, 1.0, 0.5};

        const amrex::Box x_flux_box =
            detail::eb_face_flux_box(box, 0);

        const amrex::Box y_flux_box =
            detail::eb_face_flux_box(box, 1);

        const amrex::Box z_flux_box =
            detail::eb_face_flux_box(box, 2);

        const auto x_flux =
            face_flux[0].array();

        const auto y_flux =
            face_flux[1].array();

        const auto z_flux =
            face_flux[2].array();

        amrex::Gpu::DeviceScalar<int> fatal_flag(0);
        amrex::Gpu::DeviceScalar<int> fatal_direction(-1);
        amrex::Gpu::DeviceScalar<int> fatal_i(0);
        amrex::Gpu::DeviceScalar<int> fatal_j(0);
        amrex::Gpu::DeviceScalar<int> fatal_k(0);
        int *const fatal_ptr =
            fatal_flag.dataPtr();
        int *const fatal_direction_ptr = fatal_direction.dataPtr();
        int *const fatal_i_ptr = fatal_i.dataPtr();
        int *const fatal_j_ptr = fatal_j.dataPtr();
        int *const fatal_k_ptr = fatal_k.dataPtr();

        // ------------------------------------------------------
        // X-direction hyperbolic flux
        // ------------------------------------------------------
        amrex::ParallelFor(
            x_flux_box,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
            {
                const int status = hypereb_eb_muscl_flux_x(
                    i, j, k, x_flux, state, q, eb.flags, parameters);

                if (status == hypereb_reconstruction_fatal)
                {
                    if (amrex::HostDevice::Atomic::FetchAdd(
                            fatal_ptr, 1) == 0)
                    {
                        *fatal_direction_ptr = 0;
                        *fatal_i_ptr = i;
                        *fatal_j_ptr = j;
                        *fatal_k_ptr = k;
                    }
                }
            });

        // ------------------------------------------------------
        // Y-direction hyperbolic flux
        // ------------------------------------------------------
        amrex::ParallelFor(
            y_flux_box,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
            {
                const int status = hypereb_eb_muscl_flux_y(
                    i, j, k, y_flux, state, q, eb.flags, parameters);

                if (status == hypereb_reconstruction_fatal)
                {
                    if (amrex::HostDevice::Atomic::FetchAdd(
                            fatal_ptr, 1) == 0)
                    {
                        *fatal_direction_ptr = 1;
                        *fatal_i_ptr = i;
                        *fatal_j_ptr = j;
                        *fatal_k_ptr = k;
                    }
                }
            });

        // ------------------------------------------------------
        // Z-direction hyperbolic flux
        // ------------------------------------------------------
        amrex::ParallelFor(
            z_flux_box,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
            {
                const int status = hypereb_eb_muscl_flux_z(
                    i, j, k, z_flux, state, q, eb.flags, parameters);

                if (status == hypereb_reconstruction_fatal)
                {
                    if (amrex::HostDevice::Atomic::FetchAdd(
                            fatal_ptr, 1) == 0)
                    {
                        *fatal_direction_ptr = 2;
                        *fatal_i_ptr = i;
                        *fatal_j_ptr = j;
                        *fatal_k_ptr = k;
                    }
                }
            });

#ifdef AMREX_USE_GPU
        amrex::Gpu::streamSynchronize();
#endif

        if (fatal_flag.dataValue() != 0)
        {
            // [direction:i:j:k], where direction is x=0, y=1, z=2.
            amrex::Print() << '['
                           << fatal_direction.dataValue() << ':'
                           << fatal_i.dataValue() << ':'
                           << fatal_j.dataValue() << ':'
                           << fatal_k.dataValue() << ']' << '\n';
            amrex::Abort(
                "Fatal state detected during EB hyperbolic "
                "reconstruction. The cell-average state is invalid.");
        }

        if (physics.do_diffusion)
        {
            const auto coefficients =
                diffusion_coefficient.const_array();

            amrex::ParallelFor(
                x_flux_box,
                [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                {
                    hypereb_diff_eb_x(
                        i, j, k,
                        q,
                        coefficients,
                        eb.flags,
                        inverse_spacing,
                        transverse_weights,
                        x_flux);
                });

            amrex::ParallelFor(
                y_flux_box,
                [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                {
                    hypereb_diff_eb_y(
                        i, j, k,
                        q,
                        coefficients,
                        eb.flags,
                        inverse_spacing,
                        transverse_weights,
                        y_flux);
                });

            amrex::ParallelFor(
                z_flux_box,
                [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                {
                    hypereb_diff_eb_z(
                        i, j, k,
                        q,
                        coefficients,
                        eb.flags,
                        inverse_spacing,
                        transverse_weights,
                        z_flux);
                });
        }

#ifdef HYPEREB_ENABLE_RUNTIME_CHECKS
        require_finite(face_flux[0], x_flux_box, NEQNS, "x-face flux");
        require_finite(face_flux[1], y_flux_box, NEQNS, "y-face flux");
        require_finite(face_flux[2], z_flux_box, NEQNS, "z-face flux");
#endif

        const auto divergence = conservative_divergence.array();
        const auto coefficients = diffusion_coefficient.const_array();
        const auto box_low = box.smallEnd();
        const auto box_high = box.bigEnd();

        amrex::ParallelFor(
            box_grown_2, NEQNS,
            [=] AMREX_GPU_DEVICE(int i, int j, int k, int component) noexcept
            {
                eb_compute_div(
                    i, j, k, component,
                    box_low, box_high,
                    q,
                    divergence,
                    x_flux, y_flux, z_flux,
                    output_flux[0],
                    output_flux[1],
                    output_flux[2],
                    eb.flags,
                    eb.volume_fraction,
                    eb.cell_centroid,
                    eb.boundary_centroid,
                    coefficients,
                    eb.area_fraction[0],
                    eb.area_fraction[1],
                    eb.area_fraction[2],
                    eb.face_centroid[0],
                    eb.face_centroid[1],
                    eb.face_centroid[2],
                    inverse_spacing,
                    parameters,
                    physics.do_diffusion && physics.eb_wall_diffusion);
            });

        amrex::ParallelFor(
            box_grown_2, NCONS - NEQNS,
            [=] AMREX_GPU_DEVICE(int i, int j, int k, int component) noexcept
            {
                divergence(
                    i, j, k,
                    NEQNS + component) = 0.0;
            });

#ifdef HYPEREB_ENABLE_RUNTIME_CHECKS
        require_finite(
            conservative_divergence,
            box_grown_2,
            NEQNS,
            "cut-cell divergence");
#endif

        // Native WSR uses the stage dt; reflux has an independent RK weight.
        amrex::FArrayBox scratch(box_grown_3, NCONS);
        amrex::ApplyMLRedistribution(
            box,
            NCONS,
            derivative,
            divergence,
            state,
            scratch.array(),
            eb.flags,
            AMREX_D_DECL(
                eb.area_fraction[0],
                eb.area_fraction[1],
                eb.area_fraction[2]),
            eb.volume_fraction,
            AMREX_D_DECL(
                eb.face_centroid[0],
                eb.face_centroid[1],
                eb.face_centroid[2]),
            eb.cell_centroid,
            boundary_conditions,
            geometry,
            stage_dt,
            "StateRedist",
            coupling.as_coarse,
            coupling.coarse_mass_change,
            coupling.coarse_flag,
            coupling.as_fine,
            coupling.fine_mass_change,
            coupling.level_mask,
            coupling.level_mask_not_covered,
            reflux_weight,
            false,
            0,
            0, // Frozen production WSR order, independent of MUSCL order.
            physics.srd_target_volfrac);

        report_stage_state_probe(
            box, state, conservative_divergence.const_array(), derivative,
            eb.volume_fraction, physics, stage_dt, time);

        require_stabilized_candidate_admissible(
            box, state, conservative_divergence.const_array(), derivative,
            eb, geometry, physics, stage_dt, time);

#ifdef AMREX_USE_GPU
        amrex::Gpu::streamSynchronize();
#endif
    }

    void compute_eb_surface_loads(
        const amrex::Box &box,
        const amrex::Array4<amrex::Real> &loads,
        const amrex::Array4<const amrex::Real> &state,
        const amrex::Geometry &geometry,
        const RegularPhysics &physics,
        const EBGeometryViews &eb,
        amrex::Real time)
    {
        const amrex::Box box_grown_3 = amrex::grow(box, 3);
        const amrex::Box box_grown_5 = amrex::grow(box, 5);
        Parm parameters = make_eb_parameters(geometry, physics, time);
        const auto dx = geometry.CellSizeArray();

        amrex::FArrayBox primitive(box_grown_5, NPRIM);
        const auto q = primitive.array();
        amrex::ParallelFor(box_grown_5,
                           [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                           {
                               eb_ctoprim(i, j, k, state, q, eb.flags, parameters);
                           });

        amrex::FArrayBox diffusion_coefficient;
        amrex::Array4<const amrex::Real> coefficients;
        if (physics.do_diffusion && physics.eb_wall_diffusion)
        {
            diffusion_coefficient.resize(box_grown_3, NCOEF);
            const auto coefficient_output = diffusion_coefficient.array();
            if (physics.use_constant_viscosity)
            {
                amrex::ParallelFor(box_grown_3,
                                   [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                                   {
                                       hypereb_constcoef_eb(i, j, k, coefficient_output, parameters);
                                   });
            }
            else
            {
                amrex::ParallelFor(box_grown_3,
                                   [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                                   {
                                       hypereb_diffcoef_eb(i, j, k, state, q, eb.flags,
                                                       eb.volume_fraction, coefficient_output, parameters);
                                   });
            }
            coefficients = diffusion_coefficient.const_array();
        }

        const bool include_viscous_load = physics.do_diffusion && physics.eb_wall_diffusion;
        amrex::ParallelFor(box,
                           [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                           {
                               for (int component = 0; component < NumSurfaceLoadComponents; ++component)
                               {
                                   loads(i, j, k, component) = 0.0;
                               }
                               if (!eb.flags(i, j, k).isSingleValued())
                               {
                                   return;
                               }

                               const amrex::Real area_x = eb.area_fraction[0](i, j, k) - eb.area_fraction[0](i + 1, j, k);
                               const amrex::Real area_y = eb.area_fraction[1](i, j, k) - eb.area_fraction[1](i, j + 1, k);
                               const amrex::Real area_z = eb.area_fraction[2](i, j, k) - eb.area_fraction[2](i, j, k + 1);
                               const amrex::Real physical_area_x = area_x * dx[1] * dx[2];
                               const amrex::Real physical_area_y = area_y * dx[0] * dx[2];
                               const amrex::Real physical_area_z = area_z * dx[0] * dx[1];

                               loads(i, j, k, BoundaryArea) = std::sqrt(
                                   physical_area_x * physical_area_x + physical_area_y * physical_area_y + physical_area_z * physical_area_z);
                               HyperEBWallPrimitiveState wall_state{
                                   q(i,j,k,QRHO),
                                   {q(i,j,k,QU),q(i,j,k,QV),q(i,j,k,QW)},
                                   q(i,j,k,QPRES)};
                               reconstruct_connected_hyp_wall_state(
                                   i,j,k,q,eb.volume_fraction,eb.cell_centroid,
                                   eb.boundary_centroid,eb.flags,parameters,
                                   wall_state);
                               amrex::GpuArray<amrex::Real,NEQNS> numerical_wall_flux;
                               amrex::Real numerical_wall_pressure=wall_state.pressure;
                               compute_hyp_wallflux(
                                   i,j,k,wall_state.density,
                                   wall_state.velocity[0],wall_state.velocity[1],
                                   wall_state.velocity[2],wall_state.pressure,
                                   eb.area_fraction[0](i,j,k),
                                   eb.area_fraction[0](i+1,j,k),
                                   eb.area_fraction[1](i,j,k),
                                   eb.area_fraction[1](i,j+1,k),
                                   eb.area_fraction[2](i,j,k),
                                   eb.area_fraction[2](i,j,k+1),
                                   eb.boundary_centroid,numerical_wall_flux,
                                   parameters,&numerical_wall_pressure);
                               // The finite-volume update needs the complete
                               // moving-boundary momentum flux, including the
                               // swept convective term.  The mechanical load
                               // on the body is only the Riemann pressure
                               // traction p* n dA.
                               loads(i,j,k,SurfacePressure)=
                                   numerical_wall_pressure;
                               loads(i,j,k,PressureForceX)=
                                   numerical_wall_pressure*physical_area_x;
                               loads(i,j,k,PressureForceY)=
                                   numerical_wall_pressure*physical_area_y;
                               loads(i,j,k,PressureForceZ)=
                                   numerical_wall_pressure*physical_area_z;
                               loads(i, j, k, SurfaceTemperature) = q(i, j, k, QTEMP);
                               loads(i, j, k, WallTemperature) =
                                   parameters.eb_thermal_boundary == Parm::thermal_isothermal
                                       ? parameters.eb_wall_temperature
                                       : q(i, j, k, QTEMP);
                               if (loads(i, j, k, BoundaryArea) > 0.0)
                               {
                                   const amrex::Real inverse_area = 1.0 / loads(i, j, k, BoundaryArea);
                                   loads(i, j, k, SurfaceNormalX) = physical_area_x * inverse_area;
                                   loads(i, j, k, SurfaceNormalY) = physical_area_y * inverse_area;
                                   loads(i, j, k, SurfaceNormalZ) = physical_area_z * inverse_area;
                               }

                               if (include_viscous_load
                                   && eb_wall_is_no_slip_segment(
                                       i, j, k, eb.boundary_centroid, parameters))
                               {
                                   amrex::GpuArray<amrex::Real, NEQNS> viscous_wall_flux;
                                   compute_diff_wallflux(
                                       i, j, k, q, coefficients,
                                       eb.boundary_centroid,
                                       eb.area_fraction[0](i, j, k), eb.area_fraction[0](i + 1, j, k),
                                       eb.area_fraction[1](i, j, k), eb.area_fraction[1](i, j + 1, k),
                                       eb.area_fraction[2](i, j, k), eb.area_fraction[2](i, j, k + 1),
                                       viscous_wall_flux, parameters);

                                   // The wall kernel differentiates in cell-index space.
                                   // Its momentum flux therefore becomes a physical 3-D force after
                                   // multiplication by one cell width (dx == dy == dz is enforced).
                                   loads(i, j, k, ViscousForceX) = viscous_wall_flux[UMX] * dx[0];
                                   loads(i, j, k, ViscousForceY) = viscous_wall_flux[UMY] * dx[0];
                                   loads(i, j, k, ViscousForceZ) = viscous_wall_flux[UMZ] * dx[0];

                                   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> wall_velocity;
                                   hypereb_wall_velocity(
                                       i, j, k, eb.boundary_centroid, parameters, wall_velocity);
                                   const amrex::Real wall_work =
                                       wall_velocity[0] * viscous_wall_flux[UMX] + wall_velocity[1] * viscous_wall_flux[UMY] + wall_velocity[2] * viscous_wall_flux[UMZ];
                                   // Remove moving-wall mechanical work from the total EB energy
                                   // flux. The remainder is conductive heat transfer into the wall.
                                   loads(i, j, k, HeatTransfer) =
                                       (viscous_wall_flux[UEDEN] - wall_work) * dx[0];
                                   if (loads(i, j, k, BoundaryArea) > 0.0)
                                   {
                                       loads(i, j, k, WallHeatFlux) = loads(i, j, k, HeatTransfer) / loads(i, j, k, BoundaryArea);
                                   }
                               }
                           });
    }

} // namespace hypereb::kernels
