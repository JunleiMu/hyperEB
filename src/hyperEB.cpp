// Source attribution and license obligations are recorded in THIRD_PARTY_NOTICES.md.

#include <hyperEB.H>
#include <hyperEB_EmbeddedBoundary.H>
#include <hyperEB_RegularGrid.H>
#include <hyperEB_Refinement.H>
#include <hyperEB_F.H>
#include <eb_kernels/hyperEB_index_macros.H>

#include <AMReX_EBMultiFabUtil.H>
#include <AMReX_ParmParse.H>
#include <AMReX_EBAmrUtil.H>
#include <AMReX_EBFArrayBox.H>
#include <AMReX_GpuAtomic.H>
#include <AMReX_GpuContainers.H>

#include <algorithm>
#include <climits>
#include <cmath>
#include <memory>
#include <limits>
#include <sstream>
#include <string>

using namespace amrex;

bool eb2_geometry_is_dynamic();

constexpr int HyperEB::level_mask_interior;
constexpr int HyperEB::level_mask_covered;
constexpr int HyperEB::level_mask_notcovered;
constexpr int HyperEB::level_mask_physbnd;

constexpr int HyperEB::NUM_GROW;

BCRec HyperEB::phys_bc;

int HyperEB::verbose = 0;
IntVect HyperEB::hydro_tile_size{AMREX_D_DECL(1024, 16, 16)};
Real HyperEB::cfl = 0.3;
bool HyperEB::moving_wall_cfl = false;
bool HyperEB::moving_wall_cfl_diagnostics = false;
int HyperEB::do_reflux = 1;
int HyperEB::refine_cutcells = 1;
int HyperEB::residual_int = 0;
Real HyperEB::residual_reference_floor = 1.0e-30;
std::string HyperEB::residual_file = "residual_history.csv";
bool HyperEB::residual_file_initialized = false;
int HyperEB::force_int = 0;
std::string HyperEB::force_file = "aero_history.csv";
bool HyperEB::force_file_initialized = false;
std::array<Real, 3> HyperEB::force_reference_point{0.0, 0.0, 0.0};
std::array<Real, 3> HyperEB::force_drag_direction{1.0, 0.0, 0.0};
std::array<Real, 3> HyperEB::force_lift_direction{0.0, 1.0, 0.0};
std::array<Real, 3> HyperEB::force_side_direction{0.0, 0.0, 1.0};
Real HyperEB::force_reference_density = 0.0;
Real HyperEB::force_reference_velocity = 0.0;
Real HyperEB::force_reference_area = 0.0;
Real HyperEB::force_reference_length = 1.0;
Real HyperEB::force_pressure_offset = 0.0;
Real HyperEB::force_reference_pressure = 0.0;
int HyperEB::surface_int = 0;
std::string HyperEB::surface_prefix = "surface";
int HyperEB::refine_max_pressure_jump_lev = -1;
Real HyperEB::refine_pressure_jump = 0.1;
Real HyperEB::refine_pressure_scale_floor = 0.0;
bool HyperEB::refine_pressure_window_enabled = false;
std::array<Real, 3> HyperEB::refine_pressure_window_center{0.0, 0.0, 0.0};
std::array<Real, 3> HyperEB::refine_pressure_window_velocity{0.0, 0.0, 0.0};
std::array<Real, 3> HyperEB::refine_pressure_window_lo{0.0, 0.0, 0.0};
std::array<Real, 3> HyperEB::refine_pressure_window_hi{0.0, 0.0, 0.0};
bool HyperEB::refine_sensor_verbose = false;
Vector<RealBox> HyperEB::refine_boxes;

HyperEB::HyperEB()
{
}

HyperEB::HyperEB(Amr &papa,
         int lev,
         const Geometry &level_geom,
         const BoxArray &bl,
         const DistributionMapping &dm,
         Real time)
    : AmrLevel(papa, lev, level_geom, bl, dm, time)
{
    if (do_reflux && level > 0)
    {
        flux_reg.define(bl, papa.boxArray(level - 1),
                        dm, papa.DistributionMap(level - 1),
                        level_geom, papa.Geom(level - 1),
                        papa.refRatio(level - 1), level, NUM_STATE);
    }

    buildMetrics();
}

HyperEB::~HyperEB()
{
}

void HyperEB::init(AmrLevel &old)
{
    auto &oldlev = dynamic_cast<HyperEB &>(old);

    Real dt_new = parent->dtLevel(level);
    Real cur_time = oldlev.state[State_Type].curTime();
    Real prev_time = oldlev.state[State_Type].prevTime();
    Real dt_old = cur_time - prev_time;
    setTimeLevel(cur_time, dt_old, dt_new);

    MultiFab &S_new = get_new_data(State_Type);
    FillPatch(old, S_new, 0, cur_time, State_Type, 0, NUM_STATE);

    MultiFab &C_new = get_new_data(Cost_Type);
    FillPatch(old, C_new, 0, cur_time, Cost_Type, 0, 1);

}

void HyperEB::init()
{
    Real dt = parent->dtLevel(level);
    Real cur_time = getLevel(level - 1).state[State_Type].curTime();
    Real prev_time = getLevel(level - 1).state[State_Type].prevTime();
    Real dt_old = (cur_time - prev_time) / static_cast<Real>(parent->MaxRefRatio(level - 1));
    setTimeLevel(cur_time, dt_old, dt);

    MultiFab &S_new = get_new_data(State_Type);
    FillCoarsePatch(S_new, 0, cur_time, State_Type, 0, NUM_STATE);

    MultiFab &C_new = get_new_data(Cost_Type);
    FillCoarsePatch(C_new, 0, cur_time, Cost_Type, 0, 1);

}

void HyperEB::initData()
{
    BL_PROFILE("HyperEB::initData()");

    const Real *dx = geom.CellSize();
    const Real *prob_lo = geom.ProbLo();
    MultiFab &S_new = get_new_data(State_Type);
    Real cur_time = state[State_Type].curTime();

#ifdef _OPENMP
#pragma omp parallel
#endif
    for (MFIter mfi(S_new); mfi.isValid(); ++mfi)
    {
        const Box &box = mfi.validbox();
        hypereb_initdata(&level, &cur_time,
                     BL_TO_FORTRAN_BOX(box),
                     BL_TO_FORTRAN_ANYD(S_new[mfi]),
                     dx, prob_lo);
    }

    MultiFab &C_new = get_new_data(Cost_Type);
    C_new.setVal(1.0);

}

void HyperEB::computeInitialDt(int finest_level,
                           int sub_cycle,
                           Vector<int> &n_cycle,
                           const Vector<IntVect> &ref_ratio,
                           Vector<Real> &dt_level,
                           Real stop_time)
{
    //
    // Grids have been constructed, compute dt for all levels.
    //
    if (level > 0)
    {
        return;
    }

    Real dt_0 = std::numeric_limits<Real>::max();
    int n_factor = 1;
    for (int i = 0; i <= finest_level; i++)
    {
        dt_level[i] = getLevel(i).initialTimeStep();
        n_factor *= n_cycle[i];
        dt_0 = std::min(dt_0, n_factor * dt_level[i]);
    }

    if (!std::isfinite(dt_0) || dt_0 <= 0.0)
    {
        std::ostringstream message;
        message << "HyperEB::computeInitialDt produced an invalid coarse time step: dt="
                << dt_0 << ", level candidates=(";
        for (int i = 0; i <= finest_level; ++i)
        {
            message << (i == 0 ? "" : ",") << dt_level[i];
        }
        message << ")";
        amrex::Abort(message.str());
    }

    //
    // Limit dt's by the value of stop_time.
    //
    const Real eps = 0.001 * dt_0;
    Real cur_time = state[State_Type].curTime();
    if (stop_time >= 0.0)
    {
        if ((cur_time + dt_0) > (stop_time - eps))
            dt_0 = stop_time - cur_time;
    }

    n_factor = 1;
    for (int i = 0; i <= finest_level; i++)
    {
        n_factor *= n_cycle[i];
        dt_level[i] = dt_0 / n_factor;
    }
}

void HyperEB::computeNewDt(int finest_level,
                       int sub_cycle,
                       Vector<int> &n_cycle,
                       const Vector<IntVect> &ref_ratio,
                       Vector<Real> &dt_min,
                       Vector<Real> &dt_level,
                       Real stop_time,
                       int post_regrid_flag)
{
    //
    // We are at the end of a coarse grid timecycle.
    // Compute the timesteps for the next iteration.
    //
    if (level > 0)
    {
        return;
    }

    for (int i = 0; i <= finest_level; i++)
    {
        dt_min[i] = getLevel(i).estTimeStep();
    }

    if (post_regrid_flag == 1)
    {
        //
        // Limit dt's by pre-regrid dt
        //
        for (int i = 0; i <= finest_level; i++)
        {
            dt_min[i] = std::min(dt_min[i], dt_level[i]);
        }
    }
    else
    {
        //
        // Limit dt's by change_max * old dt
        //
        static Real change_max = 1.1;
        for (int i = 0; i <= finest_level; i++)
        {
            dt_min[i] = std::min(dt_min[i], change_max * dt_level[i]);
        }
    }

    //
    // Find the minimum over all levels
    //
    Real dt_0 = std::numeric_limits<Real>::max();
    int n_factor = 1;
    for (int i = 0; i <= finest_level; i++)
    {
        n_factor *= n_cycle[i];
        dt_0 = std::min(dt_0, n_factor * dt_min[i]);
    }

    if (!std::isfinite(dt_0) || dt_0 <= 0.0)
    {
        std::ostringstream message;
        message << "HyperEB::computeNewDt produced an invalid coarse time step: dt="
                << dt_0 << ", level candidates=(";
        for (int i = 0; i <= finest_level; ++i)
        {
            message << (i == 0 ? "" : ",") << dt_min[i];
        }
        message << "). Check the first invalid fluid-state diagnostic.";
        amrex::Abort(message.str());
    }

    //
    // Limit dt's by the value of stop_time.
    //
    const Real eps = 0.001 * dt_0;
    Real cur_time = state[State_Type].curTime();
    if (stop_time >= 0.0)
    {
        if ((cur_time + dt_0) > (stop_time - eps))
        {
            dt_0 = stop_time - cur_time;
        }
    }

    n_factor = 1;
    for (int i = 0; i <= finest_level; i++)
    {
        n_factor *= n_cycle[i];
        dt_level[i] = dt_0 / n_factor;
    }
}

void HyperEB::post_regrid(int lbase, int new_finest)
{
    amrex::ignore_unused(lbase, new_finest);

    // State_Type::volfraction is deliberately the PREVIOUS-geometry field.
    // HyperEB::init(old) transfers it with the other state components when a
    // dynamic-EB regrid rebuilds the level factory.  Replacing it here by the
    // current factory's volume fraction erases exactly the old/new topology
    // change needed to identify freshly uncovered cells.  Initial data and
    // newly created levels already receive a valid history through initData
    // or FillCoarsePatch, so no post-regrid overwrite is required.
}

void HyperEB::post_timestep(int iteration)
{

    if (do_reflux && level < parent->finestLevel())
    {
        HyperEB &fine_level = getLevel(level + 1);
        MultiFab &S_crse = get_new_data(State_Type);
        MultiFab &S_fine = fine_level.get_new_data(State_Type);
        fine_level.flux_reg.Reflux(S_crse, *volfrac, S_fine, *fine_level.volfrac);
        // EB re-redistribution can modify fine valid cells too. Detect a
        // failure here rather than attributing it to next step's MUSCL.
        fine_level.checkAdmissibility(
            S_fine, "after parent EB reflux (fine side)",
            fine_level.state[State_Type].curTime(), true);
        checkAdmissibility(
            S_crse, "after reflux", state[State_Type].curTime());
    }

    if (level < parent->finestLevel())
    {
        avgDown();
        checkAdmissibility(get_new_data(State_Type), "after average-down",
                           state[State_Type].curTime());
        // RK stage fill data on the fine level depend on this coarse state.
        // Recreate the FillPatcher after refluxing and averaging down.
        getLevel(level + 1).resetFillPatcher();
    }
}

void HyperEB::postCoarseTimeStep(Real time)
{
    if (level == 0 && residual_int > 0 &&
        parent->levelSteps(0) % residual_int == 0)
    {
        writeResidualHistory(time);
    }
    if (level == 0 && force_int > 0 &&
        parent->levelSteps(0) % force_int == 0)
    {
        writePressureForceHistory(time);
    }
    if (level == 0 && surface_int > 0 &&
        parent->levelSteps(0) % surface_int == 0)
    {
        writeEBSurfaceData(time);
    }

    // This only computes sum on level 0
    if (verbose >= 2)
    {
        printTotal();
    }
}

void HyperEB::writeFinalDiagnostics(Real time)
{
    if (level != 0)
    {
        return;
    }

    const int step = parent->levelSteps(0);
    if (residual_int > 0 && step % residual_int != 0)
    {
        writeResidualHistory(time);
    }
    if (force_int > 0 && step % force_int != 0)
    {
        writePressureForceHistory(time);
    }
    if (surface_int > 0 && step % surface_int != 0)
    {
        writeEBSurfaceData(time);
    }
}

void HyperEB::printTotal() const
{
    const MultiFab &S_new = get_new_data(State_Type);
    MultiFab mf(grids, dmap, 1, 0);
    std::array<Real, 5> tot;
    for (int comp = 0; comp < 5; ++comp)
    {
        MultiFab::Copy(mf, S_new, comp, 0, 1, 0);
        MultiFab::Multiply(mf, *volfrac, 0, 0, 1, 0);
        tot[comp] = mf.sum(0, true) * geom.ProbSize();
    }
#ifdef BL_LAZY
    Lazy::QueueReduction([=]() mutable
                         {
#endif
                             ParallelDescriptor::ReduceRealSum(tot.data(), 5, ParallelDescriptor::IOProcessorNumber());
                             amrex::Print().SetPrecision(17) << "\n[HyperEB] Total mass       is " << tot[0] << "\n"
                                                             << "      Total x-momentum is " << tot[1] << "\n"
                                                             << "      Total y-momentum is " << tot[2] << "\n"
                                                             << "      Total z-momentum is " << tot[3] << "\n"
                                                             << "      Total energy     is " << tot[4] << "\n";
#ifdef BL_LAZY
                         });
#endif
}

void HyperEB::post_init(Real)
{
    if (level > 0)
        return;
    for (int k = parent->finestLevel() - 1; k >= 0; --k)
    {
        getLevel(k).avgDown();
    }

    if (verbose >= 2)
    {
        printTotal();
    }

    initializeResidualHistory(false);
    initializePressureForceHistory(false);
}

void HyperEB::post_restart()
{
    if (level == 0)
    {
        initializeResidualHistory(true);
        initializePressureForceHistory(true);
    }
}

void HyperEB::errorEst(TagBoxArray &tags, int, int, Real time, int, int)
{
    BL_PROFILE("HyperEB::errorEst()");

    if (refine_cutcells)
    {
        const MultiFab &S_new = get_new_data(State_Type);
        const auto &factory =
            dynamic_cast<EBFArrayBoxFactory const &>(S_new.Factory());
        const auto &flags = factory.getMultiEBCellFlagFab();
        const char tag_value = TagBox::SET;

#ifdef _OPENMP
#pragma omp parallel
#endif
        for (MFIter mfi(S_new, true); mfi.isValid(); ++mfi)
        {
            const Box &bx = mfi.tilebox();
            const auto &flag_fab = flags[mfi];
            const FabType type = flag_fab.getType(bx);
            if (type != FabType::regular && type != FabType::covered)
            {
                auto &tag_fab = tags[mfi];
                const auto flag = flag_fab.const_array();
                for (BoxIterator iterator(bx); iterator.ok(); ++iterator)
                {
                    const IntVect &cell = iterator();
                    if (flag(cell).isSingleValued())
                    {
                        tag_fab(cell) = tag_value;
                    }
                }
            }
        }
    }

    if (!refine_boxes.empty())
    {
        const Real *problo = geom.ProbLo();
        const Real *dx = geom.CellSize();

#ifdef _OPENMP
#pragma omp parallel
#endif
        for (MFIter mfi(tags); mfi.isValid(); ++mfi)
        {
            auto &fab = tags[mfi];
            const Box &bx = fab.box();
            for (BoxIterator bi(bx); bi.ok(); ++bi)
            {
                const IntVect &cell = bi();
                RealVect pos{AMREX_D_DECL((cell[0] + 0.5) * dx[0] + problo[0],
                                          (cell[1] + 0.5) * dx[1] + problo[1],
                                          (cell[2] + 0.5) * dx[2] + problo[2])};
                for (const auto &rbx : refine_boxes)
                {
                    if (rbx.contains(pos))
                    {
                        fab(cell) = TagBox::SET;
                    }
                }
            }
        }
    }

    if (level < refine_max_pressure_jump_lev)
    {
        auto pressure = derive("pressure", time, 1);
        const auto& state = get_new_data(State_Type);
        const auto& factory = dynamic_cast<const EBFArrayBoxFactory&>(state.Factory());
        const auto& flags = factory.getMultiEBCellFlagFab();
        const Box available_domain = geom.growPeriodicDomain(1);
        const auto prob_lo = geom.ProbLoArray();
        const auto cell_size = geom.CellSizeArray();
        std::array<Real, 3> window_center{};
        for (int direction = 0; direction < 3; ++direction)
        {
            window_center[direction] =
                refine_pressure_window_center[direction] +
                time * refine_pressure_window_velocity[direction];
        }
        Long count = 0;
#ifdef _OPENMP
#pragma omp parallel reduction(+:count)
#endif
        for (MFIter mfi(state, true); mfi.isValid(); ++mfi)
        {
            const auto p = pressure->const_array(mfi);
            const auto f = flags.const_array(mfi);
            const auto tag = tags.array(mfi);
            for (BoxIterator it(mfi.tilebox()); it.ok(); ++it)
            {
                if (refine_pressure_window_enabled)
                {
                    bool inside = true;
                    for (int direction = 0; direction < 3; ++direction)
                    {
                        const Real position = prob_lo[direction] +
                            (static_cast<Real>(it()[direction]) + 0.5) *
                                cell_size[direction];
                        const Real relative =
                            position - window_center[direction];
                        inside = inside &&
                            relative >= refine_pressure_window_lo[direction] &&
                            relative <= refine_pressure_window_hi[direction];
                    }
                    if (!inside)
                    {
                        continue;
                    }
                }
                if (hypereb::refinement::pressure_jump(
                        it(), p, f, available_domain,
                        refine_pressure_scale_floor)
                    >= refine_pressure_jump)
                {
                    tag(it()) = TagBox::SET;
                    ++count;
                }
            }
        }
        if (refine_sensor_verbose)
        {
            ParallelDescriptor::ReduceLongSum(count);
            Print() << "[hyperEB AMR pressure jump] level=" << level
                    << " time=" << time << " threshold=" << refine_pressure_jump
                    << " pressure_scale_floor="
                    << refine_pressure_scale_floor
                    << " moving_window="
                    << static_cast<int>(refine_pressure_window_enabled)
                    << " tagged_cells=" << count << '\n';
        }
    }

}

void HyperEB::read_params()
{
    ParmParse pp("hypereb");

    pp.query("v", verbose);

    Vector<int> tilesize(AMREX_SPACEDIM);
    if (pp.queryarr("hydro_tile_size", tilesize, 0, AMREX_SPACEDIM))
    {
        for (int i = 0; i < AMREX_SPACEDIM; i++)
            hydro_tile_size[i] = tilesize[i];
    }

    pp.query("cfl", cfl);
    pp.query("moving_wall_cfl", moving_wall_cfl);

    Vector<int> lo_bc(AMREX_SPACEDIM), hi_bc(AMREX_SPACEDIM);
    pp.getarr("lo_bc", lo_bc, 0, AMREX_SPACEDIM);
    pp.getarr("hi_bc", hi_bc, 0, AMREX_SPACEDIM);
    for (int i = 0; i < AMREX_SPACEDIM; ++i)
    {
        phys_bc.setLo(i, lo_bc[i]);
        phys_bc.setHi(i, hi_bc[i]);
    }

    pp.query("do_reflux", do_reflux);

    pp.query("refine_cutcells", refine_cutcells);

    ParmParse pp_diagnostics("diagnostics");
    pp_diagnostics.query("moving_wall_cfl", moving_wall_cfl_diagnostics);
    pp_diagnostics.query("residual_int", residual_int);
    pp_diagnostics.query("residual_reference_floor", residual_reference_floor);
    pp_diagnostics.query("residual_file", residual_file);
    pp_diagnostics.query("force_int", force_int);
    pp_diagnostics.query("force_file", force_file);
    pp_diagnostics.query("reference_density", force_reference_density);
    pp_diagnostics.query("reference_velocity", force_reference_velocity);
    pp_diagnostics.query("reference_area", force_reference_area);
    pp_diagnostics.query("reference_length", force_reference_length);
    pp_diagnostics.query("force_pressure_offset", force_pressure_offset);
    pp_diagnostics.query("reference_pressure", force_reference_pressure);
    pp_diagnostics.query("surface_int", surface_int);
    pp_diagnostics.query("surface_prefix", surface_prefix);

    auto read_vector = [&pp_diagnostics](const char *name,
                                         std::array<Real, 3> &values)
    {
        Vector<Real> input;
        if (pp_diagnostics.queryarr(name, input))
        {
            if (input.size() != 3)
            {
                amrex::Abort(std::string("diagnostics.") + name + " must have 3 values");
            }
            std::copy(input.begin(), input.end(), values.begin());
        }
    };
    read_vector("force_reference_point", force_reference_point);
    read_vector("drag_direction", force_drag_direction);
    read_vector("lift_direction", force_lift_direction);
    read_vector("side_direction", force_side_direction);

    auto normalize_direction = [](std::array<Real, 3> &direction,
                                  const char *name)
    {
        const Real magnitude = std::sqrt(
            direction[0] * direction[0] + direction[1] * direction[1] + direction[2] * direction[2]);
        if (!(magnitude > 0.0))
        {
            amrex::Abort(std::string("diagnostics.") + name + " must be nonzero");
        }
        for (Real &value : direction)
        {
            value /= magnitude;
        }
    };
    normalize_direction(force_drag_direction, "drag_direction");
    normalize_direction(force_lift_direction, "lift_direction");
    normalize_direction(force_side_direction, "side_direction");

    if (residual_int < 0)
    {
        amrex::Abort("diagnostics.residual_int must be non-negative");
    }
    if (force_int < 0)
    {
        amrex::Abort("diagnostics.force_int must be non-negative");
    }
    if (surface_int < 0)
    {
        amrex::Abort("diagnostics.surface_int must be non-negative");
    }
    if (!(residual_reference_floor > 0.0))
    {
        amrex::Abort("diagnostics.residual_reference_floor must be positive");
    }
    if (force_reference_density < 0.0 || force_reference_velocity < 0.0 ||
        force_reference_area < 0.0 || !(force_reference_length > 0.0))
    {
        amrex::Abort("diagnostics force reference values must be non-negative and reference_length positive");
    }

    pp.query("refine_max_pressure_jump_lev", refine_max_pressure_jump_lev);
    pp.query("refine_pressure_jump", refine_pressure_jump);
    pp.query("refine_pressure_scale_floor", refine_pressure_scale_floor);
    pp.query("refine_sensor_verbose", refine_sensor_verbose);
    if (!std::isfinite(refine_pressure_jump) ||
        refine_pressure_jump <= 0.0 || refine_pressure_jump >= 1.0)
    {
        amrex::Abort("hypereb.refine_pressure_jump must be finite and in (0,1)");
    }
    if (!std::isfinite(refine_pressure_scale_floor) ||
        refine_pressure_scale_floor < 0.0)
    {
        amrex::Abort(
            "hypereb.refine_pressure_scale_floor must be finite and non-negative");
    }
    Vector<Real> pressure_window_lo;
    refine_pressure_window_enabled = pp.queryarr(
        "refine_pressure_window_lo", pressure_window_lo, 0, 3);
    if (refine_pressure_window_enabled)
    {
        Vector<Real> pressure_window_hi;
        Vector<Real> pressure_window_center;
        Vector<Real> pressure_window_velocity;
        pp.getarr("refine_pressure_window_hi", pressure_window_hi, 0, 3);
        pp.getarr(
            "refine_pressure_window_center", pressure_window_center, 0, 3);
        pp.getarr(
            "refine_pressure_window_velocity", pressure_window_velocity, 0, 3);
        for (int direction = 0; direction < 3; ++direction)
        {
            refine_pressure_window_lo[direction] = pressure_window_lo[direction];
            refine_pressure_window_hi[direction] = pressure_window_hi[direction];
            refine_pressure_window_center[direction] =
                pressure_window_center[direction];
            refine_pressure_window_velocity[direction] =
                pressure_window_velocity[direction];
            if (!std::isfinite(refine_pressure_window_lo[direction]) ||
                !std::isfinite(refine_pressure_window_hi[direction]) ||
                !std::isfinite(refine_pressure_window_center[direction]) ||
                !std::isfinite(refine_pressure_window_velocity[direction]) ||
                !(refine_pressure_window_hi[direction] >
                  refine_pressure_window_lo[direction]))
            {
                amrex::Abort(
                    "cns pressure refinement window values must be finite "
                    "with hi greater than lo");
            }
        }
    }
    refine_boxes.clear();
    int irefbox = 0;
    Vector<Real> refboxlo, refboxhi;
    while (pp.queryarr(("refine_box_lo_" + std::to_string(irefbox)).c_str(), refboxlo))
    {
        pp.getarr(("refine_box_hi_" + std::to_string(irefbox)).c_str(), refboxhi);
        refine_boxes.emplace_back(refboxlo.data(), refboxhi.data());
        ++irefbox;
    }
}

void HyperEB::avgDown()
{
    BL_PROFILE("HyperEB::avgDown()");

    if (level == parent->finestLevel())
        return;

    auto &fine_lev = getLevel(level + 1);

    MultiFab &S_crse = get_new_data(State_Type);
    MultiFab &S_fine = fine_lev.get_new_data(State_Type);

    MultiFab volume(S_fine.boxArray(), S_fine.DistributionMap(), 1, 0);
    volume.setVal(1.0);
    amrex::EB_average_down(S_fine, S_crse, volume, fine_lev.volFrac(),
                           0, S_fine.nComp(), fine_ratio);

    const int nghost = 0;
    computeTemp(S_crse, nghost);
}

void HyperEB::buildMetrics()
{
    BL_PROFILE("HyperEB::buildMetrics()");

    // The wall and redistribution kernels use dx[0] as the common
    // length scale, so every production level must remain isotropic.
    const Real *dx = geom.CellSize();
    if (std::abs(dx[0] - dx[1]) > 1.e-12 * dx[0] || std::abs(dx[0] - dx[2]) > 1.e-12 * dx[0])
    {
        amrex::Abort("HyperEB: must have dx == dy == dz");
    }

    const auto &ebfactory = dynamic_cast<EBFArrayBoxFactory const &>(Factory());

    volfrac = &(ebfactory.getVolFrac());
    bndrycent = &(ebfactory.getBndryCent());
    areafrac = ebfactory.getAreaFrac();
    facecent = ebfactory.getFaceCent();

    if (level == 0)
    {
        std::string geometry_type = "all_regular";
        ParmParse("eb2").query("geom_type", geometry_type);
        if ((geometry_type == "moving_plane" || geometry_type == "stl") &&
            volfrac->min(0) >= 1.0 - 1.e-14)
        {
            amrex::Abort("Moving-plane or STL geometry produced no embedded-boundary cells");
        }
    }

    level_mask.clear();
    // MLStateRedistribute can inspect redistribution neighborhoods beyond the
    // first layer of fine-level ghost cells.  Match AMReX's EB_CNS driver and
    // retain three mask ghost layers for its coarse/fine bookkeeping.
    level_mask.define(grids, dmap, 1, 3);
    level_mask.BuildMask(geom.Domain(), geom.periodicity(),
                         level_mask_covered,
                         level_mask_notcovered,
                         level_mask_physbnd,
                         level_mask_interior);

}


Real HyperEB::estTimeStep()
{
    BL_PROFILE("HyperEB::estTimeStep()");

    Real estdt = std::numeric_limits<Real>::max();

    const Real *dx = geom.CellSize();
    const MultiFab &S = get_new_data(State_Type);

    auto const &fact = dynamic_cast<EBFArrayBoxFactory const &>(S.Factory());
    auto const &flags = fact.getMultiEBCellFlagFab();

#ifdef _OPENMP
#pragma omp parallel reduction(min : estdt)
#endif
    {
        Real dt = std::numeric_limits<Real>::max();
        for (MFIter mfi(S, true); mfi.isValid(); ++mfi)
        {
            const Box &box = mfi.tilebox();

            const auto &sfab = S[mfi];
            const auto &flag = flags[mfi];

            if (flag.getType(box) != FabType::covered)
            {
                hypereb_estdt(BL_TO_FORTRAN_BOX(box),
                          BL_TO_FORTRAN_ANYD(S[mfi]),
                          dx, &dt);
            }
        }
        estdt = std::min(estdt, dt);
    }

    estdt *= cfl;
    ParallelDescriptor::ReduceRealMin(estdt);

    if (moving_wall_cfl)
    {
        const auto wall_estimate = hypereb::kernels::estimate_moving_wall_cfl(
            S, geom, hypereb::kernels::regular_physics(),
            state[State_Type].curTime(), cfl, moving_wall_cfl_diagnostics);
        // Dynamic EB geometry is rebuilt once per coarse step.  AMR's usual
        // time-step aggregation multiplies a level candidate by the number
        // of subcycles leading to that level.  Applying the local wall CFL
        // unchanged would therefore allow one coarse geometry update to move
        // across that many fine cells.  Divide the dynamic-wall candidate by
        // the cumulative subcycle count here; computeInitialDt/computeNewDt
        // multiply it back, leaving the coarse step limited by the finest
        // geometry spacing.  The actual fine step remains dt_coarse/ncycle.
        int cumulative_subcycles = 1;
        for (int lev = 0; lev <= level; ++lev)
        {
            cumulative_subcycles *= parent->nCycle(lev);
        }
        const Real wall_level_candidate = eb2_geometry_is_dynamic()
            ? wall_estimate.normal_dt /
                  static_cast<Real>(cumulative_subcycles)
            : wall_estimate.normal_dt;
        const Real selected_dt = std::min(estdt, wall_level_candidate);
        if (moving_wall_cfl_diagnostics)
        {
            const char *const limiter =
                wall_level_candidate < estdt ? "wall_geometry" : "fluid";
            amrex::Print().SetPrecision(17)
                << "[hyperEB moving-wall CFL] level=" << level
                << " time=" << state[State_Type].curTime()
                << " dt_fluid=" << estdt
                << " dt_wall_normal=" << wall_estimate.normal_dt
                << " subcycles_from_coarse=" << cumulative_subcycles
                << " dt_wall_level_candidate=" << wall_level_candidate
                << " dt_wall_full=" << wall_estimate.full_dt
                << " dt_selected=" << selected_dt
                << " limiter=" << limiter;
            if (wall_estimate.normal_dt !=
                std::numeric_limits<Real>::max())
            {
                amrex::Print()
                    << " cell=" << wall_estimate.limiting_cell
                    << " vfrac=" << wall_estimate.volume_fraction
                    << " a=" << wall_estimate.sound_speed
                    << " u=(" << wall_estimate.fluid_velocity[0] << ','
                    << wall_estimate.fluid_velocity[1] << ','
                    << wall_estimate.fluid_velocity[2] << ')'
                    << " vwall=(" << wall_estimate.wall_velocity[0] << ','
                    << wall_estimate.wall_velocity[1] << ','
                    << wall_estimate.wall_velocity[2] << ')'
                    << " normal=(" << wall_estimate.eb_normal[0] << ','
                    << wall_estimate.eb_normal[1] << ','
                    << wall_estimate.eb_normal[2] << ')'
                    << " vrel_normal="
                    << wall_estimate.normal_relative_speed
                    << " vrel_full=" << wall_estimate.full_relative_speed;
            }
            amrex::Print() << '\n';
        }
        estdt = selected_dt;
    }
    return estdt;
}

Real HyperEB::initialTimeStep()
{
    return estTimeStep();
}

void HyperEB::computeTemp(MultiFab &State, int ng)
{
    BL_PROFILE("HyperEB::computeTemp()");

    auto const &fact = dynamic_cast<EBFArrayBoxFactory const &>(State.Factory());
    auto const &flags = fact.getMultiEBCellFlagFab();

    // This will reset Eint and compute Temperature
#ifdef _OPENMP
#pragma omp parallel
#endif
    for (MFIter mfi(State, true); mfi.isValid(); ++mfi)
    {
        const Box &bx = mfi.growntilebox(ng);

        const auto &sfab = State[mfi];
        const auto &flag = flags[mfi];

        if (flag.getType(bx) != FabType::covered)
        {
            hypereb_compute_temperature(BL_TO_FORTRAN_BOX(bx),
                                    BL_TO_FORTRAN_ANYD(State[mfi]));
        }
    }
}

void HyperEB::checkAdmissibility(const MultiFab &State, const char *stage,
                             Real time, bool composite_only) const
{
    if (!hypereb::kernels::admissibility_diagnostics_enabled())
    {
        return;
    }

    auto const &factory =
        dynamic_cast<EBFArrayBoxFactory const &>(State.Factory());
    auto const &flags = factory.getMultiEBCellFlagFab();
    const auto physics = hypereb::kernels::regular_physics();
    const Real internal_floor =
        physics.smallp / (physics.gamma - Real(1.0));

    amrex::iMultiFab composite_mask;
    const bool use_mask = composite_only && level < parent->finestLevel();
    if (use_mask) {
        composite_mask = amrex::makeFineMask(
            State.boxArray(), State.DistributionMap(), amrex::IntVect(0),
            parent->getLevel(level + 1).boxArray(), parent->refRatio(level),
            geom.periodicity(), 1, 0);
    }

    for (MFIter mfi(State); mfi.isValid(); ++mfi)
    {
        const Box &box = mfi.validbox();
        const auto values = State[mfi].const_array();
        const auto cell_flags = flags[mfi].const_array();
        const auto volume_fraction = (*volfrac)[mfi].const_array();
        const auto active = use_mask ? composite_mask.const_array(mfi)
                                    : amrex::Array4<const int>{};

        Gpu::DeviceScalar<int> failure_count(0);
        Gpu::DeviceScalar<int> failure_i(0);
        Gpu::DeviceScalar<int> failure_j(0);
        Gpu::DeviceScalar<int> failure_k(0);
        Gpu::DeviceScalar<Real> failure_rho(0.0);
        Gpu::DeviceScalar<Real> failure_mx(0.0);
        Gpu::DeviceScalar<Real> failure_my(0.0);
        Gpu::DeviceScalar<Real> failure_mz(0.0);
        Gpu::DeviceScalar<Real> failure_eden(0.0);
        Gpu::DeviceScalar<Real> failure_eint(0.0);
        Gpu::DeviceScalar<Real> failure_aux_eint(0.0);
        Gpu::DeviceScalar<Real> failure_vfrac(0.0);

        int *const count = failure_count.dataPtr();
        int *const bad_i = failure_i.dataPtr();
        int *const bad_j = failure_j.dataPtr();
        int *const bad_k = failure_k.dataPtr();
        Real *const bad_rho = failure_rho.dataPtr();
        Real *const bad_mx = failure_mx.dataPtr();
        Real *const bad_my = failure_my.dataPtr();
        Real *const bad_mz = failure_mz.dataPtr();
        Real *const bad_eden = failure_eden.dataPtr();
        Real *const bad_eint = failure_eint.dataPtr();
        Real *const bad_aux_eint = failure_aux_eint.dataPtr();
        Real *const bad_vfrac = failure_vfrac.dataPtr();

        ParallelFor(
            box,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
            {
                if (cell_flags(i, j, k).isCovered() ||
                    (use_mask && active(i,j,k) == 0))
                {
                    return;
                }

                const Real rho = values(i, j, k, URHO);
                const Real mx = values(i, j, k, UMX);
                const Real my = values(i, j, k, UMY);
                const Real mz = values(i, j, k, UMZ);
                const Real eden = values(i, j, k, UEDEN);
                const Real aux_eint = values(i, j, k, UEINT);
                const Real kinetic = rho > 0.0
                                         ? 0.5 * (mx * mx + my * my + mz * mz) / rho
                                         : 0.0;
                const Real eint = eden - kinetic;
                const bool invalid =
                    !std::isfinite(rho) || !std::isfinite(mx) ||
                    !std::isfinite(my) || !std::isfinite(mz) ||
                    !std::isfinite(eden) || !std::isfinite(eint) ||
                    !(rho >= physics.smallr) || !(eint >= internal_floor);

                if (invalid &&
                    HostDevice::Atomic::FetchAdd(count, 1) == 0)
                {
                    *bad_i = i;
                    *bad_j = j;
                    *bad_k = k;
                    *bad_rho = rho;
                    *bad_mx = mx;
                    *bad_my = my;
                    *bad_mz = mz;
                    *bad_eden = eden;
                    *bad_eint = eint;
                    *bad_aux_eint = aux_eint;
                    *bad_vfrac = volume_fraction(i, j, k);
                }
            });

        Gpu::streamSynchronize();
        if (failure_count.dataValue() != 0)
        {
            std::ostringstream message;
            message << "Invalid conservative state " << stage
                    << ": level=" << level << " time=" << time
                    << " cell=(" << failure_i.dataValue() << ','
                    << failure_j.dataValue() << ',' << failure_k.dataValue()
                    << ") rho=" << failure_rho.dataValue()
                    << " momentum=(" << failure_mx.dataValue() << ','
                    << failure_my.dataValue() << ',' << failure_mz.dataValue()
                    << ") rhoE=" << failure_eden.dataValue()
                    << " rhoe_total=" << failure_eint.dataValue()
                    << " rhoe_aux=" << failure_aux_eint.dataValue()
                    << " volfrac=" << failure_vfrac.dataValue();
            amrex::Abort(message.str());
        }
    }
}
