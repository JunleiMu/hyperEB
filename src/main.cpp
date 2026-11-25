
#include <AMReX.H>
#include <AMReX_BaseFab.H>
#include <AMReX_ParmParse.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Amr.H>
#include <AMReX_EB2.H>

#include <hyperEB.H>

#include <cmath>
#include <initializer_list>
#include <sstream>

using namespace amrex;

amrex::LevelBld* getLevelBld ();
void initialize_EB2 (const Geometry& geom, const int required_level, const int max_level, Real time);
void finalize_EB2();
bool eb2_geometry_is_dynamic();

namespace {

void reject_removed_parameters()
{
    const auto reject = [](const char* prefix,
                           std::initializer_list<const char*> names)
    {
        ParmParse pp(prefix);
        for (const char* name : names)
        {
            if (pp.contains(name))
            {
                amrex::Abort(std::string("Removed hyperEB v1.0 parameter: ") +
                    prefix + "." + name +
                    ". Remove this argument. hyperEB is fixed to 3D, "
                    "characteristic MUSCL/Van Leer, HLL, SSPRK2, native WSR "
                    "(order 0, target 0.5), second-order EB reconstruction, "
                    "connected WLS, and wall-normal quadratic reconstruction.");
            }
        }
    };
    reject("hypereb", {
        "dim", "reconstruction", "reconstruction_basis", "limiter", "plm_theta",
        "inviscid_flux", "time_integrator", "small_cell_method",
        "redistribution_type", "do_reredistribution", "eb_plm_order",
        "use_total_energy_as_eb_weights", "use_mass_as_eb_weights",
        "use_volfrac_as_eb_weights", "srd_max_order", "srd_target_volfrac",
        "wsr_neighborhood_selector", "wsr_neighborhood_selector_verbose",
        "wsr_wave_speed_uncertainty", "wsr_target_cap", "adaptive_wsr",
        "lpfs_positive_hybrid_verbose", "positivity", "gradient_compression",
        "reconstruction_fallback", "stage_fallback", "post_redistribution_fallback",
        "post_redistribution_fallback_max_sweeps",
        "post_redistribution_fallback_verbose", "reflux_fallback",
        "uncovered_cell_method", "geometric_volume_correction",
        "refine_max_dengrad_lev", "refine_dengrad",
        "refine_max_natarajan_lev", "refine_natarajan_tol"});
    reject("physics", {
        "static_wall_riemann", "moving_wall_use_cell_density",
        "moving_wall_material_balance", "eb_flux_interpolation"});
    reject("wall", {"reconstruction", "inviscid_reconstruction"});
    reject("eb2", {
        "wall_translation_x_max", "harmonic_axis", "harmonic_amplitude",
        "harmonic_frequency", "radial_center", "radial_amplitude",
        "radial_frequency", "rotation_center", "rotation_axis",
        "pitch_amplitude_degrees", "pitch_offset_degrees", "pitch_frequency",
        "pitch_phase_degrees", "pitch_start_time", "pitch_ramp_time"});
    for (const auto& name : ParmParse::getEntries("cns"))
    {
        amrex::Abort("Removed legacy parameter namespace: " + name +
                     ". Use the documented hyperEB v1.0 physical inputs.");
    }
}

Real checkpoint_time (const std::string& checkpoint_directory)
{
    const std::string header = checkpoint_directory + "/Header";
    Vector<char> contents;
    ParallelDescriptor::ReadAndBcastFile(header, contents);
    AMREX_ALWAYS_ASSERT(!contents.empty());
    std::istringstream input(std::string(contents.dataPtr()));
    std::string first_line;
    std::getline(input, first_line);
    int space_dimension = -1;
    if (first_line.rfind("CheckPointVersion_", 0) == 0) {
        input >> space_dimension;
    } else {
        std::istringstream(first_line) >> space_dimension;
    }
    if (space_dimension != AMREX_SPACEDIM) { amrex::Abort(); }
    Real time = -1.0;
    input >> time;
    if (!input || !std::isfinite(time) || time < 0.0) { amrex::Abort(); }
    return time;
}

void print_fab_memory(int step, const char* phase)
{
    constexpr amrex::Long bytes_per_mib = 1024L * 1024L;
    const int io_proc = amrex::ParallelDescriptor::IOProcessorNumber();

    amrex::Long current_min = amrex::TotalBytesAllocatedInFabs();
    amrex::Long current_max = current_min;
    amrex::Long current_sum = current_min;
    amrex::Long high_water_min = amrex::TotalBytesAllocatedInFabsHWM();
    amrex::Long high_water_max = high_water_min;

    amrex::ParallelDescriptor::ReduceLongMin(current_min, io_proc);
    amrex::ParallelDescriptor::ReduceLongMax(current_max, io_proc);
    amrex::ParallelDescriptor::ReduceLongSum(current_sum, io_proc);
    amrex::ParallelDescriptor::ReduceLongMin(high_water_min, io_proc);
    amrex::ParallelDescriptor::ReduceLongMax(high_water_max, io_proc);

    amrex::Print() << "[STEP " << step << "] hyperEB FAB memory (" << phase
                   << "): current_per_rank=[" << current_min/bytes_per_mib
                   << " ... " << current_max/bytes_per_mib
                   << "] MiB, current_all_ranks=" << current_sum/bytes_per_mib
                   << " MiB, high_water_per_rank=[" << high_water_min/bytes_per_mib
                   << " ... " << high_water_max/bytes_per_mib << "] MiB\n";
}

} // namespace

int main (int argc, char* argv[])
{

    amrex::Initialize(argc,argv);
    static_assert(AMREX_SPACEDIM == 3, "hyperEB requires three dimensions");
    reject_removed_parameters();

    BL_PROFILE_VAR("main()", pmain);

    Real timer_tot = amrex::second();
    Real timer_init = 0.;
    Real timer_advance = 0.;

    int  max_step;
    int  timing_int;
    int  memory_int;
    Real strt_time;
    Real stop_time;

    {
        ParmParse pp; 

        max_step  = -1;
        timing_int = 1;
        memory_int = 0;
        strt_time =  0.0;
        stop_time = -1.0;

        pp.query("max_step",max_step);
        pp.query("strt_time",strt_time);
        pp.query("stop_time",stop_time);
        ParmParse pp_diagnostics("diagnostics");
        pp_diagnostics.query("timing_int",timing_int);
        pp_diagnostics.query("memory_int",memory_int);
    }

    if (strt_time < 0.0) {
        amrex::Abort("MUST SPECIFY a non-negative strt_time"); 
    }

    if (max_step < 0 && stop_time < 0.0) {
	amrex::Abort("Exiting because neither max_step nor stop_time is non-negative.");
    }
    if (timing_int < 0) {
        amrex::Abort("diagnostics.timing_int must be non-negative");
    }
    if (memory_int < 0) {
        amrex::Abort("diagnostics.memory_int must be non-negative");
    }

    {
        timer_init = amrex::second();

	Amr amr(getLevelBld());
        const bool dynamic_geometry = eb2_geometry_is_dynamic();
        amrex::Print() << "hyperEB EB geometry mode: "
                       << (dynamic_geometry ? "dynamic" : "static") << "\n";
        AmrLevel::SetEBSupportLevel(EBSupport::full);
        // Native state redistribution requires geometry throughout its support.
        AmrLevel::SetEBMaxGrowCells(
            HyperEB::numGrow(), HyperEB::numGrow(), HyperEB::numGrow());

        Real initial_eb_time = strt_time;
        const std::string& restart_file = amr.theRestartFile();
        const bool restarting = !restart_file.empty()
            && restart_file != "init";
        if (dynamic_geometry && restarting) {
            initial_eb_time = checkpoint_time(restart_file);
            amrex::Print() << "hyperEB dynamic EB restart geometry time: "
                           << initial_eb_time << "\n";
        }
        initialize_EB2(amr.Geom(amr.maxLevel()), amr.maxLevel(),
                       amr.maxLevel(), initial_eb_time);

	//std::cout << amr.levelSteps(0) << "\n";
            
	amr.init(strt_time,stop_time);

        if (memory_int > 0) {
            print_fab_memory(amr.levelSteps(0), "after-init");
        }

        timer_init = amrex::second() - timer_init;

        timer_advance = amrex::second();

        bool first_coarse_step = true;

	while ( amr.okToContinue() &&
  	       (amr.levelSteps(0) < max_step || max_step < 0) &&
	       (amr.cumTime() < stop_time || stop_time < 0.0) )
	    
	{
	//GlobalTest();
        const Real eb_build_start = amrex::second();
        const bool reuse_initial_dynamic_geometry =
            dynamic_geometry && first_coarse_step;
        const bool rebuilt_dynamic_geometry = dynamic_geometry
            && !reuse_initial_dynamic_geometry;
        if (rebuilt_dynamic_geometry) {
            initialize_EB2(amr.Geom(amr.maxLevel()), amr.maxLevel(),
                           amr.maxLevel(), amr.cumTime());
        }
        const Real eb_build_time = amrex::second() - eb_build_start;
	    //
	    // Do a coarse timestep.  Recursively calls timeStep()
	    //
        const Real coarse_step_start = amrex::second();
	    amr.coarseTimeStep(stop_time);
        first_coarse_step = false;
        const Real coarse_step_time = amrex::second() - coarse_step_start;
        const Real eb_finalize_start = amrex::second();
        if (rebuilt_dynamic_geometry) {
	        finalize_EB2();
        }
        const Real eb_finalize_time = amrex::second() - eb_finalize_start;

        if (timing_int > 0 && amr.levelSteps(0) % timing_int == 0) {
            amrex::Print() << "[STEP " << amr.levelSteps(0)
                           << "] hyperEB phase times: eb_build=" << eb_build_time
                           << " coarse_step=" << coarse_step_time
                           << " eb_finalize=" << eb_finalize_time << " s\n";
        }
        if (memory_int > 0 && amr.levelSteps(0) % memory_int == 0) {
            print_fab_memory(amr.levelSteps(0), "after-coarse-step");
        }
	}

        timer_advance = amrex::second() - timer_advance;

        // AMReX writes a final plotfile/checkpoint below even when the final
        // step does not hit their configured intervals. Keep hyperEB diagnostics
        // consistent with that behavior without duplicating interval output.
        dynamic_cast<HyperEB&>(amr.getLevel(0)).writeFinalDiagnostics(amr.cumTime());

	// Write final checkpoint and plotfile
	if (amr.stepOfLastCheckPoint() < amr.levelSteps(0)) {
	    amr.checkPoint();
	}
	
	if (amr.stepOfLastPlotFile() < amr.levelSteps(0)) {
	    amr.writePlotFile();
	}
    }

    // Static EB data must outlive Amr and its EB factories.  Dynamic geometry
    // is finalized after each step, so this is a harmless no-op in that mode.
    finalize_EB2();

    timer_tot = amrex::second() - timer_tot;

    ParallelDescriptor::ReduceRealMax({timer_tot, timer_init, timer_advance},
                                      ParallelDescriptor::IOProcessorNumber());

    amrex::Print() << "Run Time total        = " << timer_tot     << "\n"
                   << "Run Time init         = " << timer_init    << "\n"
                   << "Run Time advance      = " << timer_advance << "\n";

    BL_PROFILE_VAR_STOP(pmain);

    amrex::Finalize();

    return 0;
}
