#ifndef HYPEREB_H_
#define HYPEREB_H_

#include <AMReX_AmrLevel.H>
#include <AMReX_EBCellFlag.H>
#include <AMReX_EBFluxRegister.H>

#include <array>
#include <memory>
#include <vector>

namespace hypereb::kernels { struct RegularPhysics; }

    void GlobalTest();

using namespace amrex;
class HyperEB
    :
    public amrex::AmrLevel
{
public:

    HyperEB ();
    HyperEB (amrex::Amr&            papa,
         int                    lev,
         const amrex::Geometry& level_geom,
         const amrex::BoxArray& bl,
         const amrex::DistributionMapping& dm,
         amrex::Real            time);
    virtual ~HyperEB ();

    HyperEB (const HyperEB& rhs) = delete;
    HyperEB& operator= (const HyperEB& rhs) = delete;
    
    // Restart from a checkpoint file.
    virtual void restart (amrex::Amr&     papa,
                          std::istream&   is,
                          bool            bReadSpecial = false) override;

    // Write checkpoint
    virtual void checkPoint(const std::string& dir,
                            std::ostream&      os,
                            amrex::VisMF::How  how = amrex::VisMF::NFiles,
                            bool               dump_old = true) override;

    virtual std::string thePlotFileType () const override {
        return {"HyperCLaw-V1.1"};
    }

    // Write a plotfile to specified directory.
    virtual void writePlotFile (const std::string& dir,
                                std::ostream&      os,
                                amrex::VisMF::How  how) override;

    // Initialize data on this level from another HyperEB (during regrid).
    virtual void init (amrex::AmrLevel& old) override;

    // Initialize data on this level after regridding if old level did not previously exist
    virtual void init () override;
    
    // Initialize grid data at problem start-up.
    virtual void initData () override;

    // Advance grids at this level in time.
    virtual amrex::Real advance (amrex::Real time,
                                 amrex::Real dt,
                                 int  iteration,
                                 int  ncycle) override;

    virtual void computeInitialDt (int                                 finest_level,
                                   int                                 sub_cycle,
                                   amrex::Vector<int>&                  n_cycle,
                                   const amrex::Vector<amrex::IntVect>& ref_ratio,
                                   amrex::Vector<amrex::Real>&          dt_level,
                                   amrex::Real                         stop_time) override;

    virtual void computeNewDt (int                                 finest_level,
                               int                                 sub_cycle,
                               amrex::Vector<int>&                  n_cycle,
                               const amrex::Vector<amrex::IntVect>& ref_ratio,
                               amrex::Vector<amrex::Real>&          dt_min,
                               amrex::Vector<amrex::Real>&          dt_level,
                               amrex::Real                         stop_time,
                               int                                 post_regrid_flag) override;

    virtual void post_regrid (int lbase, int new_finest) override;

    // Do work after timestep().
    virtual void post_timestep (int iteration) override;

    // After a full time step
    virtual void postCoarseTimeStep (amrex::Real time) override;

    // Write enabled diagnostics at the final step when it did not coincide
    // with the configured output interval.
    void writeFinalDiagnostics (amrex::Real time);

    // Do work after init().
    virtual void post_init (amrex::Real stop_time) override;

    virtual void post_restart () override;

    // Error estimation for regridding.
    virtual void errorEst (amrex::TagBoxArray& tb,
                           int                 clearval,
                           int                 tagval,
                           amrex::Real         time,
                           int                 n_error_buf = 0,
                           int                 ngrow = 0) override;

    virtual int WorkEstType () override { return Cost_Type; }

    // Define data descriptors.
    static void variableSetUp ();

    // Cleanup data descriptors at end of run.
    static void variableCleanUp ();

    static int numGrow() { return NUM_GROW; };
   
    void ZeroOutSolidWalls (const int* lev, amrex::MultiFab& S);
    void ComputeUncoveringRedistribute (const amrex::MultiFab& volfracborder, amrex::MultiFab& Sborder, amrex::MultiFab& cons_adjust, amrex::MultiFab& donortag, amrex::Real time, int moving_step);
    void FinalUncoveringRedistribute (const amrex::MultiFab& volfracborder, amrex::MultiFab& Sborder, amrex::MultiFab& cons_adjust, amrex::MultiFab& donortag);

    //MultiFab volfracmfab;

protected:

    static void read_params ();

    const amrex::MultiFab& volFrac () const { return *volfrac; }

    HyperEB& getLevel (int lev) { return dynamic_cast<HyperEB&>(parent->getLevel(lev)); }

    void avgDown ();

    void buildMetrics ();

    amrex::Real estTimeStep ();

    // Compute initial time step.
    amrex::Real initialTimeStep ();

    void computeTemp (amrex::MultiFab& State, int ng);

    void checkAdmissibility (const amrex::MultiFab& State,
                             const char* stage,
                             amrex::Real time,
                             bool composite_only = false) const;

    void compute_dSdt (const amrex::MultiFab& S, amrex::MultiFab& dSdt,
                       amrex::Real stage_dt, amrex::Real reflux_weight,
                       amrex::EBFluxRegister* fr_as_crse, amrex::EBFluxRegister* fr_as_fine,
                       Real Time);

    void printTotal () const;

    void recordResidualRhs (const amrex::MultiFab& rhs);
    void writeResidualHistory (amrex::Real time) const;
    void initializeResidualHistory (bool restart);
    void writePressureForceHistory (amrex::Real time);
    void initializePressureForceHistory (bool restart);
    void writeEBSurfaceData (amrex::Real time);
    void collectPhysicalWallData (
        amrex::Real time,
        std::array<amrex::Real,3>& pressure_force,
        std::array<amrex::Real,3>& viscous_force,
        std::array<amrex::Real,3>& pressure_moment,
        std::array<amrex::Real,3>& viscous_moment,
        amrex::Real& surface_area,
        amrex::Real& heat_transfer,
        std::vector<amrex::Real>* samples) const;
    void writePhysicalWallSurfaceData (
        amrex::Real time, const std::string& directory);

    const amrex::MultiFab* volfrac;
    const amrex::MultiCutFab* bndrycent;
    std::array<const amrex::MultiCutFab*,AMREX_SPACEDIM> areafrac;
    std::array<const amrex::MultiCutFab*,AMREX_SPACEDIM> facecent;

    amrex::iMultiFab level_mask;

    amrex::EBFluxRegister flux_reg;

    std::unique_ptr<amrex::MultiFab> residual_rhs;
    bool residual_rhs_valid = false;

    static constexpr int level_mask_interior   = 0; // valid cells
    static constexpr int level_mask_covered    = 1; // ghost cells covered by valid cells of this level
    static constexpr int level_mask_notcovered = 2; // ghost cells not covered
    static constexpr int level_mask_physbnd    = 3; // outside domain

    static constexpr int NUM_GROW = 5;
    enum StateVariable {
        Density = 0, Xmom, Ymom, Zmom, Eden, Eint, Temp, volfraction,
        NUM_STATE
    };

    enum StateDataType {
        State_Type = 0,
        Cost_Type
    };
    static int num_state_data_types;

    static amrex::BCRec phys_bc;

    // Parameters
    static int verbose;
    static amrex::IntVect hydro_tile_size;
    static amrex::Real cfl;
    static bool moving_wall_cfl;
    static bool moving_wall_cfl_diagnostics;
    static int do_reflux;

    static int refine_cutcells;

    static int residual_int;
    static amrex::Real residual_reference_floor;
    static std::string residual_file;
    static bool residual_file_initialized;

    static int force_int;
    static std::string force_file;
    static bool force_file_initialized;
    static std::array<amrex::Real,3> force_reference_point;
    static std::array<amrex::Real,3> force_drag_direction;
    static std::array<amrex::Real,3> force_lift_direction;
    static std::array<amrex::Real,3> force_side_direction;
    static amrex::Real force_reference_density;
    static amrex::Real force_reference_velocity;
    static amrex::Real force_reference_area;
    static amrex::Real force_reference_length;
    static amrex::Real force_pressure_offset;
    static amrex::Real force_reference_pressure;
    static int surface_int;
    static std::string surface_prefix;

    static int refine_max_pressure_jump_lev;
    static amrex::Real refine_pressure_jump;
    static amrex::Real refine_pressure_scale_floor;
    static bool refine_pressure_window_enabled;
    static std::array<amrex::Real,3> refine_pressure_window_center;
    static std::array<amrex::Real,3> refine_pressure_window_velocity;
    static std::array<amrex::Real,3> refine_pressure_window_lo;
    static std::array<amrex::Real,3> refine_pressure_window_hi;
    static bool refine_sensor_verbose;

    static amrex::Vector<amrex::RealBox> refine_boxes;
};

#endif
