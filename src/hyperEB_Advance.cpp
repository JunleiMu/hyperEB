// Source attribution and license obligations are recorded in THIRD_PARTY_NOTICES.md.

#include <hyperEB.H>
#include <hyperEB_EmbeddedBoundary.H>
#include <hyperEB_F.H>
#include <hyperEB_MovingBoundary.H>
#include <hyperEB_RegularGrid.H>
#include <AMReX_Amr.H>
#include <AMReX_EBFArrayBox.H>
#include <AMReX_MultiCutFab.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_iMultiFab.H>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

using namespace amrex;
bool eb2_geometry_is_dynamic();
amrex::Vector<amrex::MultiFab> volfracmfab, voltemp;

Real HyperEB::advance(Real time, Real dt, int iteration, int ncycle)
{

    BL_PROFILE("HyperEB::advance()");

    const bool dynamic_geometry = eb2_geometry_is_dynamic();

    // Derive the geometry-update phase from checkpointed AMR state so restart and
    // continuous runs perform uncovering on the same fine-level substep.
    const int stepval = parent->levelSteps(level) + 1;
    int cumulative_subcycles = 1;
    for (int lev = 0; lev <= level; ++lev)
    {
        cumulative_subcycles *= parent->nCycle(lev);
    }
    const bool do_uncovering = dynamic_geometry
        && stepval > 1
        && (stepval - 1) % cumulative_subcycles == 0;

    std::unique_ptr<MultiFab> old_volfrac;
    if (do_uncovering)
    {
        old_volfrac = std::make_unique<MultiFab>(
            grids, dmap, 1, NUM_GROW, MFInfo(), Factory());
        FillPatch(*this, *old_volfrac, NUM_GROW, time, State_Type, 7, 1, 0);
    }

    for (int i = 0; i < num_state_data_types; ++i)
    {
        state[i].allocOldData();
        state[i].swapTimeLevels(dt);
    }

    MultiFab &S_new = get_new_data(State_Type);
    MultiFab &S_old = get_old_data(State_Type);
    MultiFab Sborder(grids, dmap, NUM_STATE, NUM_GROW, MFInfo(), Factory());
    MultiFab dSdt(grids, dmap, NUM_STATE, 0, MFInfo(), Factory());

    MultiFab &C_new = get_new_data(Cost_Type);
    C_new.setVal(0.0);

    EBFluxRegister *fr_as_crse = nullptr;
    if (do_reflux && level < parent->finestLevel())
    {
        HyperEB &fine_level = getLevel(level + 1);
        fr_as_crse = &fine_level.flux_reg;
    }

    EBFluxRegister *fr_as_fine = nullptr;
    if (do_reflux && level > 0)
    {
        fr_as_fine = &flux_reg;
    }
    if (fr_as_crse)
    {
        fr_as_crse->reset();
    }

    FillPatch(*this, Sborder, NUM_GROW, time, State_Type, 0, NUM_STATE);
    if (do_uncovering)
    {
        MultiFab cons_adjust(grids, dmap, NUM_STATE, NUM_GROW, MFInfo(), Factory());
        MultiFab donortag(grids, dmap, 1, NUM_GROW, MFInfo(), Factory());
        ComputeUncoveringRedistribute(
            *old_volfrac, Sborder, cons_adjust, donortag, time, stepval);
        FinalUncoveringRedistribute(
            *old_volfrac, Sborder, cons_adjust, donortag);
        MultiFab::Copy(S_old, Sborder, 0, 0, 7, 4);

        // Rebuild same-level, coarse/fine, periodic, and physical-boundary
        // ghost data from the uncovered state now stored in StateData.
        FillPatch(
            *this, Sborder, NUM_GROW,
            time, State_Type, 0, NUM_STATE);
    }
    checkAdmissibility(Sborder, "after fill/uncovering", time);

    // ============================================================
    // SSP-RK2 stage 1
    //
    // U* = U^n + dt L(U^n)
    // ============================================================

    dSdt.setVal(0.0);

    compute_dSdt(
        Sborder, dSdt,
        dt, 0.5,
        fr_as_crse, fr_as_fine,
        time);

    MultiFab::LinComb(
        S_new,
        1.0, Sborder, 0,
        dt, dSdt, 0,
        0, NUM_STATE, 0);

    checkAdmissibility(S_new, "after RK2 stage 1", time + dt);
    computeTemp(S_new, 0);

    // ============================================================
    // SSP-RK2 stage 2
    //
    // U^{n+1} = 0.5 U^n
    //           + 0.5 [U* + dt L(U*)]
    // ============================================================

    FillPatch(
        *this, Sborder, NUM_GROW,
        time + dt,
        State_Type, 0, NUM_STATE);
    checkAdmissibility(Sborder, "RK2 stage 2 input", time + dt);

    dSdt.setVal(0.0);

    compute_dSdt(
        Sborder, dSdt,
        dt, 0.5,
        fr_as_crse, fr_as_fine,
        time + dt);

    MultiFab::LinComb(
        S_new,
        0.5, Sborder, 0,
        0.5, S_old, 0,
        0, NUM_STATE, 0);

    MultiFab::Saxpy(
        S_new,
        0.5 * dt,
        dSdt, 0,
        0, NUM_STATE, 0);

    checkAdmissibility(S_new, "after RK2 stage 2", time + dt);
    computeTemp(S_new, 0);

    // Store the effective whole-step RHS used by residual diagnostics. For
    // multi-stage RK methods the last stage RHS alone is not the time update.
    MultiFab::LinComb(
        dSdt,
        1.0 / dt, S_new, 0,
        -1.0 / dt, S_old, 0,
        0, NUM_STATE, 0);
    recordResidualRhs(dSdt);

    MultiFab::Copy(S_new, *volfrac, 0, 7, 1, 4);

    return dt;
}

void HyperEB::compute_dSdt(const MultiFab &S, MultiFab &dSdt,
                       Real stage_dt, Real reflux_weight,
                       EBFluxRegister *fr_as_crse, EBFluxRegister *fr_as_fine,
                       Real time)
{
    BL_PROFILE("HyperEB::compute_dSdt()");

    const Real *dx = geom.CellSize();
    const auto cell_size = geom.CellSizeArray();
    const int ncomp = dSdt.nComp();
    const auto physics = hypereb::kernels::regular_physics();
    const Real reflux_dt = reflux_weight * stage_dt;
    const BCRec *state_bcs = desc_lst[State_Type].getBCs().data();
    MultiFab &cost = get_new_data(Cost_Type);
    auto const &fact = dynamic_cast<EBFArrayBoxFactory const &>(S.Factory());
    auto const &flags = fact.getMultiEBCellFlagFab();

    const int as_crse = fr_as_crse != nullptr;
    const int as_fine = fr_as_fine != nullptr;

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        std::array<FArrayBox, AMREX_SPACEDIM> flux;
        FArrayBox dm_as_fine(Box::TheUnitBox(), ncomp);
        FArrayBox fab_drho_as_crse(Box::TheUnitBox(), ncomp);
        IArrayBox fab_rrflag_as_crse(Box::TheUnitBox());

        for (MFIter mfi(S, MFItInfo().EnableTiling(hydro_tile_size).SetDynamic(true));
             mfi.isValid(); ++mfi)
        {
            const Box &bx = mfi.tilebox();
            Real wt = amrex::second();
            const auto &flag = flags[mfi];

            if (flag.getType(bx) == FabType::covered)
            {
                dSdt[mfi].setVal(0.0, bx, 0, ncomp);
            }
            else
            {
                for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
                {
                    flux[idim].resize(amrex::surroundingNodes(bx, idim), ncomp);
                }

                if (flag.getType(amrex::grow(bx, 2)) == FabType::regular)
                {
                    const std::array<Array4<Real>, 3> flux_arrays{
                        flux[0].array(), flux[1].array(), flux[2].array()};
                    hypereb::kernels::compute_regular_rhs(
                        bx, dSdt[mfi].array(), S[mfi].const_array(), flux_arrays,
                        cell_size, physics);
                    if (as_crse)
                    {
                        fr_as_crse->CrseAdd(
                            mfi, {&flux[0], &flux[1], &flux[2]}, dx, reflux_dt,
                            RunOn::Cpu);
                    }
                    if (as_fine)
                    {
                        fr_as_fine->FineAdd(
                            mfi, {&flux[0], &flux[1], &flux[2]}, dx, reflux_dt,
                            RunOn::Cpu);
                    }
                }
                else
                {
                    FArrayBox *p_drho_as_crse = as_crse
                                                    ? fr_as_crse->getCrseData(mfi)
                                                    : &fab_drho_as_crse;
                    const IArrayBox *p_rrflag_as_crse = as_crse
                                                            ? fr_as_crse->getCrseFlag(mfi)
                                                            : &fab_rrflag_as_crse;

                    if (as_fine)
                    {
                        dm_as_fine.resize(amrex::grow(bx, 1), ncomp);
                    }

                    const std::array<Array4<Real>, 3> flux_arrays{
                        flux[0].array(), flux[1].array(), flux[2].array()};
                    const hypereb::kernels::EBGeometryViews eb_views{
                        flag.const_array(),
                        (*volfrac)[mfi].const_array(),
                        fact.getCentroid()[mfi].const_array(),
                        (*bndrycent)[mfi].const_array(),
                        {(*areafrac[0])[mfi].const_array(),
                         (*areafrac[1])[mfi].const_array(),
                         (*areafrac[2])[mfi].const_array()},
                        {(*facecent[0])[mfi].const_array(),
                         (*facecent[1])[mfi].const_array(),
                         (*facecent[2])[mfi].const_array()}};
                    const hypereb::kernels::EBCouplingViews coupling{
                        as_crse,
                        p_drho_as_crse->array(),
                        p_rrflag_as_crse->const_array(),
                        as_fine,
                        dm_as_fine.array(),
                        level_mask[mfi].const_array(),
                        level_mask_notcovered};
                    hypereb::kernels::compute_eb_rhs(
                        bx, dSdt[mfi].array(), S[mfi].const_array(), flux_arrays,
                        geom, physics, eb_views, coupling,
                        state_bcs,
                        stage_dt, reflux_weight, time);

                    if (as_crse)
                    {
                        fr_as_crse->CrseAdd(
                            mfi, {&flux[0], &flux[1], &flux[2]}, dx, reflux_dt,
                            (*volfrac)[mfi],
                            {&((*areafrac[0])[mfi]),
                             &((*areafrac[1])[mfi]),
                             &((*areafrac[2])[mfi])},
                            RunOn::Cpu);
                    }
                    if (as_fine)
                    {
                        fr_as_fine->FineAdd(
                            mfi, {&flux[0], &flux[1], &flux[2]}, dx, reflux_dt,
                            (*volfrac)[mfi],
                            {&((*areafrac[0])[mfi]),
                             &((*areafrac[1])[mfi]),
                             &((*areafrac[2])[mfi])},
                            dm_as_fine, RunOn::Cpu);
                    }
                }
            }

            wt = (amrex::second() - wt) / bx.d_numPts();
            cost[mfi].plus(wt, bx);
        }
    }
}


void HyperEB::ZeroOutSolidWalls(const int *lev, MultiFab &S)
{

    amrex::ignore_unused(lev);

#ifdef _OPENMP
#pragma omp parallel
#endif
    // for (MFIter mfi(S, MFItInfo().EnableTiling(hydro_tile_size).SetDynamic(true));
    //                      mfi.isValid(); ++mfi)
    for (MFIter mfi(S); mfi.isValid(); ++mfi)
    {

        const Box &bx = mfi.tilebox();

        hypereb::kernels::zero_out_solid_cells(
            bx, S[mfi].array(), S.nComp(), (*volfrac)[mfi].const_array(), 1.226);
    }
}

void HyperEB::ComputeUncoveringRedistribute(const MultiFab &volfracborder, MultiFab &Sborder,
                                        MultiFab &cons_adjust, MultiFab &donortag,
                                        Real time, int moving_step)
{

    cons_adjust.setVal(0.0);
    donortag.setVal(0.0);

    const auto physics=hypereb::kernels::regular_physics();
    const auto cell_size=geom.CellSizeArray();
    const auto& factory=
        dynamic_cast<const EBFArrayBoxFactory&>(Sborder.Factory());
    const auto& flags=factory.getMultiEBCellFlagFab();
    const auto& cell_centroid=factory.getCentroid();

#ifdef _OPENMP
#pragma omp parallel
#endif

    // std::cout << S.boxArray() << "\n";

    // for (MFIter mfi(S, MFItInfo().EnableTiling(hydro_tile_size).SetDynamic(true));
    //        mfi.isValid(); ++mfi)
    for (MFIter mfi(Sborder); mfi.isValid(); ++mfi)
    {
        const Box &bx = mfi.tilebox();

        hypereb::kernels::prepare_uncovered_cells(
            bx, volfracborder[mfi].const_array(), (*volfrac)[mfi].const_array(),
            flags[mfi].const_array(),cell_centroid[mfi].const_array(),
            Sborder[mfi].array(), cons_adjust[mfi].array(), donortag[mfi].array(),
            cell_size,physics.gamma,physics.cv,
            level, time, moving_step, NUM_STATE);
    }
}

void HyperEB::FinalUncoveringRedistribute(const MultiFab &volfracborder, MultiFab &Sborder, MultiFab &cons_adjust, MultiFab &donortag)
{

#ifdef _OPENMP
#pragma omp parallel
#endif

    for (MFIter mfi(Sborder); mfi.isValid(); ++mfi)
    {
        const Box &bx = mfi.tilebox();

        hypereb::kernels::apply_uncovered_cells(
            bx, Sborder[mfi].array(), cons_adjust[mfi].const_array(),
            donortag[mfi].const_array(), NUM_STATE);

    }

}
