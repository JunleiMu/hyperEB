
#include <AMReX_LevelBld.H>
#include <hyperEB.H>

using namespace amrex;

class HyperEBBuild
    :
    public LevelBld
{
    virtual void variableSetUp () override;
    virtual void variableCleanUp () override;
    virtual AmrLevel *operator() () override;
    virtual AmrLevel *operator() (Amr&            papa,
                                  int             lev,
                                  const Geometry& level_geom,
                                  const BoxArray& ba,
                                  const DistributionMapping& dm,
                                  Real            time) override;
};

HyperEBBuild HYPEREB_bld;

LevelBld*
getLevelBld ()
{
    return &HYPEREB_bld;
}

void
HyperEBBuild::variableSetUp ()
{
    HyperEB::variableSetUp();
}

void
HyperEBBuild::variableCleanUp ()
{
    HyperEB::variableCleanUp();
}

AmrLevel*
HyperEBBuild::operator() ()
{
    return new HyperEB;
}

AmrLevel*
HyperEBBuild::operator() (Amr&            papa,
                    int             lev,
                    const Geometry& level_geom,
                    const BoxArray& ba,
                    const DistributionMapping& dm,
                    Real            time)
{
    return new HyperEB(papa, lev, level_geom, ba, dm, time);
}
