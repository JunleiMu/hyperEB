#include <AMReX_EB2.H>
#include <AMReX_EB2_IF.H>
#include <AMReX_EB2_IndexSpace_STL.H>
#include <AMReX_ParmParse.H>
#include <hyperEB_Motion.H>

#include <algorithm>
#include <filesystem>
#include <memory>
#include <string>

using namespace amrex;

namespace
{

    std::filesystem::path resolve_case_asset(const std::string &configured_path,
                                             const std::string &subdirectory = {})
    {
        const std::filesystem::path requested(configured_path);
        if (std::filesystem::exists(requested))
        {
            return requested;
        }
#ifdef HYPEREB_CASES_DIR
        const std::filesystem::path cases_directory(HYPEREB_CASES_DIR);
        const std::filesystem::path relative_to_cases = cases_directory / requested;
        if (std::filesystem::exists(relative_to_cases))
        {
            return relative_to_cases;
        }
        const std::filesystem::path in_subdirectory = cases_directory / subdirectory / requested.filename();
        if (std::filesystem::exists(in_subdirectory))
        {
            return in_subdirectory;
        }
        const std::filesystem::path in_cases = cases_directory / requested.filename();
        if (std::filesystem::exists(in_cases))
        {
            return in_cases;
        }
#endif
        amrex::Abort("Cannot resolve case asset: " + configured_path);
        return requested;
    }

    RealArray read_real_array(ParmParse &parameters, const char *name,
                              RealArray defaults)
    {
        if (parameters.contains(name) && parameters.countval(name) != 3)
            amrex::Abort(std::string("eb2.") + name + " requires three components");
        Vector<Real> values(defaults.begin(), defaults.end());
        if (parameters.queryarr(name, values, 0, AMREX_SPACEDIM))
        {
            std::copy(values.begin(), values.end(), defaults.begin());
        }
        for (Real value : defaults)
        {
            if (!std::isfinite(value))
                amrex::Abort(std::string("eb2.") + name + " must be finite");
        }
        return defaults;
    }

} // namespace

bool eb2_geometry_is_dynamic()
{
    ParmParse ppeb2("eb2");
    std::string geom_type;
    ppeb2.get("geom_type", geom_type);
    if (geom_type != "moving_plane" && geom_type != "stl")
    {
        amrex::Abort("eb2.geom_type must be moving_plane or stl");
    }
    const bool dynamic = geom_type == "moving_plane" ||
                         hypereb::motion::read_translation().dynamic();
    int configured_dynamic = dynamic ? 1 : 0;
    ppeb2.query("dynamic_geometry", configured_dynamic);
    if ((configured_dynamic != 0) != dynamic)
    {
        amrex::Abort("eb2.dynamic_geometry conflicts with the prescribed geometry motion");
    }
    return dynamic;
}

void initialize_EB2(const Geometry &geom, const int required_coarsening_level,
                    const int max_coarsening_level, Real time)
{
    BL_PROFILE("initializeEB2");

    ParmParse ppeb2("eb2");
    std::string geom_type;
    ppeb2.get("geom_type", geom_type);
    eb2_geometry_is_dynamic();
    const auto motion = hypereb::motion::read_translation();
    if (!std::isfinite(time))
        amrex::Abort("EB geometry time must be finite");
    // Native weighted state redistribution requires five geometry ghost cells.
    const int eb_grow_cells = 5;

    if (geom_type == "moving_plane")
    {
        RealArray point = read_real_array(ppeb2, "plane_initial_point", {0.0, 0.0, 0.0});
        RealArray normal = read_real_array(ppeb2, "plane_normal", {-1.0, 0.0, 0.0});
        if (normal[0] == 0.0 && normal[1] == 0.0 && normal[2] == 0.0)
            amrex::Abort("eb2.plane_normal must be nonzero");
        for (int d = 0; d < AMREX_SPACEDIM; ++d)
        {
            point[d] += motion.displacement(d, time);
            if (!std::isfinite(point[d]))
                amrex::Abort("Nonfinite moving-plane position");
        }

        EB2::PlaneIF pf(point, normal);

        EB2::GeometryShop<EB2::PlaneIF> gshop(pf);
        EB2::Build(gshop, geom, required_coarsening_level,
                   max_coarsening_level, eb_grow_cells);
    }
    else if (geom_type == "stl")
    {
        std::string configured_file;
        ppeb2.get("stl_file", configured_file);
        const auto resolved_file = resolve_case_asset(configured_file, "geometry");
        Real scale = 1.0;
        ppeb2.query("stl_scale", scale);
        if (!(scale > 0.0) || !std::isfinite(scale))
            amrex::Abort("eb2.stl_scale must be finite and positive");
        RealArray center = read_real_array(ppeb2, "stl_center", {0.0, 0.0, 0.0});
        for (int d = 0; d < 3; ++d)
        {
            center[d] += motion.displacement(d, time);
            if (!std::isfinite(center[d]))
                amrex::Abort("Nonfinite translated STL position");
        }
        bool reverse_normal = false;
        bool use_bvh = true;
        ppeb2.query("stl_reverse_normal", reverse_normal);
        ppeb2.query("stl_use_bvh", use_bvh);
        int build_coarse_by_coarsening = 0;
        ppeb2.query("build_coarse_by_coarsening", build_coarse_by_coarsening);
        // Rebuild from the original mesh and absolute-time transform, never
        // accumulate displacement in ParmParse across AMR stages or restart.
        EB2::IndexSpace::push(std::make_unique<EB2::IndexSpaceSTL>(
            std::filesystem::absolute(resolved_file).string(), scale, center,
            int(reverse_normal), geom, required_coarsening_level,
            max_coarsening_level, eb_grow_cells, build_coarse_by_coarsening != 0,
            EB2::ExtendDomainFace(), EB2::NumCoarsenOpt(), use_bvh, false));
    }
    else
    {
        amrex::Abort("eb2.geom_type must be moving_plane or stl");
    }
}

void finalize_EB2()
{
    EB2::Finalize();
}
