#ifndef HYPEREB_HYPEREB_MOTION_H_
#define HYPEREB_HYPEREB_MOTION_H_

#include <AMReX_REAL.H>
#include <AMReX_ParmParse.H>

#include <array>
#include <cmath>
#include <string>

namespace hypereb::motion {

struct Translation
{
    std::array<amrex::Real, 3> velocity{};
    std::array<amrex::Real, 3> acceleration{};

    bool dynamic() const noexcept
    {
        for (int d = 0; d < 3; ++d)
        {
            if (velocity[d] != 0.0 || acceleration[d] != 0.0)
                return true;
        }
        return false;
    }

    amrex::Real displacement(int d, amrex::Real time) const noexcept
    {
        return velocity[d] * time + 0.5 * acceleration[d] * time * time;
    }

    amrex::Real speed(int d, amrex::Real time) const noexcept
    {
        return velocity[d] + acceleration[d] * time;
    }
};

inline Translation read_translation()
{
    amrex::ParmParse pp("eb2");
    std::string geometry;
    pp.get("geom_type", geometry);
    if (geometry != "moving_plane" && geometry != "stl")
        amrex::Abort("eb2.geom_type must be moving_plane or stl");

    const auto read = [&](const char* name, std::array<amrex::Real, 3>& values)
    {
        if (!pp.contains(name)) return false;
        if (pp.countval(name) != 3)
            amrex::Abort(std::string("eb2.") + name + " requires three components");
        amrex::Vector<amrex::Real> input;
        pp.getarr(name, input, 0, 3);
        for (int d = 0; d < 3; ++d)
        {
            if (!std::isfinite(input[d]))
                amrex::Abort(std::string("eb2.") + name + " must be finite");
            values[d] = input[d];
        }
        return true;
    };

    Translation motion;
    const bool wall_velocity = read("wall_translation", motion.velocity);
    const bool wall_acceleration = read("wall_acceleration", motion.acceleration);
    if (geometry == "moving_plane")
    {
        Translation plane = motion;
        const bool plane_velocity = read("plane_velocity", plane.velocity);
        const bool plane_acceleration = read("plane_acceleration", plane.acceleration);
        if ((wall_velocity && plane_velocity && motion.velocity != plane.velocity) ||
            (wall_acceleration && plane_acceleration &&
             motion.acceleration != plane.acceleration))
            amrex::Abort("Plane geometry and wall flux must prescribe the same motion");
        motion = plane;
    }
    else if (pp.contains("plane_velocity") || pp.contains("plane_acceleration"))
    {
        amrex::Abort("STL motion uses eb2.wall_translation and eb2.wall_acceleration");
    }
    return motion;
}


} // namespace hypereb::motion

#endif
