#include <hyperEB_F.H>
#include <hyperEB_RegularGrid.H>
#include <hyperEB_Motion.H>
#include <eb_kernels/hyperEB_diffusion_K.H>

#include <AMReX.H>
#include <AMReX_BC_TYPES.H>
#include <AMReX_ParmParse.H>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <string>

namespace
{

    constexpr int density = 0;
    constexpr int xmom = 1;
    constexpr int ymom = 2;
    constexpr int zmom = 3;
    constexpr int total_energy = 4;
    constexpr int internal_energy = 5;
    constexpr int temperature = 6;
    constexpr int volume_fraction = 7;
    constexpr int state_components = 8;

    enum class ProblemCase
    {
        regular_uniform,
        smooth_expanding_piston
    };

    struct RuntimeParameters
    {
        amrex::Real gamma = 1.4;
        amrex::Real molecular_weight = 28.97;
        amrex::Real prandtl = 0.72;
        amrex::Real sutherland_coefficient = 1.458e-6;
        amrex::Real sutherland_temperature = 110.4;
        bool viscosity_power_law = false;
        amrex::Real viscosity_reference_mu = 0.0;
        amrex::Real viscosity_reference_temperature = 0.0;
        amrex::Real viscosity_power_exponent = 0.0;
        amrex::Real cv = 0.0;
        amrex::Real cp = 0.0;
        amrex::Real specific_gas_constant = -1.0;

        amrex::Real rho0 = 1.226;
        amrex::Real pressure0 = 101325.0;
        amrex::Real mach = 0.05;
        amrex::Real gas_constant = 287.0;
        amrex::Real angle_of_attack_degrees = 0.0;
        amrex::Real sideslip_degrees = 0.0;
        bool freestream_velocity_specified = false;
        std::array<amrex::Real, 3> freestream_velocity{};
        amrex::Real piston_velocity = -0.5;

        bool use_constant_viscosity = false;
        bool do_diffusion = true;
        bool eb_wall_diffusion = true;
        hypereb::kernels::EBThermalBoundary eb_thermal_boundary = hypereb::kernels::EBThermalBoundary::adiabatic;
        amrex::Real eb_wall_temperature = 300.0;
        bool eb_noslip_segment = false;
        std::array<amrex::Real, 3> eb_noslip_origin{};
        std::array<amrex::Real, 3> eb_noslip_tangent{1.0, 0.0, 0.0};
        amrex::Real eb_noslip_s_lo = 0.0;
        amrex::Real eb_noslip_s_hi = 1.0;
        hypereb::kernels::EBThermalBoundary wall_thermal_boundary = hypereb::kernels::EBThermalBoundary::adiabatic;
        amrex::Real wall_temperature = 300.0;
        amrex::Real constant_mu = -1.0;
        amrex::Real constant_xi = -1.0;
        amrex::Real constant_lambda = -1.0;

        bool admissibility_diagnostics = false;
        std::string case_name = "external_flow";
        ProblemCase problem_case = ProblemCase::regular_uniform;
        std::array<int, 3> physical_low{};
        std::array<int, 3> physical_high{};
        std::array<amrex::Real, 3> problem_low{};
        std::array<amrex::Real, 3> problem_high{};
        std::array<amrex::Real, 3> wall_translation{};
        std::array<amrex::Real, 3> wall_acceleration{};
        std::array<amrex::Real, state_components> inflow{};
        std::array<amrex::Real, state_components> interior{};
    };

    RuntimeParameters runtime;

    template <typename T>
    class FabView
    {
    public:
        FabView(T *data, const int *low, const int *high) noexcept
            : data_(data), low_{low[0], low[1], low[2]},
              nx_(high[0] - low[0] + 1), ny_(high[1] - low[1] + 1),
              nz_(high[2] - low[2] + 1)
        {
        }

        T &operator()(int i, int j, int k, int component = 0) const noexcept
        {
            const std::size_t cell = static_cast<std::size_t>(i - low_[0]) + static_cast<std::size_t>(nx_) * (static_cast<std::size_t>(j - low_[1]) + static_cast<std::size_t>(ny_) * static_cast<std::size_t>(k - low_[2]));
            return data_[cell + static_cast<std::size_t>(nx_) * ny_ * nz_ * component];
        }

    private:
        T *data_;
        std::array<int, 3> low_;
        int nx_;
        int ny_;
        int nz_;
    };

    void update_thermodynamics()
    {
        // SI units: eos_mu is kg/kmol and Ru is J/(kmol K).
        constexpr amrex::Real universal_gas_constant = 8314.46261815324;
        const amrex::Real gas_constant = runtime.specific_gas_constant > 0.0
                                             ? runtime.specific_gas_constant
                                             : universal_gas_constant / runtime.molecular_weight;
        runtime.cv = gas_constant / (runtime.gamma - 1.0);
        runtime.cp = runtime.gamma * runtime.cv;
    }

    void set_conservative_state(std::array<amrex::Real, state_components> &state,
                                amrex::Real rho, amrex::Real u, amrex::Real v,
                                amrex::Real w, amrex::Real pressure,
                                amrex::Real state_temperature)
    {
        const amrex::Real rho_e = pressure / (runtime.gamma - 1.0);
        state[density] = rho;
        state[xmom] = rho * u;
        state[ymom] = rho * v;
        state[zmom] = rho * w;
        state[total_energy] = rho_e + 0.5 * rho * (u * u + v * v + w * w);
        state[internal_energy] = rho_e;
        state[temperature] = state_temperature;
        state[volume_fraction] = 1.0;
    }

    void update_generic_states()
    {
        const amrex::Real sound_speed = std::sqrt(
            runtime.gamma * runtime.pressure0 / runtime.rho0);
        std::array<amrex::Real, 3> velocity = runtime.freestream_velocity;
        if (!runtime.freestream_velocity_specified)
        {
            constexpr amrex::Real pi = 3.14159265358979323846;
            const amrex::Real alpha = runtime.angle_of_attack_degrees * pi / 180.0;
            const amrex::Real beta = runtime.sideslip_degrees * pi / 180.0;
            const amrex::Real magnitude = runtime.mach * sound_speed;
            velocity = {
                magnitude * std::cos(alpha) * std::cos(beta),
                magnitude * std::sin(beta),
                magnitude * std::sin(alpha) * std::cos(beta)};
        }
        const amrex::Real temperature = runtime.pressure0 / (runtime.gas_constant * runtime.rho0);
        set_conservative_state(runtime.inflow, runtime.rho0,
                               velocity[0], velocity[1], velocity[2],
                               runtime.pressure0, temperature);
        runtime.interior = runtime.inflow;
    }

    ProblemCase parse_problem_case(const std::string &name)
    {
        if (name == "external_flow")
            return ProblemCase::regular_uniform;
        if (name == "regular_uniform_test")
            return ProblemCase::regular_uniform;
        if (name == "smooth_expanding_piston")
            return ProblemCase::smooth_expanding_piston;
        amrex::Abort("Unsupported case: " + name);
        return ProblemCase::regular_uniform;
    }

    bool query_real3(amrex::ParmParse &parameters, const char *name,
                     std::array<amrex::Real, 3> &destination)
    {
        amrex::Vector<amrex::Real> values(destination.begin(), destination.end());
        if (!parameters.queryarr(name, values, 0, 3))
        {
            return false;
        }
        std::copy(values.begin(), values.end(), destination.begin());
        return true;
    }

    void read_problem_parameters()
    {
        amrex::ParmParse root;
        root.query("case", runtime.case_name);
        runtime.problem_case = parse_problem_case(runtime.case_name);
        runtime.freestream_velocity_specified = false;
        runtime.freestream_velocity = {0.0, 0.0, 0.0};
        runtime.angle_of_attack_degrees = 0.0;
        runtime.sideslip_degrees = 0.0;
        amrex::ParmParse problem("prob");
        problem.query("rho0", runtime.rho0);
        problem.query("rho_inf", runtime.rho0);
        problem.query("p0", runtime.pressure0);
        problem.query("p_inf", runtime.pressure0);
        problem.query("piston_velocity", runtime.piston_velocity);
        bool mach_was_set = problem.query("mach", runtime.mach);
        mach_was_set = problem.query("mach_inf", runtime.mach) || mach_was_set;
        problem.query("gas_constant", runtime.gas_constant);
        bool angles_were_set = problem.query(
            "aoa", runtime.angle_of_attack_degrees);
        angles_were_set = problem.query(
                              "aoa_degrees", runtime.angle_of_attack_degrees) ||
                          angles_were_set;
        angles_were_set = problem.query(
                              "beta", runtime.sideslip_degrees) ||
                          angles_were_set;
        angles_were_set = problem.query(
                              "beta_degrees", runtime.sideslip_degrees) ||
                          angles_were_set;
        if (mach_was_set || angles_were_set)
        {
            runtime.freestream_velocity_specified = false;
        }
        if (query_real3(problem, "velocity_inf", runtime.freestream_velocity))
        {
            runtime.freestream_velocity_specified = true;
        }
        const auto finite_vector = [](const std::array<amrex::Real, 3> &values)
        {
            return std::isfinite(values[0]) && std::isfinite(values[1]) && std::isfinite(values[2]);
        };
        if (!(runtime.rho0 > 0.0) || !(runtime.pressure0 > 0.0) || !(runtime.gas_constant > 0.0) || runtime.mach < 0.0 || !std::isfinite(runtime.mach) || !std::isfinite(runtime.angle_of_attack_degrees) || !std::isfinite(runtime.sideslip_degrees) || !std::isfinite(runtime.piston_velocity) || !finite_vector(runtime.freestream_velocity))
        {
            amrex::Abort("freestream rho, pressure, gas constant, and Mach number must be physical");
        }
    }

    void configure_problem_states()
    {
        update_generic_states();
    }

    struct ReflectedIndex
    {
        int index;
        bool odd = false;
        bool external = false;
    };

    constexpr ReflectedIndex reflected_index(
        int index, int domain_low, int domain_high, int low_bc, int high_bc)
    {
        bool odd = false;
        // A ghost layer can span the domain more than once.
        while (index < domain_low || index > domain_high)
        {
            const bool low_side = index < domain_low;
            const int boundary = low_side ? low_bc : high_bc;
            if (boundary != amrex::BCType::reflect_even &&
                boundary != amrex::BCType::reflect_odd)
            {
                return {low_side ? domain_low : domain_high, odd,
                        boundary == amrex::BCType::ext_dir};
            }
            odd = odd != (boundary == amrex::BCType::reflect_odd);
            index = low_side ? 2 * domain_low - index - 1
                             : 2 * domain_high - index + 1;
        }
        return {index, odd, false};
    }

    static_assert(reflected_index(-5, 0, 3, amrex::BCType::reflect_even,
                                  amrex::BCType::reflect_even).index == 3);
    static_assert(reflected_index(8, 0, 3, amrex::BCType::reflect_even,
                                  amrex::BCType::reflect_even).index == 0);
    static_assert(!reflected_index(-5, 0, 3, amrex::BCType::reflect_odd,
                                   amrex::BCType::reflect_odd).odd);
    static_assert(reflected_index(-9, 0, 3, amrex::BCType::reflect_odd,
                                  amrex::BCType::reflect_odd).odd);
    static_assert(reflected_index(-5, 0, 3, amrex::BCType::reflect_odd,
                                  amrex::BCType::reflect_even).odd);
    static_assert(reflected_index(-5, 0, 3, amrex::BCType::reflect_even,
                                  amrex::BCType::ext_dir).external);
    static_assert(reflected_index(-5, 0, 3, amrex::BCType::reflect_even,
                                  amrex::BCType::foextrap).index == 3);
    static_assert(reflected_index(-5, 0, 0, amrex::BCType::reflect_even,
                                  amrex::BCType::reflect_even).index == 0);

    bool physical_no_slip_wall(int direction, bool low_side) noexcept
    {
        const int physical_type = low_side ? runtime.physical_low[direction]
                                           : runtime.physical_high[direction];
        return physical_type == 5;
    }

    void fill_component(amrex::Real *data, const int *data_low, const int *data_high,
                        const int *domain_low, const int *domain_high, const int *bc,
                        int component, amrex::Real exterior_value,
                        const amrex::Real *spacing)
    {
        FabView<amrex::Real> state(data, data_low, data_high);
        for (int k = data_low[2]; k <= data_high[2]; ++k)
        {
            for (int j = data_low[1]; j <= data_high[1]; ++j)
            {
                for (int i = data_low[0]; i <= data_high[0]; ++i)
                {
                    if (i >= domain_low[0] && i <= domain_high[0] && j >= domain_low[1] && j <= domain_high[1] && k >= domain_low[2] && k <= domain_high[2])
                    {
                        continue;
                    }

                    bool external = false;
                    bool odd = false;
                    const int indices[3] = {i, j, k};
                    int source[3] = {i, j, k};
                    for (int direction = 0; direction < 3; ++direction)
                    {
                        const int low_bc = bc[direction];
                        const int high_bc = bc[direction + 3];
                        const auto reflected = reflected_index(
                            indices[direction], domain_low[direction], domain_high[direction],
                            low_bc, high_bc);
                        source[direction] = reflected.index;
                        external = external || reflected.external;
                        odd = odd != reflected.odd;
                    }

                    state(i, j, k, component) = external
                                                    ? exterior_value
                                                    : (odd ? -1.0 : 1.0) * state(source[0], source[1], source[2], component);
                }
            }
        }
    }

    void unexpected_boundary_callback(const char *name)
    {
        amrex::Abort(std::string("Unexpected boundary callback: ") + name);
    }

} // namespace

hypereb::kernels::RegularPhysics hypereb::kernels::regular_physics()
{
    hypereb::kernels::RegularPhysics result{
        runtime.gamma,
        runtime.cv,
        runtime.cp,
        runtime.prandtl,
        runtime.sutherland_coefficient,
        runtime.sutherland_temperature,
        runtime.use_constant_viscosity,
        runtime.constant_mu,
        runtime.constant_xi,
        runtime.constant_lambda};
    result.do_diffusion = runtime.do_diffusion;
    result.viscosity_power_law = runtime.viscosity_power_law;
    result.viscosity_reference_mu = runtime.viscosity_reference_mu;
    result.viscosity_reference_temperature = runtime.viscosity_reference_temperature;
    result.viscosity_power_exponent = runtime.viscosity_power_exponent;
    result.eb_wall_diffusion = runtime.eb_wall_diffusion;
    result.eb_thermal_boundary = runtime.eb_thermal_boundary;
    result.eb_wall_temperature = runtime.eb_wall_temperature;
    result.eb_noslip_segment = runtime.eb_noslip_segment;
    result.eb_noslip_s_lo = runtime.eb_noslip_s_lo;
    result.eb_noslip_s_hi = runtime.eb_noslip_s_hi;
    for (int direction = 0; direction < 3; ++direction)
    {
        result.eb_noslip_origin[direction] = runtime.eb_noslip_origin[direction];
        result.eb_noslip_tangent[direction] = runtime.eb_noslip_tangent[direction];
    }
    result.wall_thermal_boundary = runtime.wall_thermal_boundary;
    result.wall_temperature = runtime.wall_temperature;
    for (int direction = 0; direction < 3; ++direction)
    {
        result.problem_low[direction] = runtime.problem_low[direction];
        result.wall_translation[direction] = runtime.wall_translation[direction];
        result.wall_acceleration[direction] = runtime.wall_acceleration[direction];
    }
    return result;
}

bool hypereb::kernels::admissibility_diagnostics_enabled() noexcept
{
    return runtime.admissibility_diagnostics;
}


extern "C"
{

    void amrex_probinit(const int *, const int *, const int *,
                        const amrex_real *problem_low, const amrex_real *problem_high)
    {
        for (int direction = 0; direction < 3; ++direction)
        {
            runtime.problem_low[direction] = problem_low[direction];
            runtime.problem_high[direction] = problem_high[direction];
        }

        read_problem_parameters();
        configure_problem_states();
    }

    void hypereb_init_fort(const int *physical_low, const int *physical_high,
                       int, int, int, int, int, int, int,
                       const amrex_real *problem_low, const amrex_real *problem_high)
    {
        for (int direction = 0; direction < 3; ++direction)
        {
            runtime.physical_low[direction] = physical_low[direction];
            runtime.physical_high[direction] = physical_high[direction];
            runtime.problem_low[direction] = problem_low[direction];
            runtime.problem_high[direction] = problem_high[direction];
        }

        amrex::ParmParse physics("physics");
        physics.query("gamma", runtime.gamma);
        physics.query("eos_mu", runtime.molecular_weight);
        physics.query("Pr", runtime.prandtl);
        physics.query("C_S", runtime.sutherland_coefficient);
        physics.query("T_S", runtime.sutherland_temperature);
        physics.query("R_specific", runtime.specific_gas_constant);
        int use_constant_viscosity = 0;
        int do_diffusion = 1;
        int eb_wall_diffusion = 1;
        physics.query("use_const_visc", use_constant_viscosity);
        std::string variable_viscosity_model = "sutherland";
        physics.query("variable_viscosity_model", variable_viscosity_model);
        if (variable_viscosity_model != "sutherland"
            && variable_viscosity_model != "power_law") {
            amrex::Abort("physics.variable_viscosity_model must be sutherland or power_law");
        }
        runtime.viscosity_power_law = variable_viscosity_model == "power_law";
        if (runtime.viscosity_power_law) {
            if (use_constant_viscosity != 0) {
                amrex::Abort("power_law requires physics.use_const_visc=0");
            }
            // No guessed exponent or benchmark calibration is supplied.
            physics.get("viscosity_reference_mu", runtime.viscosity_reference_mu);
            physics.get("viscosity_reference_temperature", runtime.viscosity_reference_temperature);
            physics.get("viscosity_power_exponent", runtime.viscosity_power_exponent);
            if (!(runtime.viscosity_reference_mu > 0.0)
                || !std::isfinite(runtime.viscosity_reference_mu)
                || !(runtime.viscosity_reference_temperature > 0.0)
                || !std::isfinite(runtime.viscosity_reference_temperature)
                || !std::isfinite(runtime.viscosity_power_exponent)
                || !(runtime.prandtl > 0.0) || !std::isfinite(runtime.prandtl)) {
                amrex::Abort("Invalid power-law transport parameters or Prandtl number");
            }
            amrex::Print() << "hyperEB variable viscosity: power_law mu_ref="
                << runtime.viscosity_reference_mu << " T_ref="
                << runtime.viscosity_reference_temperature << " exponent="
                << runtime.viscosity_power_exponent << " Pr=" << runtime.prandtl << '\n';
        }
        physics.query("do_diffusion", do_diffusion);
        physics.query("eb_wall_diffusion", eb_wall_diffusion);
        std::string eb_thermal_bc = "adiabatic";
        std::string wall_thermal_bc = "adiabatic";
        physics.query("eb_thermal_bc", eb_thermal_bc);
        physics.query("eb_wall_temperature", runtime.eb_wall_temperature);
        int eb_noslip_segment = 0;
        physics.query("eb_noslip_segment", eb_noslip_segment);
        runtime.eb_noslip_segment = eb_noslip_segment != 0;
        amrex::Vector<amrex::Real> segment_values(3);
        for (int direction = 0; direction < 3; ++direction)
        {
            segment_values[direction] = runtime.eb_noslip_origin[direction];
        }
        if (physics.queryarr("eb_noslip_origin", segment_values, 0, 3))
        {
            std::copy(segment_values.begin(), segment_values.end(),
                      runtime.eb_noslip_origin.begin());
        }
        for (int direction = 0; direction < 3; ++direction)
        {
            segment_values[direction] = runtime.eb_noslip_tangent[direction];
        }
        if (physics.queryarr("eb_noslip_tangent", segment_values, 0, 3))
        {
            std::copy(segment_values.begin(), segment_values.end(),
                      runtime.eb_noslip_tangent.begin());
        }
        physics.query("eb_noslip_s_lo", runtime.eb_noslip_s_lo);
        physics.query("eb_noslip_s_hi", runtime.eb_noslip_s_hi);
        physics.query("wall_thermal_bc", wall_thermal_bc);
        physics.query("wall_temperature", runtime.wall_temperature);
        physics.query("const_visc_mu", runtime.constant_mu);
        physics.query("const_visc_ki", runtime.constant_xi);
        physics.query("const_lambda", runtime.constant_lambda);
        runtime.use_constant_viscosity = use_constant_viscosity != 0;
        runtime.do_diffusion = do_diffusion != 0;
        runtime.eb_wall_diffusion = eb_wall_diffusion != 0;
        if (runtime.eb_noslip_segment)
        {
            amrex::Real tangent_norm = 0.0;
            for (const amrex::Real value : runtime.eb_noslip_tangent)
            {
                tangent_norm += value*value;
            }
            tangent_norm = std::sqrt(tangent_norm);
            if (!(tangent_norm > 0.0) || !std::isfinite(tangent_norm)
                || !(runtime.eb_noslip_s_hi > runtime.eb_noslip_s_lo))
            {
                amrex::Abort("Invalid physics.eb_noslip segment definition");
            }
            for (amrex::Real& value : runtime.eb_noslip_tangent)
            {
                value /= tangent_norm;
            }
        }
        if (!(runtime.gamma > 1.0) || !std::isfinite(runtime.gamma))
        {
            amrex::Abort("physics.gamma must be finite and greater than one");
        }
        if (runtime.specific_gas_constant < 0.0 && (!(runtime.molecular_weight > 0.0) || runtime.molecular_weight > 1.0e4 || !std::isfinite(runtime.molecular_weight)))
        {
            amrex::Abort(
                "physics.eos_mu must be a molecular weight in kg/kmol; "
                "use physics.R_specific for an explicit gas constant");
        }
        if (runtime.specific_gas_constant == 0.0 || !std::isfinite(runtime.specific_gas_constant))
        {
            amrex::Abort("physics.R_specific must be finite and positive when specified");
        }
        if (eb_thermal_bc == "adiabatic")
        {
            runtime.eb_thermal_boundary = hypereb::kernels::EBThermalBoundary::adiabatic;
        }
        else if (eb_thermal_bc == "isothermal")
        {
            runtime.eb_thermal_boundary = hypereb::kernels::EBThermalBoundary::isothermal;
        }
        else
        {
            amrex::Abort("physics.eb_thermal_bc must be adiabatic or isothermal");
        }
        if (wall_thermal_bc == "adiabatic")
        {
            runtime.wall_thermal_boundary = hypereb::kernels::EBThermalBoundary::adiabatic;
        }
        else if (wall_thermal_bc == "isothermal")
        {
            runtime.wall_thermal_boundary = hypereb::kernels::EBThermalBoundary::isothermal;
        }
        else
        {
            amrex::Abort("physics.wall_thermal_bc must be adiabatic or isothermal");
        }
        if (runtime.eb_thermal_boundary == hypereb::kernels::EBThermalBoundary::isothermal && (!(runtime.eb_wall_temperature > 0.0) || !std::isfinite(runtime.eb_wall_temperature)))
        {
            amrex::Abort("physics.eb_wall_temperature must be finite and positive");
        }
        if (runtime.wall_thermal_boundary == hypereb::kernels::EBThermalBoundary::isothermal && (!(runtime.wall_temperature > 0.0) || !std::isfinite(runtime.wall_temperature)))
        {
            amrex::Abort("physics.wall_temperature must be finite and positive");
        }
        if ((runtime.eb_thermal_boundary == hypereb::kernels::EBThermalBoundary::isothermal || runtime.wall_thermal_boundary == hypereb::kernels::EBThermalBoundary::isothermal) && !runtime.do_diffusion)
        {
            amrex::Abort("isothermal walls require physics.do_diffusion=1");
        }
        if (!runtime.do_diffusion)
        {
            runtime.use_constant_viscosity = true;
            runtime.constant_mu = 0.0;
            runtime.constant_xi = 0.0;
            runtime.constant_lambda = 0.0;
        }
        update_thermodynamics();

        int admissibility_diagnostics = 0;
        amrex::ParmParse diagnostics("diagnostics");
        diagnostics.query("admissibility", admissibility_diagnostics);
        runtime.admissibility_diagnostics = admissibility_diagnostics != 0;

        read_problem_parameters();

        const auto motion = hypereb::motion::read_translation();
        std::copy(motion.velocity.begin(), motion.velocity.end(),
                  runtime.wall_translation.begin());
        std::copy(motion.acceleration.begin(), motion.acceleration.end(),
                  runtime.wall_acceleration.begin());
        configure_problem_states();
    }

    void hypereb_initdata(const int *, const amrex_real *, const int *low, const int *high,
                      amrex_real *data, const int *data_low, const int *data_high,
                      const amrex_real *spacing, const amrex_real *problem_low)
    {
        FabView<amrex::Real> state(data, data_low, data_high);
        auto copy_state = [&](int i, int j, int k,
                              const std::array<amrex::Real, state_components> &value)
        {
            for (int component = 0; component < state_components; ++component)
            {
                state(i, j, k, component) = value[component];
            }
        };
        auto set_cell = [&](int i, int j, int k, amrex::Real rho,
                            amrex::Real u, amrex::Real v, amrex::Real w,
                            amrex::Real pressure, amrex::Real cell_temperature,
                            amrex::Real cell_volume_fraction = 1.0)
        {
            std::array<amrex::Real, state_components> value{};
            set_conservative_state(
                value, rho, u, v, w, pressure, cell_temperature);
            value[volume_fraction] = cell_volume_fraction;
            copy_state(i, j, k, value);
        };

        for (int k = low[2] - 2; k <= high[2] + 2; ++k)
        {
            for (int j = low[1] - 2; j <= high[1] + 2; ++j)
            {
                for (int i = low[0] - 2; i <= high[0] + 2; ++i)
                {
                    const amrex::Real x = problem_low[0] + (i + 0.5) * spacing[0];

                    switch (runtime.problem_case)
                    {
                    case ProblemCase::regular_uniform:
                        copy_state(i, j, k, runtime.inflow);
                        break;
                    case ProblemCase::smooth_expanding_piston:
                    {
                        // Arienti et al. (2003), prescribed expanding-piston
                        // accuracy test.  The gas initially occupies
                        // 0.5 <= x <= 1, the left wall recedes with velocity
                        // v_l < 0, and the fixed right wall is at x=1.
                        // The exact entropy invariant is p/rho^gamma = 1.
                        constexpr amrex::Real pi =
                            3.14159265358979323846;
                        const amrex::Real rho =
                            1.0 + 0.2 * std::cos(2.0 * pi * (x - 0.5));
                        const amrex::Real u =
                            2.0 * (1.0 - x) * runtime.piston_velocity;
                        const amrex::Real pressure =
                            std::pow(rho, runtime.gamma);
                        set_cell(i, j, k, rho, u, 0.0, 0.0, pressure,
                                 pressure / (runtime.gas_constant * rho));
                        break;
                    }
                    }
                }
            }
        }
    }

    void hypereb_hypfill(amrex_real *data, const int *data_low, const int *data_high,
                     const int *domain_low, const int *domain_high,
                     const amrex_real *spacing, const amrex_real *,
                     const amrex_real *, const int *bc)
    {
        for (int component = 0; component < state_components; ++component)
        {
            fill_component(data, data_low, data_high, domain_low, domain_high,
                           bc + 6 * component, component, runtime.inflow[component],
                           spacing);
        }

        if (runtime.wall_thermal_boundary != hypereb::kernels::EBThermalBoundary::isothermal)
        {
            return;
        }

        FabView<amrex::Real> state(data, data_low, data_high);
        const int *density_bc = bc + 6 * density;
        for (int k = data_low[2]; k <= data_high[2]; ++k)
        {
            for (int j = data_low[1]; j <= data_high[1]; ++j)
            {
                for (int i = data_low[0]; i <= data_high[0]; ++i)
                {
                    if (i >= domain_low[0] && i <= domain_high[0] && j >= domain_low[1] && j <= domain_high[1] && k >= domain_low[2] && k <= domain_high[2])
                    {
                        continue;
                    }
                    const int indices[3] = {i, j, k};
                    int source[3] = {i, j, k};
                    bool external = false;
                    bool no_slip = false;
                    for (int direction = 0; direction < 3; ++direction)
                    {
                        const bool low_side = indices[direction] < domain_low[direction];
                        const bool high_side = indices[direction] > domain_high[direction];
                        const int selected_bc = low_side    ? density_bc[direction]
                                                : high_side ? density_bc[direction + 3]
                                                            : amrex::BCType::int_dir;
                        external = external || selected_bc == amrex::BCType::ext_dir;
                        no_slip = no_slip || ((low_side || high_side) && physical_no_slip_wall(direction, low_side));
                        const auto reflected = reflected_index(
                            indices[direction], domain_low[direction], domain_high[direction],
                            density_bc[direction], density_bc[direction + 3]);
                        source[direction] = reflected.index;
                        external = external || reflected.external;
                    }
                    if (external || !no_slip)
                    {
                        continue;
                    }
                    const amrex::Real rho = state(i, j, k, density);
                    const amrex::Real source_temperature =
                        state(source[0], source[1], source[2], temperature);
                    const amrex::Real ghost_temperature =
                        amrex::Real(2.0) * runtime.wall_temperature - source_temperature;
                    if (!(rho > 0.0) || !std::isfinite(rho) ||
                        !(source_temperature > 0.0) || !std::isfinite(source_temperature) ||
                        !(ghost_temperature > 0.0) || !std::isfinite(ghost_temperature))
                    {
                        amrex::Abort("Invalid density or temperature in isothermal wall ghost");
                    }
                    const amrex::Real internal = rho * runtime.cv * ghost_temperature;
                    const amrex::Real kinetic = amrex::Real(0.5) / rho * (state(i, j, k, xmom) * state(i, j, k, xmom) + state(i, j, k, ymom) * state(i, j, k, ymom) + state(i, j, k, zmom) * state(i, j, k, zmom));
                    if (!(internal > 0.0) || !std::isfinite(internal) ||
                        !std::isfinite(internal + kinetic))
                    {
                        amrex::Abort("Invalid energy in isothermal wall ghost");
                    }
                    state(i, j, k, internal_energy) = internal;
                    state(i, j, k, total_energy) = internal + kinetic;
                    state(i, j, k, temperature) = ghost_temperature;
                }
            }
        }
    }

    void hypereb_denfill(amrex_real *data, const int *data_low, const int *data_high,
                     const int *domain_low, const int *domain_high,
                     const amrex_real *spacing, const amrex_real *,
                     const amrex_real *, const int *bc)
    {
        fill_component(data, data_low, data_high, domain_low, domain_high,
                       bc, 0, runtime.inflow[density], spacing);
    }

    void hypereb_nullfill(amrex_real *, const int *, const int *, const int *, const int *,
                      const amrex_real *, const amrex_real *, const amrex_real *, const int *)
    {
        unexpected_boundary_callback("hypereb_nullfill");
    }

    void hypereb_derpres(amrex_real *derived, const int *derived_low, const int *derived_high,
                     const int *, const amrex_real *state_data, const int *state_low,
                     const int *state_high, const int *, const int *low, const int *high,
                     const int *, const int *, const amrex_real *, const amrex_real *,
                     const amrex_real *, const amrex_real *, const int *, const int *, const int *)
    {
        FabView<amrex::Real> pressure(derived, derived_low, derived_high);
        FabView<const amrex::Real> state(state_data, state_low, state_high);
        for (int k = low[2]; k <= high[2]; ++k)
        {
            for (int j = low[1]; j <= high[1]; ++j)
            {
                for (int i = low[0]; i <= high[0]; ++i)
                {
                    pressure(i, j, k) = (runtime.gamma - 1.0) * state(i, j, k);
                }
            }
        }
    }

    void hypereb_dervel(amrex_real *derived, const int *derived_low, const int *derived_high,
                    const int *, const amrex_real *state_data, const int *state_low,
                    const int *state_high, const int *, const int *low, const int *high,
                    const int *, const int *, const amrex_real *, const amrex_real *,
                    const amrex_real *, const amrex_real *, const int *, const int *, const int *)
    {
        FabView<amrex::Real> velocity(derived, derived_low, derived_high);
        FabView<const amrex::Real> state(state_data, state_low, state_high);
        for (int k = low[2]; k <= high[2]; ++k)
        {
            for (int j = low[1]; j <= high[1]; ++j)
            {
                for (int i = low[0]; i <= high[0]; ++i)
                {
                    velocity(i, j, k) = state(i, j, k, 1) / state(i, j, k, 0);
                }
            }
        }
    }

    void hypereb_compute_temperature(const int *low, const int *high, amrex_real *data,
                                 const int *data_low, const int *data_high)
    {
        FabView<amrex::Real> state(data, data_low, data_high);
        for (int k = low[2]; k <= high[2]; ++k)
        {
            for (int j = low[1]; j <= high[1]; ++j)
            {
                for (int i = low[0]; i <= high[0]; ++i)
                {
                    if (state(i, j, k, volume_fraction) <= 0.0)
                    {
                        continue;
                    }
                    const amrex::Real rho = state(i, j, k, density);
                    if (!(rho > 0.0) || !std::isfinite(rho))
                    {
                        amrex::Abort("Invalid density in temperature conversion");
                    }
                    const amrex::Real rho_inverse = 1.0 / rho;
                    const amrex::Real kinetic = 0.5 * rho_inverse * (state(i, j, k, xmom) * state(i, j, k, xmom) + state(i, j, k, ymom) * state(i, j, k, ymom) + state(i, j, k, zmom) * state(i, j, k, zmom));
                    const amrex::Real internal = state(i, j, k, total_energy) - kinetic;
                    if (!(internal > 0.0) || !std::isfinite(internal))
                    {
                        amrex::Abort("Invalid internal energy in temperature conversion");
                    }
                    state(i, j, k, internal_energy) = internal;
                    state(i, j, k, temperature) = rho_inverse * state(i, j, k, internal_energy) / runtime.cv;
                }
            }
        }
    }

    void hypereb_estdt(const int *low, const int *high, const amrex_real *data,
                   const int *data_low, const int *data_high,
                   const amrex_real *spacing, amrex_real *time_step)
    {
        FabView<const amrex::Real> state(data, data_low, data_high);
        Parm transport;
        transport.C_S = runtime.sutherland_coefficient;
        transport.T_S = runtime.sutherland_temperature;
        transport.viscosity_power_law = runtime.viscosity_power_law;
        transport.viscosity_reference_mu = runtime.viscosity_reference_mu;
        transport.viscosity_reference_temperature = runtime.viscosity_reference_temperature;
        transport.viscosity_power_exponent = runtime.viscosity_power_exponent;
        transport.cp = runtime.cp;
        transport.Pr = runtime.prandtl;
        transport.const_visc_mu = runtime.constant_mu;
        transport.const_visc_ki = runtime.constant_xi;
        transport.const_lambda = runtime.constant_lambda;
        for (int k = low[2]; k <= high[2]; ++k)
        {
            for (int j = low[1]; j <= high[1]; ++j)
            {
                for (int i = low[0]; i <= high[0]; ++i)
                {
                    if (state(i, j, k, volume_fraction) <= 0.0)
                    {
                        continue;
                    }
                    const amrex::Real rho = state(i, j, k, density);
                    const amrex::Real pressure =
                        (runtime.gamma - 1.0) * state(i, j, k, internal_energy);
                    if (!(rho > 0.0) || !std::isfinite(rho) ||
                        !(pressure > 0.0) || !std::isfinite(pressure))
                    {
                        amrex::Abort("Invalid density or pressure in timestep estimate");
                    }
                    const amrex::Real rho_inverse = 1.0 / rho;
                    const amrex::Real sound_speed = std::sqrt(
                        runtime.gamma * pressure * rho_inverse);
                    amrex::Real diffusivity = 0.0;
                    if (runtime.do_diffusion)
                    {
                        amrex::Real mu, xi, conductivity;
                        const amrex::Real t = pressure * rho_inverse /
                            ((runtime.gamma - 1.0) * runtime.cv);
                        hypereb_transport_coefficients(t, runtime.use_constant_viscosity,
                                                   transport, mu, xi, conductivity);
                        if (!(mu >= 0.0) || !(xi >= 0.0) || !(conductivity >= 0.0) ||
                            !std::isfinite(mu + xi + conductivity))
                        {
                            amrex::Abort("Invalid transport coefficients in timestep estimate");
                        }
                        diffusivity = rho_inverse * std::max(
                            (4.0 / 3.0) * mu + xi, conductivity / runtime.cv);
                    }
                    // All directional divergences are applied in one RHS. The
                    // explicit advection and diffusion rates therefore add.
                    // This is a regular-volume estimate; EB stabilization is
                    // still responsible for the small-volume restriction.
                    amrex::Real rate = 0.0;
                    for (int d = 0; d < 3; ++d)
                    {
                        rate += (std::abs(state(i, j, k, xmom + d) * rho_inverse)
                                 + sound_speed) / spacing[d]
                              + 2.0 * diffusivity / (spacing[d] * spacing[d]);
                    }
                    if (!(rate > 0.0) || !std::isfinite(rate))
                    {
                        amrex::Abort("Invalid rate in timestep estimate");
                    }
                    *time_step = std::min(*time_step, 1.0 / rate);
                }
            }
        }
    }

} // extern "C"
