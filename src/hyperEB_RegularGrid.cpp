// Source attribution and license obligations are recorded in THIRD_PARTY_NOTICES.md.
#include <hyperEB_RegularGrid.H>

#include <eb_kernels/hyperEB_diffusion_K.H>
#include <eb_kernels/hyperEB_hydro_K.H>
#include <eb_kernels/hyperEB_index_macros.H>
#include <eb_kernels/hyperEB_parm.H>

#include <AMReX_FArrayBox.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_GpuAtomic.H>
#include <AMReX_GpuContainers.H>

#include <array>

namespace hypereb::kernels
{
    namespace
    {

        Parm make_parameters(const RegularPhysics &physics)
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

        void compute_hyperbolic_fluxes(
            const amrex::Box &box,
            const amrex::Array4<const amrex::Real> &state,
            const amrex::Array4<const amrex::Real> &q,
            const std::array<amrex::Array4<amrex::Real>, 3> &flux,
            const Parm &parameters)
        {

            const amrex::Box x_face_box =
                amrex::surroundingNodes(box, 0);

            const amrex::Box y_face_box =
                amrex::surroundingNodes(box, 1);

#if (AMREX_SPACEDIM == 3)
            const amrex::Box z_face_box =
                amrex::surroundingNodes(box, 2);
#endif

            // A fatal reconstruction state cannot call the host
            // abort routine directly from a GPU kernel.  Record it
            // on the device and terminate after all face kernels
            // have completed.
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

            // --------------------------------------------------
            // X faces
            // --------------------------------------------------
            amrex::ParallelFor(
                x_face_box,
                [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                {
                    const int status = hypereb_muscl_flux_x(
                            i, j, k,
                            flux[0],
                            state,
                            q,
                            parameters,
                            true, true);

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

            // --------------------------------------------------
            // Y faces
            // --------------------------------------------------
            amrex::ParallelFor(
                y_face_box,
                [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                {
                    const int status = hypereb_muscl_flux_y(
                            i, j, k,
                            flux[1],
                            state,
                            q,
                            parameters,
                            true, true);

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

#if (AMREX_SPACEDIM == 3)

            // --------------------------------------------------
            // Z faces
            // --------------------------------------------------
            amrex::ParallelFor(
                z_face_box,
                [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                {
                    const int status = hypereb_muscl_flux_z(
                            i, j, k,
                            flux[2],
                            state,
                            q,
                            parameters,
                            true, true);

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

#endif

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
                    "Fatal state detected during regular MUSCL "
                    "reconstruction. Invalid cell-average or reconstructed state.");
            }

        }

        void compute_diffusive_fluxes(
            const amrex::Box &box,
            const amrex::Array4<const amrex::Real> &q,
            const std::array<amrex::Array4<amrex::Real>, 3> &flux,
            const amrex::GpuArray<amrex::Real, 3> &inverse_spacing,
            const RegularPhysics &physics,
            const Parm &parameters)
        {
            const amrex::Box coefficient_box = amrex::grow(box, 1);
            amrex::FArrayBox coefficient_storage(coefficient_box, NCOEF);
            const auto coefficients = coefficient_storage.array();
            if (physics.use_constant_viscosity)
            {
                amrex::ParallelFor(coefficient_box,
                                   [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                                   {
                                       hypereb_constcoef(i, j, k, coefficients, parameters);
                                   });
            }
            else
            {
                amrex::ParallelFor(coefficient_box,
                                   [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                                   {
                                       hypereb_diffcoef(i, j, k, q, coefficients, parameters);
                                   });
            }

            const auto constant_coefficients = coefficient_storage.const_array();
            const amrex::Box x_face_box = amrex::surroundingNodes(box, 0);
            amrex::ParallelFor(x_face_box,
                               [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                               {
                                   hypereb_diff_x(i, j, k, q, constant_coefficients, inverse_spacing, flux[0]);
                               });
            const amrex::Box y_face_box = amrex::surroundingNodes(box, 1);
            amrex::ParallelFor(y_face_box,
                               [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                               {
                                   hypereb_diff_y(i, j, k, q, constant_coefficients, inverse_spacing, flux[1]);
                               });
            const amrex::Box z_face_box = amrex::surroundingNodes(box, 2);
            amrex::ParallelFor(z_face_box,
                               [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                               {
                                   hypereb_diff_z(i, j, k, q, constant_coefficients, inverse_spacing, flux[2]);
                               });
        }

    } // namespace

    void compute_regular_rhs(
        const amrex::Box &box,
        const amrex::Array4<amrex::Real> &derivative,
        const amrex::Array4<const amrex::Real> &state,
        const std::array<amrex::Array4<amrex::Real>, 3> &output_flux,
        const amrex::GpuArray<amrex::Real, 3> &spacing,
        const RegularPhysics &physics)
    {
        BL_PROFILE("hypereb::kernels::compute_regular_rhs");

        const Parm parameters = make_parameters(physics);
        const amrex::GpuArray<amrex::Real, 3> inverse_spacing{
            1.0 / spacing[0], 1.0 / spacing[1], 1.0 / spacing[2]};

        // Evaluate primitive variables, hyperbolic fluxes,
        // transport coefficients, diffusive fluxes, then the divergence.
        const amrex::Box primitive_box = amrex::grow(box, 2);
        amrex::FArrayBox primitive_storage(primitive_box, NPRIM);
        const auto q = primitive_storage.array();
        amrex::ParallelFor(primitive_box,
                           [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                           {
                               hypereb_ctoprim(i, j, k, state, q, parameters);
                           });

        std::array<amrex::FArrayBox, 3> hyperbolic_storage;
        std::array<amrex::FArrayBox, 3> diffusive_storage;
        std::array<amrex::Array4<amrex::Real>, 3> hyperbolic;
        std::array<amrex::Array4<amrex::Real>, 3> diffusive;
        for (int direction = 0; direction < 3; ++direction)
        {
            const amrex::Box face_box = amrex::surroundingNodes(box, direction);
            hyperbolic_storage[direction].resize(face_box, NEQNS);
            hyperbolic[direction] = hyperbolic_storage[direction].array();
            if (physics.do_diffusion)
            {
                diffusive_storage[direction].resize(face_box, NEQNS);
                diffusive[direction] = diffusive_storage[direction].array();
            }
        }

        compute_hyperbolic_fluxes(
            box, state, primitive_storage.const_array(), hyperbolic,
            parameters);
        if (physics.do_diffusion)
        {
            compute_diffusive_fluxes(
                box, primitive_storage.const_array(), diffusive,
                inverse_spacing, physics, parameters);
        }

        for (int direction = 0; direction < 3; ++direction)
        {
            const amrex::Box face_box = amrex::surroundingNodes(box, direction);
            const auto hyperbolic_flux = hyperbolic_storage[direction].const_array();
            const auto diffusive_flux = physics.do_diffusion
                                            ? diffusive_storage[direction].const_array()
                                            : amrex::Array4<const amrex::Real>{};
            const auto combined_flux = output_flux[direction];
            const bool add_diffusion = physics.do_diffusion;
            amrex::ParallelFor(face_box, NCONS,
                               [=] AMREX_GPU_DEVICE(int i, int j, int k, int component) noexcept
                               {
                                   if (component < NEQNS)
                                   {
                                       combined_flux(i, j, k, component) = hyperbolic_flux(i, j, k, component) + (add_diffusion ? diffusive_flux(i, j, k, component) : 0.0);
                                   }
                                   else
                                   {
                                       combined_flux(i, j, k, component) = 0.0;
                                   }
                               });
        }

        const auto fx = amrex::Array4<const amrex::Real>(output_flux[0]);
        const auto fy = amrex::Array4<const amrex::Real>(output_flux[1]);
        const auto fz = amrex::Array4<const amrex::Real>(output_flux[2]);
        amrex::ParallelFor(box, NEQNS,
                           [=] AMREX_GPU_DEVICE(int i, int j, int k, int component) noexcept
                           {
                               hypereb_flux_to_dudt(i, j, k, component, derivative,
                                                fx, fy, fz, inverse_spacing);
                           });
        amrex::ParallelFor(box, NCONS - NEQNS,
                           [=] AMREX_GPU_DEVICE(int i, int j, int k, int component) noexcept
                           {
                               derivative(i, j, k, NEQNS + component) = 0.0;
                           });


#ifdef AMREX_USE_GPU
        amrex::Gpu::streamSynchronize();
#endif
    }

} // namespace hypereb::kernels
