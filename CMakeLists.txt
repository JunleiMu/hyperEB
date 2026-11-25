cmake_minimum_required(VERSION 3.24)
project(hyperEB VERSION 1.0.0 LANGUAGES C CXX)

# CPU-only, three-dimensional embedded-boundary application.
set(AMReX_SPACEDIM 3 CACHE STRING "Spatial dimension" FORCE)
set(AMReX_GPU_BACKEND NONE CACHE STRING "CPU backend" FORCE)
set(AMReX_EB ON CACHE BOOL "Embedded boundaries" FORCE)
set(AMReX_AMRLEVEL ON CACHE BOOL "AmrLevel support" FORCE)
set(AMReX_MPI ON CACHE BOOL "MPI parallelism")
set(AMReX_OMP ON CACHE BOOL "OpenMP parallelism")
set(AMReX_FORTRAN OFF CACHE BOOL "No Fortran required" FORCE)
set(AMReX_PARTICLES OFF CACHE BOOL "No particles" FORCE)
set(AMReX_LINEAR_SOLVERS OFF CACHE BOOL "No elliptic solvers" FORCE)
set(AMReX_ENABLE_TESTS OFF CACHE BOOL "No dependency tests" FORCE)
set(AMReX_INSTALL OFF CACHE BOOL "Build dependency in place" FORCE)
set(AMReX_PLOTFILE_TOOLS OFF CACHE BOOL "No dependency tools" FORCE)
set(AMReX_PROBINIT ON CACHE BOOL "Problem initialization interface" FORCE)

set(AMREX_SOURCE_DIR "" CACHE PATH "Path to an external AMReX source checkout")
if(NOT EXISTS "${AMREX_SOURCE_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "Set AMREX_SOURCE_DIR to an AMReX source checkout")
endif()
get_filename_component(AMREX_SOURCE_DIR "${AMREX_SOURCE_DIR}" ABSOLUTE)
add_subdirectory("${AMREX_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/amrex")

file(GLOB solver_sources CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
add_executable(hyperEB ${solver_sources})
target_compile_features(hyperEB PRIVATE cxx_std_20)
target_include_directories(hyperEB PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/eb_kernels")
target_compile_definitions(hyperEB PRIVATE
    HYPEREB_CASES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/cases")
target_link_libraries(hyperEB PRIVATE AMReX::amrex_3d)
