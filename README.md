# hyperEB

An adaptive Cartesian cut-cell solver for three-dimensional high-speed flows
with moving embedded boundaries.

**Version:** 1.0.0 release candidate. Redistribution clearance is pending;
this directory must not yet be published. See [third-party notices](THIRD_PARTY_NOTICES.md).

## Overview

hyperEB is an adaptive Cartesian cut-cell solver for three-dimensional high-speed
flows with moving embedded boundaries. Built upon the AMReX framework, it
integrates finite-volume discretization, embedded-boundary geometry, and
block-structured adaptive mesh refinement.

The solver advances the compressible Euler or Navier-Stokes equations for a
calorically perfect gas on Cartesian meshes. Geometry is represented by a closed
STL surface or an analytical moving plane. Prescribed translation supports
constant velocity and constant acceleration. CPU execution uses MPI and OpenMP.

## Dependencies

Requirements: CMake 3.24+, a C++20 compiler, and compatible MPI and OpenMP
installations. AMReX must be downloaded separately; it is not bundled.
The tested upstream revision is `00279739539ef3b85a9bf8f2d0eb50ecbcefa787`.

With the AMReX source checkout at `../amrex`, build from this directory:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DAMREX_SOURCE_DIR=../amrex
cmake --build build --parallel 4
```

The executable is `build/hyperEB` (`hyperEB.exe` on Windows).
Set `AMREX_SOURCE_DIR` to your checkout location. For a build without MPI,
add `-DAMReX_MPI=OFF`.
On Windows, this AMReX revision requires symbolic-link privileges during its
own build.

## License

The proposed license for authorized hyperEB contributions is the GNU General
Public License version 2 only (SPDX: GPL-2.0-only). Release authorization is not
complete. No project-wide redistribution license is granted by this candidate
README. Existing third-party licenses and notices remain applicable.
See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
