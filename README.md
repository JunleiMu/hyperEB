# hyperEB

An adaptive Cartesian cut-cell solver for three-dimensional high-speed flows with moving embedded boundaries.

**Version:** 1.0.0

## Overview

hyperEB is developed within the AMReX framework, integrating finite-volume discretization, embedded-boundary geometry, and block-structured adaptive mesh refinement.

The solver advances the compressible Euler and Navier–Stokes equations for a calorically perfect gas on Cartesian meshes. Geometry is represented by a closed STL surface or an analytical moving plane. Prescribed translation supports constant velocity and constant acceleration. Parallel computation is supported through MPI and OpenMP.

## Dependencies

- CMake 3.24 or later
- C++20 compiler
- MPI
- OpenMP
- [AMReX](https://github.com/AMReX-Codes/amrex)

AMReX must be downloaded separately. The tested upstream revision is:

`00279739539ef3b85a9bf8f2d0eb50ecbcefa787`

## Build

With the AMReX source checkout at `../amrex`, build hyperEB using:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DAMREX_SOURCE_DIR=../amrex
cmake --build build --parallel 4
```

The executable is `build/hyperEB` (`hyperEB.exe` on Windows).

Set `AMREX_SOURCE_DIR` to the location of your AMReX source checkout.

For a build without MPI, add `-DAMReX_MPI=OFF` to the CMake configuration command.

On Windows, the specified AMReX revision requires symbolic-link privileges during compilation.

## License

hyperEB is distributed under the GNU General Public License, version 2 (GPL-2.0-only).

AMReX retains its original BSD 3-Clause License.

See [LICENSE](LICENSE) for details.
