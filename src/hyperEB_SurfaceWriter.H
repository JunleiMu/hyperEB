/*
AMReX Copyright (c) 2024, The Regents of the University of California,
through Lawrence Berkeley National Laboratory (subject to receipt of any
required approvals from the U.S. Dept. of Energy).  All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

(1) Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

(2) Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

(3) Neither the name of the University of California, Lawrence Berkeley
National Laboratory, U.S. Dept. of Energy nor the names of its contributors
may be used to endorse or promote products derived from this software
without specific prior written permission.


THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
*/

// Derived from AMReX EBToPVD with application surface-field and AMR-mask extensions.
// See the project-root THIRD_PARTY_NOTICES.md for consolidated attribution.

#ifndef HYPEREB_SURFACE_WRITER_H_
#define HYPEREB_SURFACE_WRITER_H_
#include <AMReX_Config.H>
#include <AMReX_EBCellFlag.H>
#include <AMReX_Array.H>
#include <AMReX_Array4.H>
#include <AMReX_Box.H>

#include <vector>
#include <array>
#include <iosfwd>
#include <string>

/**
 * \file hyperEB_SurfaceWriter.H
 *
 * Tools to export EB surfaces and coverage information in ParaView-friendly formats.
 */

namespace hypereb {

//! Helper that converts EB surfaces into polygonal PVD files.
class SurfaceWriter {

public:
   SurfaceWriter() = default;

   /**
    * \brief Convert EB data in \p bx into polygons stored internally.
    *
    * \param problo Physical low corner of the domain.
    * \param dx     Cell spacing.
    * \param bx     Tile box being exported.
    * \param flag   EB flags.
    * \param bcent  Boundary centroids.
    * \param apx/apy/apz Area fractions per direction.
    */
   void EBToPolygon(const amrex::Real* problo, const amrex::Real* dx,
         const amrex::Box & bx, amrex::Array4<amrex::EBCellFlag const> const& flag,
         amrex::Array4<amrex::Real const> const& bcent,
         amrex::Array4<amrex::Real const> const& apx, amrex::Array4<amrex::Real const> const& apy, amrex::Array4<amrex::Real const> const& apz);

   //! Name the cell arrays attached to subsequently generated polygons.
   void SetCellDataNames(std::vector<std::string> names);

   //! Convert EB data and attach one tuple of cell data to each polygon.
   void EBToPolygon(const amrex::Real* problo, const amrex::Real* dx,
         const amrex::Box & bx, amrex::Array4<amrex::EBCellFlag const> const& flag,
         amrex::Array4<amrex::Real const> const& bcent,
         amrex::Array4<amrex::Real const> const& apx, amrex::Array4<amrex::Real const> const& apy,
         amrex::Array4<amrex::Real const> const& apz, amrex::Array4<amrex::Real const> const& cell_data,
         amrex::Array4<int const> const& mask, bool use_mask);

   //! Write the accumulated EB polygons for rank \p myID into a VTP file.
   void WriteEBVTP(int myID, const std::string& directory = ".",
                   const std::string& stem = "eb") const;
   //! Write a PVD file referencing per-rank outputs.
   static void WritePVTP(int nProcs,
                         const std::vector<std::string>& cell_data_names = {},
                         const std::string& directory = ".",
                         const std::string& stem = "eb") ;

   /**
    * \brief Record which cells are cut/covered to visualize grid coverage.
    */
   void EBGridCoverage(int myID, const amrex::Real* problo, const amrex::Real* dx,
         const amrex::Box &bx, amrex::Array4<amrex::EBCellFlag const> const& flag);

private:
   //! Reorder polygon vertices to satisfy VTK connectivity expectations.
   void reorder_polygon(const std::vector<std::array<amrex::Real,3>>& lpoints, std::array<int,7>& lconnect,
         const std::array<amrex::Real,3>& lnormal);

   // Calculates the Hesse Normal FOrm corresponding to normal and centroid
   static void calc_hesse(amrex::Real& distance, std::array<amrex::Real,3>& n0, amrex::Real& p,
         const std::array<amrex::Real,3>& normal, const std::array<amrex::Real,3>& centroid) ;

   // Fills the alpha vector
   static void calc_alpha(std::array<amrex::Real,12>& alpha,
         const std::array<amrex::Real,3>& n0, amrex::Real p,
         const std::array<std::array<amrex::Real,3>,8>& vertex,
         const amrex::Real* dx) ;

   // Fills count and flags selecting the alphas which are in (0,1)
   static void calc_intersects(int& int_count, std::array<bool,12>& intersects_flags,
         const std::array<amrex::Real,12>& alpha) ;

   void print_points(std::ofstream& myfile) const;
   void print_connectivity(std::ofstream& myfile) const;
   void print_grids(std::ofstream& myfile) const;

   std::vector<std::array<amrex::Real,3>> m_points;
   std::vector<std::array<int,7>> m_connectivity;
   std::vector<std::string> m_cell_data_names;
   std::vector<std::vector<amrex::Real>> m_cell_data;
   int m_grid{0};

};

}

#endif
