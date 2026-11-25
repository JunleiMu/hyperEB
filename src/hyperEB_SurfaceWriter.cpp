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

#include <hyperEB_SurfaceWriter.H>
#include <AMReX_BLassert.H>
#include <AMReX_Dim3.H>

#include <fstream>
#include <string>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <limits>
#include <filesystem>
#include <utility>

namespace hypereb {

namespace {
amrex::Real dot_product(const std::array<amrex::Real,3>& a, const std::array<amrex::Real,3>& b)
{
   return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

bool intersects(amrex::Real val)
{
   return (val > 0.0 && val < 1.0);
}

}

void SurfaceWriter::EBToPolygon(const amrex::Real* problo, const amrex::Real* dx,
      const amrex::Box & bx, amrex::Array4<amrex::EBCellFlag const> const& flag,
      amrex::Array4<amrex::Real const> const& bcent,
      amrex::Array4<amrex::Real const> const& apx, amrex::Array4<amrex::Real const> const& apy, amrex::Array4<amrex::Real const> const& apz)
{
   EBToPolygon(problo, dx, bx, flag, bcent, apx, apy, apz,
               amrex::Array4<amrex::Real const>{}, amrex::Array4<int const>{}, false);
}

void SurfaceWriter::SetCellDataNames(std::vector<std::string> names)
{
   m_cell_data_names = std::move(names);
   m_cell_data.clear();
   m_cell_data.resize(m_cell_data_names.size());
}

void SurfaceWriter::EBToPolygon(const amrex::Real* problo, const amrex::Real* dx,
      const amrex::Box & bx, amrex::Array4<amrex::EBCellFlag const> const& flag,
      amrex::Array4<amrex::Real const> const& bcent,
      amrex::Array4<amrex::Real const> const& apx, amrex::Array4<amrex::Real const> const& apy,
      amrex::Array4<amrex::Real const> const& apz, amrex::Array4<amrex::Real const> const& cell_data,
      amrex::Array4<int const> const& mask, bool use_mask)
{
   const auto lo = amrex::lbound(bx);
   const auto hi = amrex::ubound(bx);

   for(int k = lo.z; k <= hi.z; ++k) {
      for(int j = lo.y; j <= hi.y; ++j) {
         for(int i = lo.x; i <= hi.x; ++i) {
            // NOTE: do not skip fully enclosed cells (is_covered_cell), as this seems
            // to skip thin walls in the domain:
            // if(.not.is_regular_cell(flag(i,j,k)) .and. &
            // .not.is_covered_cell(flag(i,j,k))) then

            // Instead only look for EBs
            // if( .not.is_regular_cell(flag(i,j,k))) then

            // If covered cells are accounted for in this loop, a FPE arises
            // since apnorm is zero.

            if(flag(i,j,k).isSingleValued() && (!use_mask || mask(i,j,k) != 0)) {
               amrex::Real axm = apx(i  ,j  ,k  );
               amrex::Real axp = apx(i+1,j  ,k  );
               amrex::Real aym = apy(i  ,j  ,k  );
               amrex::Real ayp = apy(i  ,j+1,k  );
               amrex::Real azm = apz(i  ,j  ,k  );
               amrex::Real azp = apz(i  ,j  ,k+1);

               amrex::Real adx = (axm-axp) * dx[1] * dx[2];
               amrex::Real ady = (aym-ayp) * dx[0] * dx[2];
               amrex::Real adz = (azm-azp) * dx[0] * dx[1];

               amrex::Real apnorm = std::sqrt(adx*adx + ady*ady + adz*adz);
               amrex::Real apnorminv = amrex::Real(1.0)/apnorm;

               std::array<amrex::Real,3> normal, centroid;
               std::array<std::array<amrex::Real,3>,8> vertex;

               normal[0] = adx * apnorminv;
               normal[1] = ady * apnorminv;
               normal[2] = adz * apnorminv;

               // convert bcent to global coordinate system centered at plo
               centroid[0] = problo[0] + bcent(i,j,k,0)*dx[0] + (static_cast<amrex::Real>(i) + amrex::Real(0.5))*dx[0];
               centroid[1] = problo[1] + bcent(i,j,k,1)*dx[1] + (static_cast<amrex::Real>(j) + amrex::Real(0.5))*dx[1];
               centroid[2] = problo[2] + bcent(i,j,k,2)*dx[2] + (static_cast<amrex::Real>(k) + amrex::Real(0.5))*dx[2];

               // vertices of bounding cell (i,j,k)
               vertex[0] = {problo[0] + static_cast<amrex::Real>(i  )*dx[0], problo[1] + static_cast<amrex::Real>(j  )*dx[1], problo[2] + static_cast<amrex::Real>(k  )*dx[2]};
               vertex[1] = {problo[0] + static_cast<amrex::Real>(i+1)*dx[0], problo[1] + static_cast<amrex::Real>(j  )*dx[1], problo[2] + static_cast<amrex::Real>(k  )*dx[2]};
               vertex[2] = {problo[0] + static_cast<amrex::Real>(i  )*dx[0], problo[1] + static_cast<amrex::Real>(j+1)*dx[1], problo[2] + static_cast<amrex::Real>(k  )*dx[2]};
               vertex[3] = {problo[0] + static_cast<amrex::Real>(i+1)*dx[0], problo[1] + static_cast<amrex::Real>(j+1)*dx[1], problo[2] + static_cast<amrex::Real>(k  )*dx[2]};
               vertex[4] = {problo[0] + static_cast<amrex::Real>(i  )*dx[0], problo[1] + static_cast<amrex::Real>(j  )*dx[1], problo[2] + static_cast<amrex::Real>(k+1)*dx[2]};
               vertex[5] = {problo[0] + static_cast<amrex::Real>(i+1)*dx[0], problo[1] + static_cast<amrex::Real>(j  )*dx[1], problo[2] + static_cast<amrex::Real>(k+1)*dx[2]};
               vertex[6] = {problo[0] + static_cast<amrex::Real>(i  )*dx[0], problo[1] + static_cast<amrex::Real>(j+1)*dx[1], problo[2] + static_cast<amrex::Real>(k+1)*dx[2]};
               vertex[7] = {problo[0] + static_cast<amrex::Real>(i+1)*dx[0], problo[1] + static_cast<amrex::Real>(j+1)*dx[1], problo[2] + static_cast<amrex::Real>(k+1)*dx[2]};

               // NOTE: this seems to be unnecessary:
               // skip cells that have a tiny intersection and cells that have
               // the centroid on a face/edge/corner
               // if(apnorm > stol    .and. &
               //   vertex(1,1) < centroid(1) .and. centroid(1) < vertex(8,1) .and. &
               //   vertex(1,2) < centroid(2) .and. centroid(2) < vertex(8,2) .and. &
               //   vertex(1,3) < centroid(3) .and. centroid(3) < vertex(8,3)) then

               // Compute EB facets for current cell
               int count;
               amrex::Real distance, p;
               std::array<amrex::Real,3> n0;
               std::array<amrex::Real,12> alpha;
               std::array<bool,12> alpha_intersect;

               calc_hesse(distance, n0, p, normal, centroid);
               calc_alpha(alpha, n0, p, vertex, dx);
               calc_intersects(count, alpha_intersect, alpha);

               // If the number of facet "contained" in does not describe a facet:
               // ... I.e. there's less than 3 (not even a triangle) or more than 6
               // ... (I have no idea what that is):
               //   => Move the centroid a little back and forth along the normal
               //      to see if that makes a difference:

               if((count < 3) || (count > 6)) {
                  int count_d;
                  amrex::Real p_d;
                  std::array<amrex::Real,3> n0_d;
                  std::array<amrex::Real,12> alpha_d;
                  std::array<bool,12> alpha_d_intersect;

                  amrex::Real tol = std::min({dx[0], dx[1], dx[2]})/amrex::Real(100.); // bit of a fudge factor

                  std::array<amrex::Real,3> centroid_d;
                  for(int idim = 0; idim < 3; ++idim) {
                     centroid_d[idim] = centroid[idim] + tol*normal[idim];
                  }

                  calc_hesse(distance, n0_d, p_d, normal, centroid_d);
                  calc_alpha(alpha_d, n0_d, p_d, vertex, dx);
                  calc_intersects(count_d, alpha_d_intersect, alpha_d);
                  if((count_d >= 3) && (count_d <= 6)) {
                     count = count_d;
                     alpha_intersect = alpha_d_intersect;
                  }

                  for(int idim = 0; idim < 3; ++idim) {
                     centroid_d[idim] = centroid[idim] - tol*normal[idim];
                  }

                  calc_hesse(distance, n0_d, p_d, normal, centroid_d);
                  calc_alpha(alpha_d, n0_d, p_d, vertex, dx);
                  calc_intersects(count_d, alpha_d_intersect, alpha_d);
                  if((count_d >= 3) && (count_d <= 6)) {
                     count = count_d;
                     alpha_intersect = alpha_d_intersect;
                  }
               }
               // I know this was a bit of a hack, but it's the only way I prevent
               // missing facets...

               if((count >=3) && (count <=6)) {
                  m_connectivity.push_back({0,0,0,0,0,0,0});
                  for (std::size_t component = 0;
                       component < m_cell_data.size(); ++component) {
                     m_cell_data[component].push_back(
                        cell_data(i,j,k,static_cast<int>(component)));
                  }

                  // calculate intersection points.
                  std::array<std::array<amrex::Real,3>,12> apoints;

                  std::array<amrex::Real,3> ihat = {1, 0, 0};
                  std::array<amrex::Real,3> jhat = {0, 1, 0};
                  std::array<amrex::Real,3> khat = {0, 0, 1};

                  for(int idim = 0; idim < 3; ++idim) {
                     apoints[ 0][idim] = vertex[0][idim] + ihat[idim]*dx[0]*alpha[ 0];
                     apoints[ 1][idim] = vertex[1][idim] + jhat[idim]*dx[1]*alpha[ 1];
                     apoints[ 2][idim] = vertex[2][idim] + ihat[idim]*dx[0]*alpha[ 2];
                     apoints[ 3][idim] = vertex[0][idim] + jhat[idim]*dx[1]*alpha[ 3];
                     apoints[ 4][idim] = vertex[0][idim] + khat[idim]*dx[2]*alpha[ 4];
                     apoints[ 5][idim] = vertex[1][idim] + khat[idim]*dx[2]*alpha[ 5];
                     apoints[ 6][idim] = vertex[3][idim] + khat[idim]*dx[2]*alpha[ 6];
                     apoints[ 7][idim] = vertex[2][idim] + khat[idim]*dx[2]*alpha[ 7];
                     apoints[ 8][idim] = vertex[4][idim] + ihat[idim]*dx[0]*alpha[ 8];
                     apoints[ 9][idim] = vertex[5][idim] + jhat[idim]*dx[1]*alpha[ 9];
                     apoints[10][idim] = vertex[6][idim] + ihat[idim]*dx[0]*alpha[10];
                     apoints[11][idim] = vertex[4][idim] + jhat[idim]*dx[1]*alpha[11];
                  }

                  // store intersections with grid cell alpha in [0,1]
                  for(int lc1 = 0; lc1 < 12; ++lc1) {
                     if(alpha_intersect[lc1]) {
                        m_points.push_back(apoints[lc1]);
                        int lc2 = m_connectivity.back()[0]+1;
                        m_connectivity.back()[0] = lc2;
                        m_connectivity.back()[lc2] = static_cast<int>(m_points.size()-1);
                     }
                  }

                  reorder_polygon(m_points, m_connectivity.back(), n0);
               }
            }
         }
      }
   };
}

void SurfaceWriter::WriteEBVTP(const int myID, const std::string& directory,
                         const std::string& stem) const
{
   std::stringstream ss;
   ss << std::setw(8) << std::setfill('0') << myID;
   std::filesystem::create_directories(directory);
   std::string cID = (std::filesystem::path(directory) /
                      (stem + "_" + ss.str() + ".vtp")).string();

   std::ofstream myfile(cID);
   if(myfile.is_open()) {
      myfile.precision(6);
      myfile << "<?xml version=\"1.0\"?>\n";
      myfile << "<VTKFile type=\"PolyData\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
      myfile << "<PolyData>\n";
      myfile << "<Piece NumberOfPoints=\"" << m_points.size() << "\" NumberOfVerts=\"0\" " // NOLINT
         << "NumberOfLines=\"0\" NumberOfStrips=\"0\" NumberOfPolys=\" " // NOLINT
         << m_connectivity.size() << "\">\n";
      print_points(myfile);
      print_connectivity(myfile);
      myfile << "<PointData></PointData>\n";
      myfile << "<CellData>\n";
      for (std::size_t component = 0; component < m_cell_data.size(); ++component) {
         AMREX_ALWAYS_ASSERT(m_cell_data[component].size() == m_connectivity.size());
         myfile << "<DataArray type=\"Float64\" Name=\""
                << m_cell_data_names[component] << "\" format=\"ascii\">\n";
         for (amrex::Real value : m_cell_data[component]) {
            myfile << std::setprecision(17) << value << '\n';
         }
         myfile << "</DataArray>\n";
      }
      myfile << "</CellData>\n";
      myfile << "</Piece>\n";
      myfile << "</PolyData>\n";
      myfile << "</VTKFile>\n";

      myfile.close();
   }
}

void SurfaceWriter::WritePVTP(const int nProcs,
                        const std::vector<std::string>& cell_data_names,
                        const std::string& directory, const std::string& stem)
{
   std::filesystem::create_directories(directory);
   std::ofstream myfile(std::filesystem::path(directory) / (stem + ".pvtp"));

   if(myfile.is_open()) {
      myfile << "<?xml version=\"1.0\"?>\n";
      myfile << "<VTKFile type=\"PPolyData\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
      myfile << "<PPolyData GhostLevel=\"0\">\n";
      myfile << "<PPointData/>\n";
      myfile << "<PCellData>\n";
      for (const auto& name : cell_data_names) {
         myfile << "<PDataArray type=\"Float64\" Name=\"" << name << "\"/>\n";
      }
      myfile << "</PCellData>\n";
      myfile << "<PPoints>\n";
      myfile << "<PDataArray type=\"Float32\" NumberOfComponents=\"3\"/>\n";
      myfile << "</PPoints>\n";

      for(int lc1 = 0; lc1 < nProcs; ++lc1) {
         std::stringstream ss;
         ss << std::setw(8) << std::setfill('0') << lc1;
         std::string clc1 = stem + "_" + ss.str() + ".vtp";
         myfile << "<Piece Source=\"" << clc1 << "\"/>\n";
      }

      myfile << "</PPolyData>\n";
      myfile << "</VTKFile>\n";
      myfile.close();
   }
}
void SurfaceWriter::reorder_polygon(const std::vector<std::array<amrex::Real,3>>& lpoints,
      std::array<int,7>& lconnect,
      const std::array<amrex::Real,3>& lnormal)
{
   std::array<amrex::Real,3> center;
   center.fill(0.0);

   int longest = 2;
   if(std::abs(lnormal[0]) > std::abs(lnormal[1])) {
      if(std::abs(lnormal[0]) > std::abs(lnormal[2])) {
         longest = 0;
      }
   }
   else {
      if(std::abs(lnormal[1]) > std::abs(lnormal[2])) {
         longest = 1;
      }
   }

   for(int i = 1; i <= lconnect[0]; ++i) {
      center[0] += m_points[lconnect[i]][0];
      center[1] += m_points[lconnect[i]][1];
      center[2] += m_points[lconnect[i]][2];
   }
   center = {center[0]/static_cast<amrex::Real>(lconnect[0]),
             center[1]/static_cast<amrex::Real>(lconnect[0]),
             center[2]/static_cast<amrex::Real>(lconnect[0])};

   int pi, pk;
   amrex::Real ref_angle, angle;
   if(longest == 0)
   {
      for(int i = 1; i <= lconnect[0]-1; ++i) {
         pi = lconnect[i];
         ref_angle = std::atan2(lpoints[pi][2]-center[2], lpoints[pi][1]-center[1]);
         for(int k = i+1; k <= lconnect[0]; ++k) {
            pk = lconnect[k];
            angle = std::atan2(lpoints[pk][2]-center[2], lpoints[pk][1]-center[1]);
            if(angle < ref_angle) {
               ref_angle = angle;
               lconnect[k] = pi;
               lconnect[i] = pk;
               pi = pk;
            }
         }
      }
   }
   else if(longest == 1) {
      for(int i = 1; i <= lconnect[0]-1; ++i) {
         pi = lconnect[i];
         ref_angle = std::atan2(lpoints[pi][0]-center[0], lpoints[pi][2]-center[2]);
         for(int k = i+1; k <= lconnect[0]; ++k) {
            pk = lconnect[k];
            angle = std::atan2(lpoints[pk][0]-center[0], lpoints[pk][2]-center[2]);
            if(angle < ref_angle) {
               ref_angle = angle;
               lconnect[k] = pi;
               lconnect[i] = pk;
               pi = pk;
            }
         }
      }
   }
   else if(longest == 2) {
      for(int i = 1; i <= lconnect[0]-1; ++i) {
         pi = lconnect[i];
         ref_angle = std::atan2(lpoints[pi][1]-center[1], lpoints[pi][0]-center[0]);
         for(int k = i+1; k <= lconnect[0]; ++k) {
            pk = lconnect[k];
            angle = std::atan2(lpoints[pk][1]-center[1], lpoints[pk][0]-center[0]);
            if(angle < ref_angle) {
               ref_angle = angle;
               lconnect[k] = pi;
               lconnect[i] = pk;
               pi = pk;
            }
         }
      }
   }
}

void SurfaceWriter::calc_hesse(amrex::Real& distance, std::array<amrex::Real,3>& n0, amrex::Real& p,
      const std::array<amrex::Real,3>& normal, const std::array<amrex::Real,3>& centroid)
{
   amrex::Real sign_of_dist;

   // General equation of a plane: Ax + By + Cz + D = 0
   // here D := distance
   distance = -dot_product(normal, centroid);

   // Get the sign of the distance
   sign_of_dist = (distance <= amrex::Real(0.0)) ? amrex::Real(1.0) : amrex::Real(-1.0);

   // Get the Hessian form
   amrex::Real fac = sign_of_dist/dot_product(normal, normal);
   for(int idim = 0; idim < 3; ++idim) {
      n0[idim] = fac*normal[idim];
   }
   p = sign_of_dist*(-distance);
}

void SurfaceWriter::calc_alpha(std::array<amrex::Real,12>& alpha,
      const std::array<amrex::Real,3>& n0, amrex::Real p,
      const std::array<std::array<amrex::Real,3>,8>& vertex,
      const amrex::Real* dx)
{
   // default (large) value
   std::ranges::fill(alpha, 10.0);

   // Ray-xAxis intersection
   if(std::abs(n0[0]) > std::numeric_limits<amrex::Real>::epsilon()) {
      alpha[0]  = (p - dot_product(n0,vertex[0]))/(n0[0]*dx[0]);
      alpha[2]  = (p - dot_product(n0,vertex[2]))/(n0[0]*dx[0]);
      alpha[8]  = (p - dot_product(n0,vertex[4]))/(n0[0]*dx[0]);
      alpha[10] = (p - dot_product(n0,vertex[6]))/(n0[0]*dx[0]);
   }

   // Ray-yAxis intersection
   if(std::abs(n0[1]) > std::numeric_limits<amrex::Real>::epsilon()) {
      alpha[1]  = (p - dot_product(n0,vertex[1]))/(n0[1]*dx[1]);
      alpha[3]  = (p - dot_product(n0,vertex[0]))/(n0[1]*dx[1]);
      alpha[9]  = (p - dot_product(n0,vertex[5]))/(n0[1]*dx[1]);
      alpha[11] = (p - dot_product(n0,vertex[4]))/(n0[1]*dx[1]);
   }

   // Ray-zAxis intersection
   if(std::abs(n0[2]) > std::numeric_limits<amrex::Real>::epsilon()) {
      alpha[4] = (p - dot_product(n0,vertex[0]))/(n0[2]*dx[2]);
      alpha[5] = (p - dot_product(n0,vertex[1]))/(n0[2]*dx[2]);
      alpha[6] = (p - dot_product(n0,vertex[3]))/(n0[2]*dx[2]);
      alpha[7] = (p - dot_product(n0,vertex[2]))/(n0[2]*dx[2]);
   }
}

void SurfaceWriter::calc_intersects(int& int_count, std::array<bool,12>& intersects_flags,
      const std::array<amrex::Real,12>& alpha)
{
   int_count = 0;
   std::ranges::fill(intersects_flags, false);

   for(int lc1 = 0; lc1 < 12; ++lc1) {
      if(intersects(alpha[lc1])) {
         ++int_count;
         intersects_flags[lc1] = true;
      }
   }
}

void SurfaceWriter::print_points(std::ofstream& myfile) const
{
   myfile << "<Points>\n";
   myfile << "<DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">\n";

   for(const auto & m_point : m_points) {
      myfile << std::fixed << std::scientific
         << m_point[0] << " " << m_point[1] << " " << m_point[2] << "\n";
   }

   myfile << "</DataArray>\n";
   myfile << "</Points>\n";
}

void SurfaceWriter::print_connectivity(std::ofstream& myfile) const
{
   myfile << "<Polys>\n";
   myfile << "<DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n";
   for(const auto & lc1 : m_connectivity) {
      for(int lc2 = 1; lc2 <= lc1[0]; ++lc2) {
         myfile << " " << lc1[lc2];
      }
      myfile << "\n";
   }
   myfile << "</DataArray>\n";

   myfile << "<DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n";
   int lc2 = 0;
   for(const auto & lc1 : m_connectivity) {
      lc2 = lc2 + lc1[0];
      myfile << " " << lc2;
   }
   myfile << "\n";
   myfile << "</DataArray>\n";

   myfile << "</Polys>\n";
}

void SurfaceWriter::EBGridCoverage(const int myID, const amrex::Real* problo, const amrex::Real* dx,
      const amrex::Box &bx, amrex::Array4<amrex::EBCellFlag const> const& flag)
{
   int lc1 = 0;

   const auto lo = amrex::lbound(bx);
   const auto hi = amrex::ubound(bx);

   std::array<int,3> nodes = {hi.x-lo.x + 1, hi.y-lo.y + 1, hi.z-lo.z + 1};
   std::array<int,3> low = {lo.x, lo.y, lo.z};

   for(int k = lo.z; k <= hi.z; ++k) {
      for(int j = lo.y; j <= hi.y; ++j) {
         for(int i = lo.x; i <= hi.x; ++i)
         {
            if(flag(i,j,k).isSingleValued()) {
               lc1 = lc1 + 1;
            }
         }
      }
   };

   ++m_grid;
   if(lc1 == 0) { return; }

   std::stringstream ss;
   ss << std::setw(4) << std::setfill('0') << myID;
   std::string cID = ss.str();

   ss.str("");
   ss.clear();
   ss << std::setw(4) << std::setfill('0') << m_grid;
   std::string cgrid = ss.str();
   std::string fname = "eb_grid_" + cID + "_" + cgrid + ".vtr";

   std::ofstream myfile(fname);
   if(myfile.is_open()) {
      myfile.precision(6);
      myfile << "<?xml version=\"1.0\"?>\n";
      myfile << "<VTKFile type=\"RectilinearGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
      myfile << "<RectilinearGrid WholeExtent=\" 0 "
         << nodes[0] << " 0 " << nodes[1] << " 0 " << nodes[2] << "\">\n";
      myfile << "<Piece Extent=\" 0 "
         << nodes[0] << " 0 " << nodes[1] << " 0 " << nodes[2] << "\">\n";
      myfile << "<Coordinates>\n";

      for(int idim = 0; idim < 3; ++idim) {
         std::vector<amrex::Real> lines(nodes[idim]+1);
         amrex::Real grid_start = problo[idim] + static_cast<amrex::Real>(low[idim])*dx[idim];
         for(int llc = 0; llc <= nodes[idim]; ++llc) {
            lines[llc] = grid_start + static_cast<amrex::Real>(llc)*dx[idim];
         }

         myfile << "<DataArray type=\"Float32\" format=\"ascii\" RangeMin=\"" // NOLINT
            << std::fixed
            << lines[0] << "\" RangeMax=\"" << lines[nodes[idim]] << "\">\n";

         for(auto line : lines) {
            myfile << " " << line;
         }
         myfile << "\n";

         myfile << "</DataArray>\n";
      }

      myfile << "</Coordinates>\n";
      myfile << "</Piece>\n";
      myfile << "</RectilinearGrid>\n";
      myfile << "</VTKFile>\n";
   }

   myfile.close();
}

}
