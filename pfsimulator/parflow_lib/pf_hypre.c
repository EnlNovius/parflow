/*BHEADER**********************************************************************
*
*  Copyright (c) 1995-2024, Lawrence Livermore National Security,
*  LLC. Produced at the Lawrence Livermore National Laboratory. Written
*  by the Parflow Team (see the CONTRIBUTORS file)
*  <parflow@lists.llnl.gov> CODE-OCEC-08-103. All rights reserved.
*
*  This file is part of Parflow. For details, see
*  http://www.llnl.gov/casc/parflow
*
*  Please read the COPYRIGHT file or Our Notice and the LICENSE file
*  for the GNU Lesser General Public License.
*
*  This program is free software; you can redistribute it and/or modify
*  it under the terms of the GNU General Public License (as published
*  by the Free Software Foundation) version 2.1 dated February 1999.
*
*  This program is distributed in the hope that it will be useful, but
*  WITHOUT ANY WARRANTY; without even the IMPLIED WARRANTY OF
*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the terms
*  and conditions of the GNU General Public License for more details.
*
*  You should have received a copy of the GNU Lesser General Public
*  License along with this program; if not, write to the Free Software
*  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
*  USA
**********************************************************************EHEADER*/

#include "parflow.h"

/*--------------------------------------------------------------------------
 * Common functions for HYPRE
 *--------------------------------------------------------------------------*/

#ifdef HAVE_HYPRE
#include "hypre_dependences.h"

/** @brief Box of a subgrid and of the data space that holds it.
 *
 * Fills the corners of the subgrid (ilower, iupper) and of the data space
 * of its ParFlow subvector or submatrix, ghost layer included (vilower,
 * viupper), as hypre's *BoxValues2 functions take them.
 *
 * @param subgrid subgrid to copy
 * @param data_space data space of the subvector or submatrix on subgrid
 * @param ilower lower corner of subgrid (output)
 * @param iupper upper corner of subgrid (output)
 * @param vilower lower corner of data_space (output)
 * @param viupper upper corner of data_space (output)
 */
static void HypreBoxes(Subgrid* subgrid,
                       Subgrid* data_space,
                       int*     ilower,
                       int*     iupper,
                       int*     vilower,
                       int*     viupper)
{
  ilower[0] = SubgridIX(subgrid);
  ilower[1] = SubgridIY(subgrid);
  ilower[2] = SubgridIZ(subgrid);
  iupper[0] = ilower[0] + SubgridNX(subgrid) - 1;
  iupper[1] = ilower[1] + SubgridNY(subgrid) - 1;
  iupper[2] = ilower[2] + SubgridNZ(subgrid) - 1;

  vilower[0] = SubgridIX(data_space);
  vilower[1] = SubgridIY(data_space);
  vilower[2] = SubgridIZ(data_space);
  viupper[0] = vilower[0] + SubgridNX(data_space) - 1;
  viupper[1] = vilower[1] + SubgridNY(data_space) - 1;
  viupper[2] = vilower[2] + SubgridNZ(data_space) - 1;
}

/** @brief Copies the values of a ParFlow array on a subgrid to hypre.
 *
 * One call to hypre per subgrid (hypre >= 2.15), instead of one per cell.
 * The array is laid out on data_space, x fastest, as hypre expects.
 *
 * @param hypre_b hypre vector, or NULL to set a matrix entry
 * @param hypre_mat hypre matrix (when hypre_b is NULL)
 * @param entry hypre stencil entry of the matrix
 * @param subgrid subgrid to copy
 * @param data_space data space of the ParFlow array
 * @param data ParFlow array
 */
static void HypreSetBox(HYPRE_StructVector* hypre_b,
                        HYPRE_StructMatrix* hypre_mat,
                        int                 entry,
                        Subgrid*            subgrid,
                        Subgrid*            data_space,
                        double*             data)
{
  int ilower[3], iupper[3], vilower[3], viupper[3];

  if (SubgridNX(subgrid) * SubgridNY(subgrid) * SubgridNZ(subgrid) == 0)
    return;

  HypreBoxes(subgrid, data_space, ilower, iupper, vilower, viupper);

#if HYPRE_RELEASE_NUMBER >= 21500
  if (hypre_b)
    HYPRE_StructVectorSetBoxValues2(*hypre_b, ilower, iupper,
                                    vilower, viupper, data);
  else
    HYPRE_StructMatrixSetBoxValues2(*hypre_mat, ilower, iupper, 1, &entry,
                                    vilower, viupper, data);
#else
  int nx_v = viupper[0] - vilower[0] + 1;
  int ny_v = viupper[1] - vilower[1] + 1;
  int index[3];

  for (index[2] = ilower[2]; index[2] <= iupper[2]; index[2]++)
    for (index[1] = ilower[1]; index[1] <= iupper[1]; index[1]++)
      for (index[0] = ilower[0]; index[0] <= iupper[0]; index[0]++)
      {
        double value = data[(index[0] - vilower[0])
                            + ((index[1] - vilower[1])
                               + (index[2] - vilower[2]) * ny_v) * nx_v];
        if (hypre_b)
          HYPRE_StructVectorSetValues(*hypre_b, index, value);
        else
          HYPRE_StructMatrixSetValues(*hypre_mat, index, 1, &entry, &value);
      }
#endif
}

void CopyParFlowVectorToHypreVector(Vector *            rhs,
                                    HYPRE_StructVector* hypre_b)
{
  Grid* grid = VectorGrid(rhs);
  int sg;

  ForSubgridI(sg, GridSubgrids(grid))
  {
    Subgrid* subgrid = SubgridArraySubgrid(GridSubgrids(grid), sg);
    Subvector* rhs_sub = VectorSubvector(rhs, sg);

    HypreSetBox(hypre_b, NULL, 0, subgrid, SubvectorDataSpace(rhs_sub),
                SubvectorData(rhs_sub));
  }
  HYPRE_StructVectorAssemble(*hypre_b);
}


void CopyHypreVectorToParflowVector(HYPRE_StructVector* hypre_x,
                                    Vector *            soln)
{
  Grid* grid = VectorGrid(soln);
  int sg;
  int ilower[3], iupper[3], vilower[3], viupper[3];

  ForSubgridI(sg, GridSubgrids(grid))
  {
    Subgrid* subgrid = SubgridArraySubgrid(GridSubgrids(grid), sg);
    Subvector* soln_sub = VectorSubvector(soln, sg);
    double* soln_ptr = SubvectorData(soln_sub);

    if (SubgridNX(subgrid) * SubgridNY(subgrid) * SubgridNZ(subgrid) == 0)
      continue;

    HypreBoxes(subgrid, SubvectorDataSpace(soln_sub),
               ilower, iupper, vilower, viupper);

#if HYPRE_RELEASE_NUMBER >= 21500
    HYPRE_StructVectorGetBoxValues2(*hypre_x, ilower, iupper,
                                    vilower, viupper, soln_ptr);
#else
    int nx_v = viupper[0] - vilower[0] + 1;
    int ny_v = viupper[1] - vilower[1] + 1;
    int index[3];

    for (index[2] = ilower[2]; index[2] <= iupper[2]; index[2]++)
      for (index[1] = ilower[1]; index[1] <= iupper[1]; index[1]++)
        for (index[0] = ilower[0]; index[0] <= iupper[0]; index[0]++)
          HYPRE_StructVectorGetValues(*hypre_x, index,
                                      &soln_ptr[(index[0] - vilower[0])
                                                + ((index[1] - vilower[1])
                                                   + (index[2] - vilower[2]) * ny_v) * nx_v]);
#endif
  }
}


void HypreAssembleGrid(
                       Grid*             pf_grid,
                       HYPRE_StructGrid* hypre_grid,
                       double*           dxyz
                       )
{
  int sg;

  int ilo[3];
  int ihi[3];

  if (pf_grid != NULL)
  {
    /* Free the HYPRE grid */
    if (*hypre_grid)
    {
      HYPRE_StructGridDestroy(*hypre_grid);
      hypre_grid = NULL;
    }

    /* Set the HYPRE grid */
    HYPRE_StructGridCreate(amps_CommWorld, 3, hypre_grid);

    /* Set local grid extents as global grid values */
    ForSubgridI(sg, GridSubgrids(pf_grid))
    {
      Subgrid* subgrid = GridSubgrid(pf_grid, sg);

      ilo[0] = SubgridIX(subgrid);
      ilo[1] = SubgridIY(subgrid);
      ilo[2] = SubgridIZ(subgrid);
      ihi[0] = ilo[0] + SubgridNX(subgrid) - 1;
      ihi[1] = ilo[1] + SubgridNY(subgrid) - 1;
      ihi[2] = ilo[2] + SubgridNZ(subgrid) - 1;

      dxyz[0] = SubgridDX(subgrid);
      dxyz[1] = SubgridDY(subgrid);
      dxyz[2] = SubgridDZ(subgrid);
    }
    HYPRE_StructGridSetExtents(*hypre_grid, ilo, ihi);
    HYPRE_StructGridAssemble(*hypre_grid);
  }
}

void HypreInitialize(Matrix*              pf_Bmat,
                     HYPRE_StructGrid*    hypre_grid,
                     HYPRE_StructStencil* hypre_stencil,
                     HYPRE_StructMatrix*  hypre_mat,
                     HYPRE_StructVector*  hypre_b,
                     HYPRE_StructVector*  hypre_x
                     )
{
  int full_ghosts[6] = { 1, 1, 1, 1, 1, 1 };
  int no_ghosts[6] = { 0, 0, 0, 0, 0, 0 };

  /* For remainder of routine, assume matrix is structured the same for
   * entire nonlinear solve process */
  /* Set stencil parameters */
  int stencil_size = MatrixDataStencilSize(pf_Bmat);

  if (!(*hypre_stencil))
  {
    HYPRE_StructStencilCreate(3, stencil_size, hypre_stencil);

    for (int i = 0; i < stencil_size; i++)
    {
      HYPRE_StructStencilSetElement(*hypre_stencil, i,
                                    &(MatrixDataStencil(pf_Bmat))[i * 3]);
    }
  }

  /* Set up new matrix */
  int symmetric = MatrixSymmetric(pf_Bmat);
  if (!(*hypre_mat))
  {
    HYPRE_StructMatrixCreate(amps_CommWorld, *hypre_grid,
                             *hypre_stencil,
                             hypre_mat);
    HYPRE_StructMatrixSetNumGhost(*hypre_mat, full_ghosts);
    HYPRE_StructMatrixSetSymmetric(*hypre_mat, symmetric);
    HYPRE_StructMatrixInitialize(*hypre_mat);
  }

  /* Set up new right-hand-side vector */
  if (!(*hypre_b))
  {
    HYPRE_StructVectorCreate(amps_CommWorld,
                             *hypre_grid,
                             hypre_b);
    HYPRE_StructVectorSetNumGhost(*hypre_b, no_ghosts);
    HYPRE_StructVectorInitialize(*hypre_b);
  }

  /* Set up new solution vector */
  if (!(*hypre_x))
  {
    HYPRE_StructVectorCreate(amps_CommWorld,
                             *hypre_grid,
                             hypre_x);
    HYPRE_StructVectorSetNumGhost(*hypre_x, full_ghosts);
    HYPRE_StructVectorInitialize(*hypre_x);
  }
  HYPRE_StructVectorSetConstantValues(*hypre_x, 0.0e0);
  HYPRE_StructVectorAssemble(*hypre_x);
}

void HypreAssembleMatrixAsElements(
                                   Matrix *            pf_Bmat,
                                   Matrix *            pf_Cmat,
                                   HYPRE_StructMatrix* hypre_mat,
                                   ProblemData *       problem_data
                                   )
{
  Grid *mat_grid = MatrixGrid(pf_Bmat);
  double *cp, *wp = NULL, *ep, *sop = NULL, *np, *lp = NULL, *up = NULL;
  double *cp_c, *wp_c = NULL, *ep_c = NULL, *sop_c = NULL, *np_c = NULL, *top_dat;
  int sg, e;
  int ix, iy, iz;
  int nx, ny, nz;
  int sy_v;
  int i, j, k, itop, k1;
  int im, io;

  int stencil_indices[7] = { 0, 1, 2, 3, 4, 5, 6 };
  int stencil_indices_symm[4] = { 0, 1, 2, 3 };
  /* ParFlow stencil entry of each hypre stencil entry: all seven, or the
   * upper triangle (center, east, north, upper) of a symmetric matrix */
  int pf_entries[7] = { 0, 1, 2, 3, 4, 5, 6 };
  int pf_entries_symm[4] = { 0, 2, 4, 6 };
  int index[3];

  double coeffs[7];
  double coeffs_symm[4];

  int stencil_size = MatrixDataStencilSize(pf_Bmat);
  int symmetric = MatrixSymmetric(pf_Bmat);

  Vector* top = ProblemDataIndexOfDomainTop(problem_data);

  /* Copy the coefficients of B one subgrid and one stencil entry at a time */
  ForSubgridI(sg, GridSubgrids(mat_grid))
  {
    Subgrid* subgrid = GridSubgrid(mat_grid, sg);
    Submatrix* pfB_sub = MatrixSubmatrix(pf_Bmat, sg);

    for (e = 0; e < stencil_size; e++)
    {
      HypreSetBox(NULL, hypre_mat, e, subgrid, SubmatrixDataSpace(pfB_sub),
                  SubmatrixStencilData(pfB_sub,
                                       symmetric ? pf_entries_symm[e] : pf_entries[e]));
    }
  }

  if (pf_Cmat != NULL)  /* Overland flow is activated. Update preconditioning matrix */
  {
    /* The coefficients of the top cells are replaced by the surface ones */
    ForSubgridI(sg, GridSubgrids(mat_grid))
    {
      Subgrid* subgrid = GridSubgrid(mat_grid, sg);

      Submatrix* pfB_sub = MatrixSubmatrix(pf_Bmat, sg);
      Submatrix* pfC_sub = MatrixSubmatrix(pf_Cmat, sg);

      Subvector* top_sub = VectorSubvector(top, sg);

      cp = SubmatrixStencilData(pfB_sub, 0);
      ep = SubmatrixStencilData(pfB_sub, 2);
      np = SubmatrixStencilData(pfB_sub, 4);
      up = SubmatrixStencilData(pfB_sub, 6);
      if (!symmetric)
      {
        wp = SubmatrixStencilData(pfB_sub, 1);
        sop = SubmatrixStencilData(pfB_sub, 3);
        lp = SubmatrixStencilData(pfB_sub, 5);
      }

      cp_c = SubmatrixStencilData(pfC_sub, 0);
      wp_c = SubmatrixStencilData(pfC_sub, 1);
      ep_c = SubmatrixStencilData(pfC_sub, 2);
      sop_c = SubmatrixStencilData(pfC_sub, 3);
      np_c = SubmatrixStencilData(pfC_sub, 4);
      top_dat = SubvectorData(top_sub);

      ix = SubgridIX(subgrid);
      iy = SubgridIY(subgrid);
      iz = SubgridIZ(subgrid);

      nx = SubgridNX(subgrid);
      ny = SubgridNY(subgrid);
      nz = SubgridNZ(subgrid);

      sy_v = SubvectorNX(top_sub);

      for (j = iy; j < iy + ny; j++)
      {
        for (i = ix; i < ix + nx; i++)
        {
          itop = SubvectorEltIndex(top_sub, i, j, 0);
          k = (int)top_dat[itop];
          /* Inactive column, or top cell on another subgrid */
          if (k < iz || k >= iz + nz)
            continue;

          im = SubmatrixEltIndex(pfB_sub, i, j, k);
          io = SubmatrixEltIndex(pfC_sub, i, j, iz);
          index[0] = i;
          index[1] = j;
          index[2] = k;

          if (symmetric)
          {
            /* update diagonal coeff */
            coeffs_symm[0] = cp_c[io];               //cp[im] is zero
            /* update east coeff */
            coeffs_symm[1] = ep[im];
            /* update north coeff */
            coeffs_symm[2] = np[im];
            /* update upper coeff */
            coeffs_symm[3] = up[im];               // JB keeps upper term on surface. This should be zero

            HYPRE_StructMatrixSetValues(*hypre_mat,
                                        index,
                                        stencil_size,
                                        stencil_indices_symm,
                                        coeffs_symm);
          }
          else
          {
            /* update diagonal coeff */
            coeffs[0] = cp_c[io];               //cp[im] is zero
            /* update west coeff */
            k1 = (int)top_dat[itop - 1];
            if (k1 == k)
              coeffs[1] = wp_c[io];                  //wp[im] is zero
            else
              coeffs[1] = wp[im];
            /* update east coeff */
            k1 = (int)top_dat[itop + 1];
            if (k1 == k)
              coeffs[2] = ep_c[io];                  //ep[im] is zero
            else
              coeffs[2] = ep[im];
            /* update south coeff */
            k1 = (int)top_dat[itop - sy_v];
            if (k1 == k)
              coeffs[3] = sop_c[io];                  //sop[im] is zero
            else
              coeffs[3] = sop[im];
            /* update north coeff */
            k1 = (int)top_dat[itop + sy_v];
            if (k1 == k)
              coeffs[4] = np_c[io];                  //np[im] is zero
            else
              coeffs[4] = np[im];
            /* update upper coeff */
            coeffs[5] = lp[im];               // JB keeps lower term on surface.
            /* update upper coeff */
            coeffs[6] = up[im];               // JB keeps upper term on surface. This should be zero

            HYPRE_StructMatrixSetValues(*hypre_mat,
                                        index,
                                        stencil_size,
                                        stencil_indices, coeffs);
          }
        }
      }
    }   /* End subgrid loop */
  }  /* end if pf_Cmat != NULL */

  HYPRE_StructMatrixAssemble(*hypre_mat);
}

#endif // HAVE_HYPRE
