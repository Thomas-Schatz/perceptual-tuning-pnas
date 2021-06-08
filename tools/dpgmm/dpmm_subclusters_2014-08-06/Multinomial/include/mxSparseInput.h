// =============================================================================
// == mxSparseInput.h
// == --------------------------------------------------------------------------
// == A class to do I/O with Matlab's sparse matrices
// == --------------------------------------------------------------------------
// == Copyright 2013. MIT. All Rights Reserved.
// == Written by Jason Chang 11-03-2013
// == --------------------------------------------------------------------------
// == If this code is used, the following should be cited:
// == 
// == [1] J. Chang and J. W. Fisher II, "Parallel Sampling of DP Mixtures
// ==     Models using Sub-Cluster Splits". Neural Information Processing
// ==     Systems (NIPS 2013), Lake Tahoe, NV, USA, Dec 2013.
// =============================================================================

#ifndef _MXSPARSEINPUT_H_INCLUDED_
#define _MXSPARSEINPUT_H_INCLUDED_

#include "matrix.h"
#include "mex.h"
#include <math.h>

class mxSparseInput
{
public:
   int rows;
   int cols;
   long Nzmax;
   long* Ir;
   double* Pr;
   long* Jc;

   // --------------------------------------------------------------------------
   // -- mxSparseInput
   // --   constructor; intializes to all the values given
   // --------------------------------------------------------------------------
   mxSparseInput();
   mxSparseInput(const mxArray *pm);

   void getColumn(int c, long* &col_rows, double* &col_row_vals, long &Ncol) const;
};

// --------------------------------------------------------------------------
// -- mxSparseInput
// --   constructor; intializes to all the values given
// --------------------------------------------------------------------------
inline mxSparseInput::mxSparseInput() : rows(0), cols(0), Nzmax(0), Ir(NULL), Pr(NULL), Jc(NULL)
{
}
inline mxSparseInput::mxSparseInput(const mxArray *pm) :
   rows(mxGetM(pm)), cols(mxGetN(pm)), Nzmax(mxGetNzmax(pm)),
   Ir((long*) mxGetIr(pm)), Pr((double*) mxGetPr(pm)), Jc((long*) mxGetJc(pm))
{
}

// --------------------------------------------------------------------------
// -- mxSparseInput
// --   constructor; intializes to all the values given
// --------------------------------------------------------------------------
inline void mxSparseInput::getColumn(int c, long* &col_rows, double* &col_row_vals, long &Ncol) const
{
   Ncol = Jc[c+1] - Jc[c];
   col_rows = Ir + Jc[c];
   col_row_vals = Pr + Jc[c];
}




#endif
