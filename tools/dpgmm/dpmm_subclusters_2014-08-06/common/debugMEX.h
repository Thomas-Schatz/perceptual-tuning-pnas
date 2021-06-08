// =============================================================================
// == debugMEX.h
// == --------------------------------------------------------------------------
// == Functions to help in MEX interfaces
// == --------------------------------------------------------------------------
// == Copyright 2011. MIT. All Rights Reserved.
// == Written by Jason Chang 06-13-2011
// =============================================================================

#ifndef DEBUGMEX
#define DEBUGMEX

#include "helperMEX.h"
#include "matrix.h"
#include "array.h"

#ifdef DEBUG_ARR
   #define arr(type)    array<type>
#else
   #define arr(type)    type*
#endif

inline void figure()
{
   mexEvalString("figure;");
}
inline void figure(int f)
{
   mxArray* tim[1];
   tim[0] = mxCreateNumericMatrix(1,1,mxDOUBLE_CLASS,mxREAL);
   double* timptr = (double *) mxGetData(tim[0]);
   timptr[0] = f;
   mexCallMATLAB(0,NULL,1,tim,"figure");
   mxDestroyArray(tim[0]);
}

inline void drawnow()
{
   mexEvalString("drawnow;");
}
inline void waitforbuttonpress()
{
   drawnow();
   mexCallMATLAB(0,NULL,0,NULL,"waitforbuttonpress");
}
template <typename T>
void imagesc(arr(T) im, int xdim, int ydim)
{
   mxArray* tim[1];
   tim[0] = mxCreateNumericMatrix(xdim, ydim, mxDOUBLE_CLASS, mxREAL);
   double* timptr = (double *) mxGetData(tim[0]);
   for (int i=0; i<xdim*ydim; i++)
      timptr[i] = im[i];
   mexCallMATLAB(0,NULL,1,tim,"imagesc");
   drawnow;
   mxDestroyArray(tim[0]);
}

template <typename T>
void imagesc(arr(T) im, int xdim, int ydim, double minVal, double maxVal)
{
   mxArray* tim[2];
   tim[0] = mxCreateNumericMatrix(xdim, ydim, mxDOUBLE_CLASS, mxREAL);
   double* timptr = (double *) mxGetData(tim[0]);
   for (int i=0; i<xdim*ydim; i++)
      timptr[i] = im[i];

   tim[1] = mxCreateNumericMatrix(1, 2, mxDOUBLE_CLASS, mxREAL);
   timptr = (double *) mxGetData(tim[1]);
   timptr[0] = minVal;
   timptr[1] = maxVal;

   mexCallMATLAB(0,NULL,2,tim,"imagesc");
   drawnow;
   mxDestroyArray(tim[0]);
   mxDestroyArray(tim[1]);
}

inline void title(int num)
{
   mxArray* tnum[1];
   tnum[0] = mxCreateNumericMatrix(1, 1, mxDOUBLE_CLASS, mxREAL);
   double* tnumptr = (double *) mxGetData(tnum[0]);
   tnumptr[0] = num;
   mexCallMATLAB(0,NULL,1,tnum,"title");
   drawnow;
   mxDestroyArray(tnum[0]);
}

#endif
