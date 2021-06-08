// =============================================================================
// == dpgmm_FSD.cpp
// == --------------------------------------------------------------------------
// == The MEX interface file to sample a DP Gaussian Mixture Model with
// == finite symmetric Dirichlet approximation
// == --------------------------------------------------------------------------
// == Copyright 2013. MIT. All Rights Reserved.
// == Written by Jason Chang 11-03-2013
// == --------------------------------------------------------------------------
// == If this code is used, the following should be cited:
// == 
// == [1] J. Chang and J. W. Fisher II, "Parallel Sampling of DP Mixtures
// ==     Models using Sub-Cluster Splits". Neural Information Processing
// ==     Systems (NIPS 2013), Lake Tahoe, NV, USA, Dec 2013.
// == [2] H. Ishwaran and M. Zarepour. Exact and Approximate
// ==     Sum-Representations for the Dirichlet Process. Canadian Journal of
// ==     Statistics, 30:269-283, 2002.
// =============================================================================

#include "mex.h"
#include <iostream>
#include <math.h>
#include <algorithm>
#include <time.h>
#include "debugMEX.h"
#include "helperMEX.h"
#include "matrix.h"
#include "linkedList.cpp"
#include "linear_algebra.h"
#include "gsl/gsl_rng.h"
#include "gsl/gsl_randist.h"
#include "gsl/gsl_permutation.h"
#include "gsl/gsl_cdf.h"
#include "myfuncs.h"
#include "clusters_FSD.h"
#include "niw_sampled.h"
#include <omp.h>
#include "stopwatch.h"

#include "reduction_array.h"
#include "reduction_array2.h"

#define NUMARGS 3
#define NUMOUT 2

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
   // Check for proper number of arguments
   if (nrhs != NUMARGS) {
         mexErrMsgTxt("Incorrect number of input arguments required.");
   } else if (nlhs > NUMOUT) {
         mexErrMsgTxt("Too many output arguments expected.");
   }

   checkInput(prhs[0], "double"); //x
   checkInput(prhs[1], VECTOR,  "uint32"); //phi
   checkInput(prhs[2], STRUCT); // params

   const int *dims = mxGetDimensions(prhs[0]);
   int N = mxGetNumberOfElements(prhs[0]);
   int D = 1;
   if (mxGetNumberOfDimensions(prhs[0])>1)
      D = dims[0];
   N /= D;
   int D2 = D*D;

   arr(double) x = getArrayInput<double>(prhs[0]);
   arr(unsigned int) z = getArrayInput<unsigned int>(prhs[1]);

   const mxArray* params = prhs[2];
   double alphah = getInput<double>(getField(params,0,"alpha"));
   double logalphah = log(alphah);
   double kappah = getInput<double>(getField(params,0,"kappa"));
   double nuh = getInput<double>(getField(params,0,"nu"));
   arr(double) thetah = getArrayInput<double>(getField(params,0,"theta"));
   arr(double) deltah = getArrayInput<double>(getField(params,0,"delta"));
   int its_crp = getInput<double>(getField(params,0,"its_crp"));
   int its_ms = getInput<double>(getField(params,0,"its_ms"));
   int its = max(its_crp, its_ms);
   int Mproc = getInput<double>(getField(params,0,"Mproc"));
   int K = getInput<double>(getField(params,0,"K"));

   niw_sampled hyper(D, kappah, nuh, thetah, deltah);
   hyper.update_posteriors_sample();

   // processor temporary variables
   //Mproc = min(Mproc,omp_get_max_threads());
   omp_set_num_threads(Mproc);

   clusters_FSD model(N, D, K, x, z, alphah, hyper, Mproc);
   model.initialize();


   plhs[0] = mxCreateNumericMatrix(its_crp,1,mxDOUBLE_CLASS,mxREAL);
   plhs[1] = mxCreateNumericMatrix(its_crp,1,mxDOUBLE_CLASS,mxREAL);
   arr(double) timeArr = getArrayInput<double>(plhs[0]);
   arr(double) EArr = getArrayInput<double>(plhs[1]);
   stopwatch timer;
   for (int it=0; it<its_crp; it++)
   {
      timer.tic();
      model.sample_params();
      model.sample_labels();
      double time = timer.toc();
      timeArr[it] = time;
      EArr[it] = model.joint_loglikelihood();
   }
}