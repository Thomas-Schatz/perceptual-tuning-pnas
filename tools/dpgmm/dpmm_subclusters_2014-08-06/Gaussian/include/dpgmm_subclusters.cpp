// =============================================================================
// == dpgmm_subclusters.cpp
// == --------------------------------------------------------------------------
// == The MEX interface file to sample a DP Gaussian Mixture Model with
// == sub-clusters
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
#include "clusters.h"
#include "niw_sampled.h"
#include <omp.h>
#include "stopwatch.h"

#include "reduction_array.h"
#include "reduction_array2.h"

#define NUMARGS 4
#define NUMOUT 5

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
   // Check for proper number of arguments
   if (nrhs != NUMARGS) {
         mexErrMsgTxt("Incorrect number of input arguments required.");
   } else if (nlhs > NUMOUT) {
         mexErrMsgTxt("Too many output arguments expected.");
   }

   checkInput(prhs[0], "double"); //x
   checkInput(prhs[1], VECTOR,  "double"); //phi
   checkInput(prhs[2], STRUCT); // cluster_params
   checkInput(prhs[3], STRUCT); // params

   const int *dims = mxGetDimensions(prhs[0]);
   int N = mxGetNumberOfElements(prhs[0]);
   int D = 1;
   if (mxGetNumberOfDimensions(prhs[0])>1)
      D = dims[0];
   N /= D;
   int D2 = D*D;

   arr(double) x = getArrayInput<double>(prhs[0]);
   arr(double) phi = getArrayInput<double>(prhs[1]);

   const mxArray* cluster_params = prhs[2];


   const mxArray* params = prhs[3];
   double alphah = getInput<double>(getField(params,0,"alpha"));
   double logalphah = log(alphah);
   double kappah = getInput<double>(getField(params,0,"kappa"));
   double nuh = getInput<double>(getField(params,0,"nu"));
   arr(double) thetah = getArrayInput<double>(getField(params,0,"theta"));
   arr(double) deltah = getArrayInput<double>(getField(params,0,"delta"));
   int its_crp = getInput<double>(getField(params,0,"its_crp"));
   int Mproc = getInput<double>(getField(params,0,"Mproc"));
   bool useSuperclusters = getInput<bool>(getField(params,0,"useSuperclusters"));
   bool always_splittable = getInput<bool>(getField(params,0,"always_splittable"));

   niw_sampled hyper(D, kappah, nuh, thetah, deltah);
   hyper.update_posteriors_sample();

   // processor temporary variables
   //Mproc = min(Mproc,omp_get_max_threads());
   omp_set_num_threads(Mproc);

   clusters model(N, D, x, phi, alphah, hyper, Mproc, useSuperclusters, always_splittable);
   model.initialize(cluster_params);

   plhs[1] = mxCreateNumericMatrix(its_crp,1,mxDOUBLE_CLASS,mxREAL);
   plhs[2] = mxCreateNumericMatrix(its_crp,1,mxDOUBLE_CLASS,mxREAL);
   plhs[3] = mxCreateNumericMatrix(its_crp,1,mxDOUBLE_CLASS,mxREAL);
   plhs[4] = mxCreateNumericMatrix(its_crp,1,mxDOUBLE_CLASS,mxREAL);
   arr(double) timeArr = getArrayInput<double>(plhs[1]);
   arr(double) EArr = getArrayInput<double>(plhs[2]);
   arr(double) K = getArrayInput<double>(plhs[3]);
   arr(double) NK = getArrayInput<double>(plhs[4]);


   stopwatch timer;
   for (int it=0; it<its_crp; it++)
   {
      timer.tic();
      model.sample_params();
      model.sample_superclusters();
      model.sample_labels();

      if (rand()%100==0)
         model.propose_random_splits();
      model.propose_random_merges();

      model.propose_splits();
      double time = timer.toc();
      timeArr[it] = time;
      EArr[it] = model.joint_loglikelihood();
      K[it] = model.getK();
      NK[it] = model.getNK();
   }
   
   model.write_output(plhs[0]);
}
