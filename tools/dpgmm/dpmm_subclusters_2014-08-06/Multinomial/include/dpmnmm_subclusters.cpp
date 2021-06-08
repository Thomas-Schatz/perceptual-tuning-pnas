// =============================================================================
// == dpmnmm_subclusters.cpp
// == --------------------------------------------------------------------------
// == The MEX interface file to sample a DP Multinomial Mixture Model with
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
#include "gsl/gsl_rng.h"
#include "gsl/gsl_randist.h"
#include "gsl/gsl_permutation.h"
#include "gsl/gsl_cdf.h"
#include "clusters_mn.h"
#include "dir_sampled_hash.h"
#include "dir_sampled_full.h"
#include <omp.h>
#include "stopwatch.h"

#include "reduction_array.h"
#include "reduction_array2.h"

#define NUMARGS 4
#define NUMOUT 5

#ifdef USEFULL
   typedef dir_sampled_full dir_sampled;
#else
   typedef dir_sampled_hash dir_sampled;
#endif

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
   // Check for proper number of arguments
   if (nrhs != NUMARGS) {
         mexErrMsgTxt("Incorrect number of input arguments required.");
   } else if (nlhs > NUMOUT) {
         mexErrMsgTxt("Too many output arguments expected.");
   }

   if (!mxIsSparse(prhs[0]))
      mexErrMsgTxt("Not sparse!\n");
   checkInput(prhs[0], "double"); //x
   checkInput(prhs[1], VECTOR,  "double"); //phi
   checkInput(prhs[2], STRUCT); // cluster_params
   checkInput(prhs[3], STRUCT); // params

   int N = mxGetN(prhs[0]);
   int D = mxGetM(prhs[0]);

   const mxArray* x = prhs[0];
   arr(double) phi = getArrayInput<double>(prhs[1]);
   const mxArray* cluster_params = prhs[2];

   const mxArray* params = prhs[3];
   double alphah = getInput<double>(getField(params,0,"alpha"));
   double diralphah = getInput<double>(getField(params,0,"diralpha"));
   double logalphah = log(alphah);
   int its_crp = getInput<double>(getField(params,0,"its_crp"));
   int Mproc = getInput<double>(getField(params,0,"Mproc"));
   bool useSuperclusters = getInput<bool>(getField(params,0,"useSuperclusters"));
   bool always_splittable = getInput<bool>(getField(params,0,"always_splittable"));

   dir_sampled hyper(Mproc, D, diralphah);
   hyper.update_posteriors();

   // processor temporary variables
   //Mproc = min(Mproc,omp_get_max_threads());
   omp_set_num_threads(Mproc);
   clusters_mn model(N, D, x, phi, alphah, hyper, Mproc, useSuperclusters, always_splittable);
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
