// =============================================================================
// == dpmnmm_calc_posterior.cpp
// == --------------------------------------------------------------------------
// == The MEX interface file to calculate the posterior log likelihood of a DP
// == Multinomial Mixture Model
// == --------------------------------------------------------------------------
// == Copyright 2013. MIT. All Rights Reserved.
// == Written by Jason Chang 11-03-2013
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
#include "myfuncs.h"
#include "dir_sampled_hash.h"
#include "dir_sampled_full.h"
#include "mxSparseInput.h"
#include "cluster_single_mn.h"

#define NUMARGS 3
#define NUMOUT 1

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

   checkInput(prhs[0], "double"); //x
   checkInput(prhs[1], VECTOR,  "uint32"); //z
   checkInput(prhs[2], STRUCT); // params

   int N = mxGetN(prhs[0]);
   int D = mxGetM(prhs[0]);

   mxSparseInput x(prhs[0]);
   arr(unsigned int) z = getArrayInput<unsigned int>(prhs[1]);

   const mxArray* params = prhs[2];
   double alphah = getInput<double>(getField(params,0,"alpha"));
   double diralphah = getInput<double>(getField(params,0,"diralpha"));
   double logalphah = log(alphah);
   int its_crp = getInput<double>(getField(params,0,"its_crp"));

   dir_sampled hyper(1, D, diralphah);
   hyper.update_posteriors();

   arr(dir_sampled*) clusters = allocate_memory<dir_sampled*>(N, NULL);
   linkedList<int> alive;

   // populate initial statistics for left and right halves
   for (int i=0; i<N; i++)
   {
      int m = z[i];
      if (clusters[m]==NULL)
      {
         alive.addNodeEnd(m);
         clusters[m] = new dir_sampled(hyper);
      }
      long* colInds; double* colVals; long nnz;
      x.getColumn(i, colInds, colVals, nnz);
      clusters[m]->add_data_init(colInds, colVals, nnz);
   }

   // populate the statistics
   linkedListNode<int>* node = alive.getFirst();
   while (node!=NULL)
   {
      int m = node->getData();
      clusters[m]->update_posteriors();
      node = node->getNext();
   }


   node = alive.getFirst();
   double logposterior = gsl_sf_lngamma(alphah) - gsl_sf_lngamma(alphah+N);
   while (node!=NULL)
   {
      int m = node->getData();
//      logposterior += clusters[m]->data_loglikelihood() + myloggamma(clusters[m]->getN()) + logalphah;
      logposterior += logalphah + gsl_sf_lngamma(clusters[m]->getN()) + clusters[m]->data_loglikelihood_marginalized();
      node = node->getNext();
   }

   plhs[0] = mxCreateNumericMatrix(1,1,mxDOUBLE_CLASS,mxREAL);
   arr(double) output = getArrayInput<double>(plhs[0]);
   output[0] = logposterior;

   for (int i=0; i<N; i++)
      if (clusters[i])
         delete clusters[i];
   deallocate_memory(clusters);
}
