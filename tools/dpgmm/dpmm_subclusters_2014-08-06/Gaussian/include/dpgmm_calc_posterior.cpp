// =============================================================================
// == calc_sdfIMPORT.cpp
// == --------------------------------------------------------------------------
// == The MEX interface file to calculate a signed distance function
// == --------------------------------------------------------------------------
// == Copyright 2011. MIT. All Rights Reserved.
// == Written by Jason Chang 06-13-2011
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
#include "niw.h"

#define NUMARGS 3
#define NUMOUT 1


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

   niw hyper(true, D, kappah, nuh, thetah, deltah);
   hyper.update();

   arr(niw*) clusters = allocate_memory<niw*>(N, NULL);
   linkedList<int> alive;

   // populate initial statistics for left and right halves
   for (int i=0; i<N; i++)
   {
      int m = z[i];
      if (clusters[m]==NULL)
      {
         alive.addNodeEnd(m);
         clusters[m] = new niw(hyper);
      }
      clusters[m]->add_data_init(x+i*D);
   }

   // populate the statistics
   linkedListNode<int>* node = alive.getFirst();
   while (node!=NULL)
   {
      int m = node->getData();
      clusters[m]->update();
      node = node->getNext();
   }


   node = alive.getFirst();
   double logposterior = gsl_sf_lngamma(alphah) - gsl_sf_lngamma(alphah+N);
   while (node!=NULL)
   {
      int m = node->getData();
      logposterior += logalphah + gsl_sf_lngamma(clusters[m]->getN()) + clusters[m]->data_loglikelihood();
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
