// =============================================================================
// == clusters_FSD.h
// == --------------------------------------------------------------------------
// == A class for all Gaussian clusters with the Finite Symmetric Dirichlet
// == approximation
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

#ifndef _CLUSTERS_TSB_FSD_H_INCLUDED_
#define _CLUSTERS_TSB_FSD_H_INCLUDED_

#include "matrix.h"
#include "mex.h"
#include <math.h>
#include "array.h"

#include "helperMEX.h"
#include "debugMEX.h"

#include "niw_sampled.h"
#include "linkedList.cpp"

#include "gsl/gsl_rng.h"
#include "gsl/gsl_randist.h"
#include "gsl/gsl_permutation.h"
#include "gsl/gsl_cdf.h"


class clusters_FSD
{
public:
   int N; // number of data points
   int D; // dimensionality of data
   int D2;
   int K;
   int Nthreads; // number of threads to use in parallel

   arr(double) data; // not dynamically allocated, just a pointer to the data
   arr(unsigned int) z;  // not dynamically allocated, just a pointer to the heights

   // cluster parameters
   niw_sampled hyper;
   arr(niw_sampled) params;

   // DP stick-breaking stuff
   double alpha;
   double logalpha;
   arr(double) sticks;

   // random number generation
   gsl_rng *rand_gen;

   // temporary space for parallel processing
   arr(arr(double)) probsArr;
   arr(gsl_rng*) rArr;

public:
   // --------------------------------------------------------------------------
   // -- clusters_FSD
   // --   constructor; initializes to empty
   // --------------------------------------------------------------------------
   clusters_FSD();
   // --------------------------------------------------------------------------
   // -- clusters_FSD
   // --   copy constructor;
   // --------------------------------------------------------------------------
   clusters_FSD(const clusters_FSD& that);
   // --------------------------------------------------------------------------
   // -- operator=
   // --   assignment operator
   // --------------------------------------------------------------------------
   clusters_FSD& operator=(const clusters_FSD& that);
   // --------------------------------------------------------------------------
   // -- copy
   // --   returns a copy of this
   // --------------------------------------------------------------------------
   void copy(const clusters_FSD& that);
   // --------------------------------------------------------------------------
   // -- clusters_FSD
   // --   constructor; intializes to all the values given
   // --------------------------------------------------------------------------
   clusters_FSD(int _N, int _D, int _K, arr(double) _data, arr(unsigned int) _z,
            double _alpha, niw_sampled &_hyper, int _Nthreads);

   // --------------------------------------------------------------------------
   // -- ~clusters_FSD
   // --   destructor
   // --------------------------------------------------------------------------
   virtual ~clusters_FSD();

public:
   // --------------------------------------------------------------------------
   // -- initialize
   // --   populates the initial statistics
   // --------------------------------------------------------------------------
   void initialize();

   // --------------------------------------------------------------------------
   // -- sample_params
   // --   Samples the parameters for each cluster and the mixture weights
   // --------------------------------------------------------------------------
   void sample_params();

   // --------------------------------------------------------------------------
   // -- sample_superclusters_FSD
   // --   Samples the supercluster assignments
   // --------------------------------------------------------------------------
   void sample_superclusters_FSD();

   // --------------------------------------------------------------------------
   // -- sample_labels
   // --   Samples the label assignments for each data point
   // --------------------------------------------------------------------------
   void sample_labels();

   double joint_loglikelihood();
};


#endif
