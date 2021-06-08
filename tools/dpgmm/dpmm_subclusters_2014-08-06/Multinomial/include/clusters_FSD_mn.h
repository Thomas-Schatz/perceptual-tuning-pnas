// =============================================================================
// == clusters_FSD_mn.h
// == --------------------------------------------------------------------------
// == A class for all multinomial clusters with the Finite Symmetric Dirichlet
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

#ifndef _CLUSTERS_FSD_MN_H_INCLUDED_
#define _CLUSTERS_FSD_MN_H_INCLUDED_

#include "matrix.h"
#include "mex.h"
#include <math.h>
#include "array.h"

#include "helperMEX.h"
#include "debugMEX.h"

#include "dir_sampled_hash.h"
#include "dir_sampled_full.h"
#include "linkedList.cpp"
#include "reduction_hash2.h"

#include "mxSparseInput.h"

#include "gsl/gsl_rng.h"
#include "gsl/gsl_randist.h"
#include "gsl/gsl_permutation.h"
#include "gsl/gsl_cdf.h"

#include <boost/unordered_map.hpp>
using boost::unordered_map;

#ifdef USEFULL
   typedef dir_sampled_full dir_sampled;
#else
   typedef dir_sampled_hash dir_sampled;
#endif


class clusters_FSD_mn
{
public:
   int N; // number of data points
   int D; // dimensionality of data
   int D2;
   int K;
   int Nthreads; // number of threads to use in parallel


   mxSparseInput data; // not dynamically allocated, just a pointer to the data
   arr(unsigned int) z;  // not dynamically allocated, just a pointer to the heights

   // cluster parameters
   dir_sampled hyper;
   arr(dir_sampled) params;

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
   // -- clusters_FSD_mn
   // --   constructor; initializes to empty
   // --------------------------------------------------------------------------
   clusters_FSD_mn();
   // --------------------------------------------------------------------------
   // -- clusters_FSD_mn
   // --   copy constructor;
   // --------------------------------------------------------------------------
   clusters_FSD_mn(const clusters_FSD_mn& that);
   // --------------------------------------------------------------------------
   // -- operator=
   // --   assignment operator
   // --------------------------------------------------------------------------
   clusters_FSD_mn& operator=(const clusters_FSD_mn& that);
   // --------------------------------------------------------------------------
   // -- copy
   // --   returns a copy of this
   // --------------------------------------------------------------------------
   void copy(const clusters_FSD_mn& that);
   // --------------------------------------------------------------------------
   // -- clusters_FSD_mn
   // --   constructor; intializes to all the values given
   // --------------------------------------------------------------------------
   clusters_FSD_mn(int _N, int _D, int _K, const mxArray* _data, arr(unsigned int) _z,
            double _alpha, dir_sampled &_hyper, int _Nthreads);

   // --------------------------------------------------------------------------
   // -- ~clusters_FSD_mn
   // --   destructor
   // --------------------------------------------------------------------------
   virtual ~clusters_FSD_mn();

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
   // -- sample_superclusters_FSD_mn
   // --   Samples the supercluster assignments
   // --------------------------------------------------------------------------
   void sample_superclusters_FSD_mn();

   // --------------------------------------------------------------------------
   // -- sample_labels
   // --   Samples the label assignments for each data point
   // --------------------------------------------------------------------------
   void sample_labels();
   
   double joint_loglikelihood();
};


#endif
