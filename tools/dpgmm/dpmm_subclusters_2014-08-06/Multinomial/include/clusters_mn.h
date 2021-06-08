// =============================================================================
// == clusters_mn.h
// == --------------------------------------------------------------------------
// == A class for all multinomial clusters with sub-clusters
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

#ifndef _CLUSTERS_MN_H_INCLUDED_
#define _CLUSTERS_MN_H_INCLUDED_

#include "matrix.h"
#include "mex.h"
#include <math.h>

#include "helperMEX.h"
#include "debugMEX.h"

#include "dir_sampled_hash.h"
#include "dir_sampled_full.h"
#include "cluster_sampledT.cpp" //templated
#include "linkedList.cpp"
#include "mxSparseInput.h"

#include "gsl/gsl_rng.h"
#include "gsl/gsl_randist.h"
#include "gsl/gsl_permutation.h"
#include "gsl/gsl_cdf.h"

#include <boost/unordered_map.hpp>

#ifdef USEFULL
   typedef dir_sampled_full dir_sampled;
#else
   typedef dir_sampled_hash dir_sampled;
#endif

class clusters_mn
{
public:
   int N; // number of data points
   int D; // dimensionality of data
   int K;
   int Nthreads; // number of threads to use in parallel

   mxSparseInput data; // not dynamically allocated, just a pointer to the data
   double* phi;  // not dynamically allocated, just a pointer to the heights
   vector<bool> randomSplitIndex; // dynamically allocated

   // cluster parameters
   dir_sampled hyper;
   vector< cluster_sampledT<dir_sampled>* > params;
   vector<double> likelihoodOld;
   vector<double> likelihoodDelta;
   vector<bool> splittable;

   // DP stick-breaking stuff
   double alpha;
   double logalpha;
   vector<double> sticks;
   bool always_splittable;

   // misc
   vector<int> k2z;
   vector<int> z2k;
   linkedList<int> alive;
   vector< linkedListNode<int>* > alive_ptrs;

   // random number generation
   gsl_rng *rand_gen;

   // temporary space for parallel processing
   vector< vector<double> > probsArr;
   vector<gsl_rng*> rArr;

   // supercluster stuff
   bool useSuperclusters;
   vector<int> superclusters;
   vector<int> supercluster_labels;
   vector<int> supercluster_labels_count;



public:
   // --------------------------------------------------------------------------
   // -- clusters_mn
   // --   constructor; initializes to empty
   // --------------------------------------------------------------------------
   clusters_mn();
   // --------------------------------------------------------------------------
   // -- clusters_mn
   // --   copy constructor;
   // --------------------------------------------------------------------------
   clusters_mn(const clusters_mn& that);
   // --------------------------------------------------------------------------
   // -- operator=
   // --   assignment operator
   // --------------------------------------------------------------------------
   clusters_mn& operator=(const clusters_mn& that);
   // --------------------------------------------------------------------------
   // -- copy
   // --   returns a copy of this
   // --------------------------------------------------------------------------
   void copy(const clusters_mn& that);
   // --------------------------------------------------------------------------
   // -- clusters_mn
   // --   constructor; intializes to all the values given
   // --------------------------------------------------------------------------
   clusters_mn(int _N, int _D, const mxArray* _data, double* _phi,
            double _alpha, dir_sampled &_hyper, int _Nthreads,
            bool _useSuperclusters_mn, bool _always_splittable);

   // --------------------------------------------------------------------------
   // -- ~clusters_mn
   // --   destructor
   // --------------------------------------------------------------------------
   virtual ~clusters_mn();

public:
   // --------------------------------------------------------------------------
   // -- initialize
   // --   populates the initial statistics
   // --------------------------------------------------------------------------
   void initialize(const mxArray* cluster_params);

   // --------------------------------------------------------------------------
   // -- write_output
   // --   creates and writes the output cluster structure
   // --------------------------------------------------------------------------
   void write_output(mxArray* &plhs);

   // --------------------------------------------------------------------------
   // -- populate_k2z_z2k
   // --   Populates the k and z mappings
   // --------------------------------------------------------------------------
   void populate_k2z_z2k();

   // --------------------------------------------------------------------------
   // -- sample_params
   // --   Samples the parameters for each cluster and the mixture weights
   // --------------------------------------------------------------------------
   void sample_params();

   // --------------------------------------------------------------------------
   // -- sample_superclusters
   // --   Samples the supercluster assignments
   // --------------------------------------------------------------------------
   void sample_superclusters();

   // --------------------------------------------------------------------------
   // -- sample_labels
   // --   Samples the label assignments for each data point
   // --------------------------------------------------------------------------
   void sample_labels();

   // --------------------------------------------------------------------------
   // -- propose_merges
   // --   Samples the label assignments for each data point
   // --------------------------------------------------------------------------
   void propose_merges();
   
   // --------------------------------------------------------------------------
   // -- propose_splits
   // --   Samples the label assignments for each data point
   // --------------------------------------------------------------------------
   void propose_splits();
   
   void propose_random_split_assignments();
   void propose_random_splits();
   void propose_random_merges();

   double joint_loglikelihood();

   int getK() const;
   int getNK() const;
};

inline int clusters_mn::getK() const
{
   return alive.getLength();
}
inline int clusters_mn::getNK() const
{
   int maxNK = 0;
   linkedListNode<int>* node = alive.getFirst();
   while (node!=NULL)
   {
      int m = node->getData();
      maxNK = max(maxNK, params[m]->getN());
      node = node->getNext();
   }
   return maxNK;
}


#endif
