// =============================================================================
// == dir_sampled_hash.h
// == --------------------------------------------------------------------------
// == A class for a dirichlet distribution. Uses a hash table to maintin counts
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

#ifndef _DIR_SAMPLED_HASH_H_INCLUDED_
#define _DIR_SAMPLED_HASH_H_INCLUDED_

#include "matrix.h"
#include "mex.h"
#include <math.h>
#include "array.h"

#include "helperMEX.h"
#include "debugMEX.h"

//#include "myfuncs.h"
#include "tableFuncs.h"

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

#include "multinomial.h"
#include <omp.h>

#include <boost/unordered_map.hpp>
using boost::unordered_map;

class dir_sampled_hash
{
   public:
   int N;
   int totalCounts;
   int Nthreads;
   arr(gsl_rng*) rArr;

   // prior hyperparameters
   double alphah;
   int D;

   // sufficient statistics of the observed data
   // assume sparse
   unordered_map<int, int> indices_counts;
   double data_terms;

   // instantiated multinomial parameters
   multinomial param;


   // temporary space for parallel traversal of unordered_map
   int** temp_counts_itr;
   int* temp_indices_itr;
   int temp_counts_N;
   bool temp_counts_invalid;


   double (*gammaln)(double);

public:
   // --------------------------------------------------------------------------
   // -- dir_sampled_hash
   // --   constructor; initializes to empty
   // --------------------------------------------------------------------------
   dir_sampled_hash();
   // --------------------------------------------------------------------------
   // -- dir_sampled_hash
   // --   copy constructor;
   // --------------------------------------------------------------------------
   dir_sampled_hash(const dir_sampled_hash& that);
   // --------------------------------------------------------------------------
   // -- operator=
   // --   assignment operator
   // --------------------------------------------------------------------------
   dir_sampled_hash& operator=(const dir_sampled_hash& that);
   // --------------------------------------------------------------------------
   // -- copy
   // --   returns a copy of this
   // --------------------------------------------------------------------------
   void copy(const dir_sampled_hash& that);
   // --------------------------------------------------------------------------
   // -- dir_sampled_hash
   // --   constructor; intializes to all the values given
   // --------------------------------------------------------------------------
   dir_sampled_hash(int _Nthreads, int _D, double _alphah);

   // --------------------------------------------------------------------------
   // -- ~dir_sampled_hash
   // --   destructor
   // --------------------------------------------------------------------------
   virtual ~dir_sampled_hash();

   // --------------------------------------------------------------------------
   // -- ~cleanup
   // --   deletes all the memory allocated by this
   // --------------------------------------------------------------------------
   virtual void cleanup();

public:
   // --------------------------------------------------------------------------
   // -- empty
   // --   Empties out the statistics of the dir_sampled_hash (i.e. no data).
   // --------------------------------------------------------------------------
   void clear();
   void empty();

   bool isempty() const;
   int getN() const;
   int getD() const;
   arr(double) get_logpi() const;
   gsl_rng* get_r();

   void set_multinomial(multinomial &other);
   void set_multinomial(arr(double) _logpi);
   multinomial* get_multinomial();

   // --------------------------------------------------------------------------
   // -- update_posteriors
   // --   Updates the posterior hyperparameters
   // --------------------------------------------------------------------------
   void update_posteriors();
   void update_posteriors_sample();

   // helper function to merge to indices_counts
   void merge_indices_counts(unordered_map<int,int> &indices_counts2);

   void merge_with(dir_sampled_hash* other, bool doSample);
   void merge_with(dir_sampled_hash* other1, dir_sampled_hash* other2, bool doSample);
   void merge_with(dir_sampled_hash other, bool doSample);
   void merge_with(dir_sampled_hash other1, dir_sampled_hash other2, bool doSample);
   void set_stats(int _N, int _totalCounts, double _data_terms, unordered_map<int,int> &_indices_counts);

   double predictive_loglikelihood(arr(long) indices, arr(double) values, long nnz) const;
   double predictive_loglikelihood_marginalized(arr(long) indices, arr(double) values, long nnz);
   double predictive_loglikelihood_marginalized_hyper(arr(long) indices, arr(double) values, long nnz);
   double data_loglikelihood() const;
   double data_loglikelihood_marginalized();
   double data_loglikelihood_marginalized_testmerge(dir_sampled_hash *other) const;

   void sample();

   double Jdivergence(const dir_sampled_hash &other);

   // --------------------------------------------------------------------------
   // -- add_data
   // --   functions to add an observation to the dir_sampled_hash. Updates the sufficient
   // -- statistics, posterior hyperparameters, and predictive parameters
   // --
   // --   parameters:
   // --     - data : the new observed data point of size [1 D]
   // --------------------------------------------------------------------------
   void rem_data(arr(long) indices, arr(double) values, long nnz);
   void add_data_init(arr(long) indices, arr(double) values, long nnz);
   void add_data(arr(long) indices, arr(double) values, long nnz);

   void prepare_indices_counts_itr();
};

inline double dir_sampled_hash::predictive_loglikelihood(arr(long) indices, arr(double) values, long nnz) const
{
   return param.predictive_loglikelihood(indices, values, nnz);
}

inline double dir_sampled_hash::predictive_loglikelihood_marginalized(arr(long) indices, arr(double) values, long nnz)
{
   double log_likelihood = 0;
   double A = alphah*D + totalCounts;

   int C = 0;
   for (long di=0; di<nnz; di++)
   {
      long ind = indices[di];
      double val = values[di];
      C += val;
      double alpha = alphah;
      unordered_map<int,int>::const_iterator itr = indices_counts.find(ind);
      if (itr != indices_counts.end()) // is it found?
         alpha += itr->second;
      log_likelihood += gammaln(val + alpha) - gammaln(alpha) - gammalnint(val+1);
   }
   log_likelihood += gammalnint(C+1) + gammaln(A) - gammaln(C+A);
   
   return log_likelihood;
}

inline double dir_sampled_hash::predictive_loglikelihood_marginalized_hyper(arr(long) indices, arr(double) values, long nnz)
{
   double log_likelihood = 0;
   double A = alphah*D;
   int C = 0;
   for (long di=0; di<nnz; di++)
   {
      long ind = indices[di];
      double val = values[di];
      C += val;
      log_likelihood += gammaln(val + alphah) - gammaln(alphah) - gammalnint(val+1);
   }
   log_likelihood += gammalnint(C+1) + gammaln(A) - gammaln(C+A);
   
   return log_likelihood;
}



#endif
