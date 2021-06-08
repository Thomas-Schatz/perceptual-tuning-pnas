// =============================================================================
// == dir_sampled_full.cpp
// == --------------------------------------------------------------------------
// == A class for a dirichlet distribution. Uses a full array to maintin counts
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

#ifndef _DIR_SAMPLED_FULL_H_INCLUDED_
#define _DIR_SAMPLED_FULL_H_INCLUDED_

#include "matrix.h"
#include "mex.h"
#include <math.h>

#include "helperMEX.h"
#include "debugMEX.h"

//#include "myfuncs.h"
#include "tableFuncs.h"

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

#include "multinomial.h"
#include <omp.h>
#include <vector>
using std::vector;


class dir_sampled_full
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
   vector<int> indices_counts;
   double data_terms;

   // instantiated multinomial parameters
   multinomial param;

   double (*gammaln)(double);

public:
   // --------------------------------------------------------------------------
   // -- dir_sampled_full
   // --   constructor; initializes to empty
   // --------------------------------------------------------------------------
   dir_sampled_full();
   // --------------------------------------------------------------------------
   // -- dir_sampled_full
   // --   copy constructor;
   // --------------------------------------------------------------------------
   dir_sampled_full(const dir_sampled_full& that);
   // --------------------------------------------------------------------------
   // -- operator=
   // --   assignment operator
   // --------------------------------------------------------------------------
   dir_sampled_full& operator=(const dir_sampled_full& that);
   // --------------------------------------------------------------------------
   // -- copy
   // --   returns a copy of this
   // --------------------------------------------------------------------------
   void copy(const dir_sampled_full& that);
   // --------------------------------------------------------------------------
   // -- dir_sampled_full
   // --   constructor; intializes to all the values given
   // --------------------------------------------------------------------------
   dir_sampled_full(int _Nthreads, int _D, double _alphah);

   // --------------------------------------------------------------------------
   // -- ~dir_sampled_full
   // --   destructor
   // --------------------------------------------------------------------------
   virtual ~dir_sampled_full();

   // --------------------------------------------------------------------------
   // -- ~cleanup
   // --   deletes all the memory allocated by this
   // --------------------------------------------------------------------------
   virtual void cleanup();

public:
   // --------------------------------------------------------------------------
   // -- empty
   // --   Empties out the statistics of the dir_sampled_full (i.e. no data).
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
   void merge_indices_counts(int* indices_counts2);

   void merge_with(dir_sampled_full* other, bool doSample);
   void merge_with(dir_sampled_full* other1, dir_sampled_full* other2, bool doSample);
   void merge_with(dir_sampled_full other, bool doSample);
   void merge_with(dir_sampled_full other1, dir_sampled_full other2, bool doSample);
   void set_stats(int _N, int _totalCounts, double _data_terms, int* _indices_counts);

   double predictive_loglikelihood(arr(long) indices, arr(double) values, long nnz) const;
   double predictive_loglikelihood_marginalized(arr(long) indices, arr(double) values, long nnz);
   double predictive_loglikelihood_marginalized_hyper(arr(long) indices, arr(double) values, long nnz);
   double data_loglikelihood() const;
   double data_loglikelihood_marginalized();
   double data_loglikelihood_marginalized_testmerge(dir_sampled_full *other) const;

   void sample();

   double Jdivergence(const dir_sampled_full &other);

   // --------------------------------------------------------------------------
   // -- add_data
   // --   functions to add an observation to the dir_sampled_full. Updates the sufficient
   // -- statistics, posterior hyperparameters, and predictive parameters
   // --
   // --   parameters:
   // --     - data : the new observed data point of size [1 D]
   // --------------------------------------------------------------------------
   void rem_data(arr(long) indices, arr(double) values, long nnz);
   void add_data_init(arr(long) indices, arr(double) values, long nnz);
   void add_data(arr(long) indices, arr(double) values, long nnz);
};

inline double dir_sampled_full::predictive_loglikelihood(arr(long) indices, arr(double) values, long nnz) const
{
   return param.predictive_loglikelihood(indices, values, nnz);
}

inline double dir_sampled_full::predictive_loglikelihood_marginalized(arr(long) indices, arr(double) values, long nnz)
{
   double log_likelihood = 0;
   double A = alphah*D + totalCounts;

   int C = 0;
   for (long di=0; di<nnz; di++)
   {
      int ind = indices[di];
      double val = values[di];
      C += val;
      double alpha = alphah;
      alpha += indices_counts[ind];
      log_likelihood += gammaln(val + alpha) - gammaln(alpha) - gammalnint(val+1);
   }
   log_likelihood += gammalnint(C+1) + gammaln(A) - gammaln(C+A);
   
   return log_likelihood;
}

inline double dir_sampled_full::predictive_loglikelihood_marginalized_hyper(arr(long) indices, arr(double) values, long nnz)
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
