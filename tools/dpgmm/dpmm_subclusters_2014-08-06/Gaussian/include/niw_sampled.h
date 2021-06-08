// =============================================================================
// == niw_sampled.h
// == --------------------------------------------------------------------------
// == A class for a Normal Inverse-Wishart distribution
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

#ifndef _NIW_SAMPLED_H_INCLUDED_
#define _NIW_SAMPLED_H_INCLUDED_

#include "matrix.h"
#include "mex.h"
#include <math.h>
#include "array.h"

#include "helperMEX.h"
#include "debugMEX.h"
#include "normal.h"

#include "linear_algebra.h"
#include "myfuncs.h"

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <Eigen/Cholesky>

#include <vector>
using std::vector;

class niw_sampled
{
   gsl_rng *r;

   // prior hyperparameters
   double kappah;
   double nuh;
   arr(double) thetah;
   arr(double) Deltah;
   int D;
   int D2;

   // sufficient statistics of the observed data
   arr(double) t;
   arr(double) T;
   int N;

   // posterior hyperparameters
   double kappa;
   double nu;
   arr(double) theta;
   arr(double) Delta;

   arr(double) tempVec;
   arr(double) tempMtx;

   // instantiated gaussian parameters
   normal param;
   double logpmu_prior, logpmu_posterior;

public:
   // --------------------------------------------------------------------------
   // -- niw_sampled
   // --   constructor; initializes to empty
   // --------------------------------------------------------------------------
   niw_sampled();
   // --------------------------------------------------------------------------
   // -- niw_sampled
   // --   copy constructor;
   // --------------------------------------------------------------------------
   niw_sampled(const niw_sampled& that);
   // --------------------------------------------------------------------------
   // -- operator=
   // --   assignment operator
   // --------------------------------------------------------------------------
   niw_sampled& operator=(const niw_sampled& that);
   // --------------------------------------------------------------------------
   // -- copy
   // --   returns a copy of this
   // --------------------------------------------------------------------------
   void copy(const niw_sampled& that);
   // --------------------------------------------------------------------------
   // -- niw_sampled
   // --   constructor; intializes to all the values given
   // --------------------------------------------------------------------------
   niw_sampled(int _D, double _kappah, double _nuh, arr(double) _thetah, arr(double) _Deltah);

   // --------------------------------------------------------------------------
   // -- ~niw_sampled
   // --   destructor
   // --------------------------------------------------------------------------
   virtual ~niw_sampled();

   // --------------------------------------------------------------------------
   // -- ~cleanup
   // --   deletes all the memory allocated by this
   // --------------------------------------------------------------------------
   virtual void cleanup();

public:
   // --------------------------------------------------------------------------
   // -- empty
   // --   Empties out the statistics of the niw_sampled (i.e. no data).
   // --------------------------------------------------------------------------
   void clear();
   void empty();

   bool isempty() const;
   int getN() const;
   int getD() const;
   arr(double) get_mean() const;
   arr(double) get_cov() const;
   arr(double) get_prec() const;
   gsl_rng* get_r();

   void set_normal(normal &other);
   void set_normal(arr(double) _mean, arr(double) _cov);
   normal* get_normal();

   // --------------------------------------------------------------------------
   // -- update_posteriors
   // --   Updates the posterior hyperparameters
   // --------------------------------------------------------------------------
   void update_posteriors();
   void update_posteriors_sample();

   // --------------------------------------------------------------------------
   // -- add_data
   // --   functions to add an observation to the niw_sampled. Updates the sufficient
   // -- statistics, posterior hyperparameters, and predictive parameters
   // --
   // --   parameters:
   // --     - data : the new observed data point of size [1 D]
   // --------------------------------------------------------------------------
   void add_data_init(arr(double) data);
   void add_data(arr(double) data);
   void merge_with(niw_sampled* &other, bool doSample);
   void merge_with(niw_sampled* &other1, niw_sampled* &other2, bool doSample);
   void merge_with(niw_sampled &other, bool doSample);
   void merge_with(niw_sampled &other1, niw_sampled &other2, bool doSample);
   void set_stats(int _N, arr(double) _t, arr(double) _T);

   double Jdivergence(const niw_sampled &other);

   double predictive_loglikelihood(arr(double) data) const;
   double predictive_loglikelihood_mode(arr(double) data) const;
   double data_loglikelihood() const;
   double data_loglikelihood_marginalized() const;

   double data_loglikelihood_marginalized_testmerge(niw_sampled *other) const;
   void sample(normal &_param);
   void sample_scale(normal &_param);
   void sample();
   void find_mode();
   double logmu_posterior(const normal &_param) const;
   double logmu_posterior() const;
   double logmu_prior(const normal &_param) const;
   double logmu_prior() const;
};

inline double niw_sampled::predictive_loglikelihood(arr(double) data) const
{
   return param.predictive_loglikelihood(data);
}


#endif
