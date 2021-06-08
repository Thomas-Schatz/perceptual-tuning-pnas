// =============================================================================
// == niw.h
// == --------------------------------------------------------------------------
// == A class for a Normal Inverse-Wishart distribution that doesn't maintain
// == explicit parameters
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

#ifndef _NIW_H_INCLUDED_
#define _NIW_H_INCLUDED_

#include "matrix.h"
#include "mex.h"
#include <math.h>
#include "array.h"

#include "helperMEX.h"
#include "debugMEX.h"

#include "linear_algebra.h"
#include "myfuncs.h"

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <Eigen/Cholesky>

class niw
{
   gsl_rng *r;
   bool marginalize; // indicates if parameters should be marginalized

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

   // instantiated gaussian parameters
   double predictive_Z;
   arr(double) mean;
   arr(double) prec;
   double logDetCov;
   double logprior;
   double logmu;

public:
   // --------------------------------------------------------------------------
   // -- niw
   // --   constructor; initializes to empty
   // --------------------------------------------------------------------------
   niw();
   // --------------------------------------------------------------------------
   // -- niw
   // --   copy constructor;
   // --------------------------------------------------------------------------
   niw(const niw& that);
   // --------------------------------------------------------------------------
   // -- operator=
   // --   assignment operator
   // --------------------------------------------------------------------------
   niw& operator=(const niw& that);
   // --------------------------------------------------------------------------
   // -- copy
   // --   returns a copy of this
   // --------------------------------------------------------------------------
   void copy(const niw& that);
   // --------------------------------------------------------------------------
   // -- niw
   // --   constructor; intializes to all the values given
   // --------------------------------------------------------------------------
   niw(bool _marginalize, int _D, double _kappah, double _nuh, arr(double) _thetah, arr(double) _Deltah);

   // --------------------------------------------------------------------------
   // -- ~niw
   // --   destructor
   // --------------------------------------------------------------------------
   virtual ~niw();

   // --------------------------------------------------------------------------
   // -- ~cleanup
   // --   deletes all the memory allocated by this
   // --------------------------------------------------------------------------
   virtual void cleanup();

public:
   // --------------------------------------------------------------------------
   // -- empty
   // --   Empties out the statistics of the niw (i.e. no data).
   // --------------------------------------------------------------------------
   void clear();
   void empty();

   bool isempty();
   int getN();
   arr(double) getPrec();
   arr(double) getMean();
   double getLogprior();
   double get_logmu();
   void set_marginalize(bool _marginalize);

   // --------------------------------------------------------------------------
   // -- update_posteriors
   // --   Updates the posterior hyperparameters
   // --------------------------------------------------------------------------
   void update_posteriors();

   // --------------------------------------------------------------------------
   // -- update_predictive
   // --   Updates the parameters for the predictive moment-matched gaussian
   // --------------------------------------------------------------------------
   void update_predictive();

   void update(bool force_predictive=false);

   // --------------------------------------------------------------------------
   // -- add_data
   // --   functions to add an observation to the niw. Updates the sufficient
   // -- statistics, posterior hyperparameters, and predictive parameters
   // --
   // --   parameters:
   // --     - data : the new observed data point of size [1 D]
   // --------------------------------------------------------------------------
   void rem_data(arr(double) data);
   void add_data_init(arr(double) data);
   void add_data(arr(double) data);
   void merge_with(niw* &other);
   void merge_with(niw* &other1, niw* &other2);
   void merge_with(niw &other);
   void merge_with(niw &other1, niw &other2);

   double predictive_loglikelihood(arr(double) data) const;
   double data_loglikelihood();

   void sample();
};


#endif
