// =============================================================================
// == multinomial.h
// == --------------------------------------------------------------------------
// == A class for a multinomial distribution
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

#ifndef _MULTINOMIAL_H_INCLUDED_
#define _MULTINOMIAL_H_INCLUDED_

#include "matrix.h"
#include "mex.h"
#include <math.h>
#include "array.h"
#include "linkedList.cpp"
#include "gsl/gsl_sf_gamma.h"

#include "tableFuncs.h"

#include "helperMEX.h"
#include "debugMEX.h"

#include <boost/unordered_map.hpp>
using boost::unordered_map;

class multinomial
{
   // instantiated gaussian parameters
   int D;
   arr(double) logpi;

public:
   // --------------------------------------------------------------------------
   // -- multinomial
   // --   constructor; initializes to empty
   // --------------------------------------------------------------------------
   multinomial();
   // --------------------------------------------------------------------------
   // -- multinomial
   // --   copy constructor;
   // --------------------------------------------------------------------------
   multinomial(const multinomial& that);
   // --------------------------------------------------------------------------
   // -- operator=
   // --   assignment operator
   // --------------------------------------------------------------------------
   multinomial& operator=(const multinomial& that);
   // --------------------------------------------------------------------------
   // -- copy
   // --   returns a copy of this
   // --------------------------------------------------------------------------
   void copy(const multinomial& that);
   // --------------------------------------------------------------------------
   // -- multinomial
   // --   constructor; intializes to all the values given
   // --------------------------------------------------------------------------
   multinomial(int _D);

   // --------------------------------------------------------------------------
   // -- ~multinomial
   // --   destructor
   // --------------------------------------------------------------------------
   virtual ~multinomial();

   // --------------------------------------------------------------------------
   // -- ~cleanup
   // --   deletes all the memory allocated by this
   // --------------------------------------------------------------------------
   virtual void cleanup();

   void set_values(arr(double) _logpi);
   double predictive_loglikelihood(arr(long) indices, arr(double) values, long nnz) const;
   double data_loglikelihood(const unordered_map<int,int> &indices_counts) const;
   double data_loglikelihood(const int* indices_counts) const;
   double Jdivergence(const multinomial &other);

   friend class dir_sampled_full;
   friend class dir_sampled_hash;
   friend class cluster_sampled;
};

inline void multinomial::set_values(arr(double) _logpi)
{
   memcpy(logpi, _logpi, sizeof(double)*D);
}

inline double multinomial::predictive_loglikelihood(arr(long) indices, arr(double) values, long nnz) const
{
   double loglikelihood = gammalnint(nnz+1);
   for (long i=0; i<nnz; i++)
   {
      long d = indices[i];
      int count = values[i];
      loglikelihood += count*logpi[d] - gammalnint(count+1);
   }
   return loglikelihood;
}


inline double multinomial::data_loglikelihood(const unordered_map<int,int> &indices_counts) const
{
   int N = 0;
   double loglikelihood = 0;
   for (unordered_map<int,int>::const_iterator itr=indices_counts.begin(); itr!=indices_counts.end(); itr++)
   {
      int d = itr->first;
      int count = itr->second;
      loglikelihood += count*logpi[d] - gammalnint(count+1);
      N += count;
   }
   loglikelihood += gammalnint(N+1);
   return loglikelihood;
}

inline double multinomial::data_loglikelihood(const int* indices_counts) const
{
   int N = 0;
   double loglikelihood = 0;
   for (int d=0; d<D; d++)
   {
      int count = indices_counts[d];
      loglikelihood += count*logpi[d] - gammalnint(count+1);
      N += count;
   }
   loglikelihood += gammalnint(N+1);
   return loglikelihood;
}

inline double multinomial::Jdivergence(const multinomial &other)
{
   double J = 0;
   #pragma omp parallel for reduction(+:J)
   for (int d=0; d<D; d++)
   {
      double logpi1 = logpi[d];
      double logpi2 = other.logpi[d];
      J += (exp(logpi1)-exp(logpi2)) * (logpi1-logpi2);
      //double temp = logpi[d] - other.logpi[d];
      //J += temp*exp(temp);
   }
   return J;
}

#endif
