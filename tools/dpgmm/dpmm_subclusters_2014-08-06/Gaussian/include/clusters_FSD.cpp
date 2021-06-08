// =============================================================================
// == clusters_FSD.cpp
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

#include "clusters_FSD.h"
#include "reduction_array.h"
#include "reduction_array2.h"
#include "linear_algebra.h"
#include "sample_categorical.h"

#ifndef pi
#define pi 3.14159265
#endif

// --------------------------------------------------------------------------
// -- clusters_FSD
// --   constructor; initializes to empty
// --------------------------------------------------------------------------
clusters_FSD::clusters_FSD() :
   N(0), D(0), K(0), params(NULL), sticks(NULL), Nthreads(0), probsArr(NULL),
   rArr(NULL)
{
   D2 = D*D;
   rand_gen = initialize_gsl_rand(mx_rand());
}

// --------------------------------------------------------------------------
// -- clusters_FSD
// --   copy constructor;
// --------------------------------------------------------------------------
clusters_FSD::clusters_FSD(const clusters_FSD& that)
{
   copy(that);
}
// --------------------------------------------------------------------------
// -- operator=
// --   assignment operator
// --------------------------------------------------------------------------
clusters_FSD& clusters_FSD::operator=(const clusters_FSD& that)
{
   if (this != &that)
   {
      if (params!=NULL) deallocate_memory(params);
      if (sticks!=NULL) deallocate_memory(sticks);
      if (Nthreads>0)
      {
         for (int t=0; t<Nthreads; t++)
         {
            if (probsArr[t]!=NULL) deallocate_memory(probsArr[t]);
            if (rArr[t]!=NULL) gsl_rng_free(rArr[t]);
         }
         if (probsArr!=NULL) deallocate_memory(probsArr);
         if (rArr!=NULL) deallocate_memory(rArr);
      }
      gsl_rng_free(rand_gen);
      copy(that);
   }
   return *this;
}
// --------------------------------------------------------------------------
// -- copy
// --   returns a copy of this
// --------------------------------------------------------------------------
void clusters_FSD::copy(const clusters_FSD& that)
{
   N = that.N;
   D = that.D;
   K = that.K;
   D2 = D*D;
   Nthreads = that.Nthreads;

   data = that.data;
   z = that.z;

   alpha = that.alpha;
   logalpha = log(alpha);
   params = allocate_memory<niw_sampled>(N);
   sticks = allocate_memory<double>(N);

   for (int i=0; i<N; i++)
      params[i] = that.params[i];
   copy_memory<double>(sticks, that.sticks, sizeof(double)*N);
   rand_gen = initialize_gsl_rand(mx_rand());

   probsArr = allocate_memory<arr(double)>(Nthreads);
   rArr = allocate_memory<gsl_rng*>(Nthreads);
   for (int t=0; t<Nthreads; t++)
   {
      probsArr[t] = allocate_memory<double>(N);
      rArr[t] = initialize_gsl_rand(mx_rand());
   }
}

// --------------------------------------------------------------------------
// -- clusters_FSD
// --   constructor; intializes to all the values given
// --------------------------------------------------------------------------
clusters_FSD::clusters_FSD(int _N, int _D, int _K, arr(double) _data,
   arr(unsigned int) _z, double _alpha, niw_sampled &_hyper, int _Nthreads) :
   N(_N), D(_D), K(_K), data(_data), z(_z), alpha(_alpha), hyper(_hyper),
   Nthreads(_Nthreads)
{
   D2 = D*D;
   params = allocate_memory<niw_sampled>(K,hyper);
   sticks = allocate_memory<double>(K);
   logalpha = log(alpha);
   rand_gen = initialize_gsl_rand(mx_rand());

   probsArr = allocate_memory<arr(double)>(Nthreads);
   rArr = allocate_memory<gsl_rng*>(Nthreads);
   for (int t=0; t<Nthreads; t++)
   {
      probsArr[t] = allocate_memory<double>(N);
      rArr[t] = initialize_gsl_rand(mx_rand());
   }
}


// --------------------------------------------------------------------------
// -- ~clusters_FSD
// --   destructor
// --------------------------------------------------------------------------
clusters_FSD::~clusters_FSD()
{
   if (params!=NULL) deallocate_memory(params);
   if (sticks!=NULL) deallocate_memory(sticks);
   if (Nthreads>0)
   {
      for (int t=0; t<Nthreads; t++)
      {
         if (probsArr[t]!=NULL) deallocate_memory(probsArr[t]);
         if (rArr[t]!=NULL) gsl_rng_free(rArr[t]);
      }
      if (probsArr!=NULL) deallocate_memory(probsArr);
      if (rArr!=NULL) deallocate_memory(rArr);
   }
   gsl_rng_free(rand_gen);
}


// --------------------------------------------------------------------------
// -- initialize
// --   populates the initial statistics
// --------------------------------------------------------------------------
void clusters_FSD::initialize()
{
   // populate initial statistics for left and right halves
   for (int i=0; i<N; i++)
   {
      int k = z[i];
      params[k].add_data_init(data+i*D);
   }

   for (int k=0; k<K; k++)
      params[k].update_posteriors_sample();
}

// --------------------------------------------------------------------------
// -- sample_params
// --   Samples the parameters for each cluster and the mixture weights
// --------------------------------------------------------------------------
void clusters_FSD::sample_params()
{
   for (int k=0; k<K; k++)
      sticks[k] = params[k].getN();

   // sample the cluster parameters and the gamma distributions
   double total = 0;
   #pragma omp parallel for reduction(+:total)
   for (int k=0; k<K; k++)
   {
      params[k].update_posteriors_sample();
      params[k].empty();
      sticks[k] = gsl_ran_gamma(rArr[omp_get_thread_num()], sticks[k] + alpha/(double)K, 1);
      total += sticks[k];
   }
   // store the log of the stick lenghts
   for (int k=0; k<K; k++)
      sticks[k] = log(sticks[k]) - log(total);
}

// --------------------------------------------------------------------------
// -- sample_labels
// --   Samples the label assignments for each data point
// --------------------------------------------------------------------------
void clusters_FSD::sample_labels()
{
   // temporary reduction array variables
   reduction_array<int> NArr(Nthreads, K, 0);
   reduction_array2<double> tArr(Nthreads, K, D, 0);
   reduction_array2<double> TArr(Nthreads, K, D2, 0);

   // loop through points and sample labels
   #pragma omp parallel for
   for (int i=0; i<N; i++)
   {
      int proc = omp_get_thread_num();
      arr(double) tprobabilities = probsArr[proc];

      // find the distribution over possible ones
      double maxProb = -mxGetInf();
      for (int k2=0; k2<K; k2++)
      {
         // find the probability of the data belonging to this component
         double prob = params[k2].predictive_loglikelihood(data+i*D) + sticks[k2];
         maxProb = max(maxProb, prob);
         tprobabilities[k2] = prob;
      }

      // sample a new cluster label
      double totalProb = total_logcategorical(tprobabilities, K, maxProb);
      int k = sample_logcategorical(tprobabilities, K, maxProb, totalProb, rArr[proc]);
      z[i] = k;

      // accumulate the N
      NArr.reduce_inc(proc, k);

      // accumulate the t
      tArr.reduce_add(proc, k, data+(i*D));
      TArr.reduce_add_outerprod(proc, k, data+i*D);
   }

   // accumulate cluster statistics
   arr(int) fNArr = NArr.final_reduce_add();
   arr(double) ftArr = tArr.final_reduce_add();
   arr(double) fTArr = TArr.final_reduce_add();
   for (int k=0; k<K; k++)
   {
      params[k].set_stats(fNArr[k], ftArr+k*D, fTArr+k*D2);
      params[k].update_posteriors();
   }
}


double clusters_FSD::joint_loglikelihood()
{
   double loglikelihood = gsl_sf_lngamma(alpha) - gsl_sf_lngamma(alpha+N);
   for (int k=0; k<K; k++)
   {
      if (!params[k].isempty())
         loglikelihood += logalpha + gsl_sf_lngamma(params[k].getN()) + params[k].data_loglikelihood_marginalized();
   }
   return loglikelihood;
}