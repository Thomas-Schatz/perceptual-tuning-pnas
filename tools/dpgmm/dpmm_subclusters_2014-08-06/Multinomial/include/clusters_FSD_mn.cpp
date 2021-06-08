// =============================================================================
// == clusters_FSD_mn.cpp
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

#include "clusters_FSD_mn.h"
#include "reduction_array.h"
#include "reduction_array2.h"
#include "sample_categorical.h"

#ifndef pi
#define pi 3.14159265
#endif

// --------------------------------------------------------------------------
// -- clusters_FSD_mn
// --   constructor; initializes to empty
// --------------------------------------------------------------------------
clusters_FSD_mn::clusters_FSD_mn() :
   N(0), D(0), K(0), params(NULL), sticks(NULL), Nthreads(0), probsArr(NULL),
   rArr(NULL)
{
   rand_gen = initialize_gsl_rand(rand());
}

// --------------------------------------------------------------------------
// -- clusters_FSD_mn
// --   copy constructor;
// --------------------------------------------------------------------------
clusters_FSD_mn::clusters_FSD_mn(const clusters_FSD_mn& that)
{
   copy(that);
}
// --------------------------------------------------------------------------
// -- operator=
// --   assignment operator
// --------------------------------------------------------------------------
clusters_FSD_mn& clusters_FSD_mn::operator=(const clusters_FSD_mn& that)
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
void clusters_FSD_mn::copy(const clusters_FSD_mn& that)
{
   N = that.N;
   D = that.D;
   K = that.K;
   Nthreads = that.Nthreads;

   data = that.data;
   z = that.z;

   alpha = that.alpha;
   logalpha = log(alpha);
   params = allocate_memory<dir_sampled>(N);
   sticks = allocate_memory<double>(N);

   for (int i=0; i<N; i++)
      params[i] = that.params[i];
   copy_memory<double>(sticks, that.sticks, sizeof(double)*N);
   rand_gen = initialize_gsl_rand(rand());

   probsArr = allocate_memory<arr(double)>(Nthreads);
   rArr = allocate_memory<gsl_rng*>(Nthreads);
   for (int t=0; t<Nthreads; t++)
   {
      probsArr[t] = allocate_memory<double>(N);
      rArr[t] = initialize_gsl_rand(rand());
   }
}

// --------------------------------------------------------------------------
// -- clusters_FSD_mn
// --   constructor; intializes to all the values given
// --------------------------------------------------------------------------
clusters_FSD_mn::clusters_FSD_mn(int _N, int _D, int _K, const mxArray* _data,
   arr(unsigned int) _z, double _alpha, dir_sampled &_hyper, int _Nthreads) :
   N(_N), D(_D), K(_K), data(_data), z(_z), alpha(_alpha), hyper(_hyper),
   Nthreads(_Nthreads)
{
   params = allocate_memory<dir_sampled>(K,hyper);
   sticks = allocate_memory<double>(K);
   logalpha = log(alpha);
   rand_gen = initialize_gsl_rand(mx_rand());

   probsArr = allocate_memory<arr(double)>(Nthreads);
   rArr = allocate_memory<gsl_rng*>(Nthreads);
   for (int t=0; t<Nthreads; t++)
   {
      probsArr[t] = allocate_memory<double>(N);
      rArr[t] = initialize_gsl_rand(rand());
   }
}


// --------------------------------------------------------------------------
// -- ~clusters_FSD_mn
// --   destructor
// --------------------------------------------------------------------------
clusters_FSD_mn::~clusters_FSD_mn()
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
void clusters_FSD_mn::initialize()
{
   // populate initial statistics for left and right halves
   for (int i=0; i<N; i++)
   {
      int k = z[i];
      long* colInds; double* colVals; long nnz;
      data.getColumn(i, colInds, colVals, nnz);
      params[k].add_data_init(colInds, colVals, nnz);
   }

   for (int k=0; k<K; k++)
      params[k].update_posteriors_sample();
}

// --------------------------------------------------------------------------
// -- sample_params
// --   Samples the parameters for each cluster and the mixture weights
// --------------------------------------------------------------------------
void clusters_FSD_mn::sample_params()
{
   for (int k=0; k<K; k++)
      sticks[k] = params[k].getN();

   // sample the cluster parameters and the gamma distributions
   double total = 0;
   for (int k=0; k<K; k++)
   {
      params[k].update_posteriors_sample();
      params[k].empty();
   }
   #pragma omp parallel for reduction(+:total)
   for (int k=0; k<K; k++)
   {
      sticks[k] = gsl_ran_gamma(rArr[omp_get_thread_num()], sticks[k] + alpha, 1);
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
void clusters_FSD_mn::sample_labels()
{
   // temporary reduction array variables
   reduction_array<int> NArr(Nthreads, K, 0);
   reduction_array<int> totalCountsArr(Nthreads, K, 0);
   reduction_array<double> datatermsArr(Nthreads, K, 0);
#ifdef USEFULL
   reduction_array2<int> countsArr(Nthreads, K, hyper.getD(), 0);
#else
   reduction_hash2<int> countsArr(Nthreads, K);
#endif

   // loop through points and sample labels
   #pragma omp parallel for schedule(dynamic)
   for (int i=0; i<N; i++)
   {
      int proc = omp_get_thread_num();
      arr(double) tprobabilities = probsArr[proc];
      long* colInds; double* colVals; long nnz;
      data.getColumn(i, colInds, colVals, nnz);

      // find the distribution over possible ones
      double maxProb = -mxGetInf();
      for (int k2=0; k2<K; k2++)
      {
         // find the probability of the data belonging to this component
         double prob = params[k2].predictive_loglikelihood(colInds, colVals, nnz) + sticks[k2];
         maxProb = max(maxProb, prob);
         tprobabilities[k2] = prob;
      }

      // sample a new cluster label
      double totalProb = total_logcategorical(tprobabilities, K, maxProb);
      int k = sample_logcategorical(tprobabilities, K, maxProb, totalProb, rArr[proc]);
      z[i] = k;

      // accumulate
      NArr.reduce_inc(proc, k);
      countsArr.reduce_add(proc, k, colInds, colVals, nnz);

      double new_dataterm = 0;
      int Ci = 0;
      for (int di=0; di<nnz; di++)
      {
         Ci += colVals[di];
         new_dataterm -= gammalnint(colVals[di]+1);
      }
      new_dataterm += gammalnint(Ci+1);
      totalCountsArr.reduce_add(proc, k, Ci);
      datatermsArr.reduce_add(proc, k, new_dataterm);
   }

   // accumulate cluster statistics
   arr(int) fNArr = NArr.final_reduce_add();
   arr(double) fdatatermsArr = datatermsArr.final_reduce_add();
   arr(int) ftotalCountsArr = totalCountsArr.final_reduce_add();
#ifdef USEFULL
   arr(int) fcountsArr = countsArr.final_reduce_add();
#else
   unordered_map<int,int>* fcountsArr = countsArr.final_reduce_add();
#endif
   #pragma omp parallel for
   for (int k=0; k<K; k++)
   {
#ifdef USEFULL
      params[k].set_stats(fNArr[k], ftotalCountsArr[k], fdatatermsArr[k], fcountsArr+(k)*hyper.getD());
#else
      params[k].set_stats(fNArr[k], ftotalCountsArr[k], fdatatermsArr[k], fcountsArr[k]);
#endif
   }

}


double clusters_FSD_mn::joint_loglikelihood()
{
   double loglikelihood = gsl_sf_lngamma(alpha) - gsl_sf_lngamma(alpha+N);
   for (int m=0; m<K; m++) 
      if (!params[m].isempty())
         loglikelihood += logalpha + gsl_sf_lngamma(params[m].getN()) + params[m].data_loglikelihood_marginalized();
   return loglikelihood;
}
