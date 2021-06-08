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

#include "dir_sampled_full.h"

#ifndef pi
#define pi 3.14159265
#endif

#ifndef clogpi
#define clogpi 1.144729885849
#endif

// --------------------------------------------------------------------------
// -- dir_sampled_full
// --   constructor; initializes to empty
// --------------------------------------------------------------------------
dir_sampled_full::dir_sampled_full() :
   D(0), Nthreads(0), alphah(0), N(0), totalCounts(0), param(0), rArr(NULL), data_terms(0),
   indices_counts(0)
{
   gammaln = &gsl_sf_lngamma;
}

// --------------------------------------------------------------------------
// -- dir_sampled_full
// --   copy constructor;
// --------------------------------------------------------------------------
dir_sampled_full::dir_sampled_full(const dir_sampled_full& that)
{
   copy(that);
}
// --------------------------------------------------------------------------
// -- operator=
// --   assignment operator
// --------------------------------------------------------------------------
dir_sampled_full& dir_sampled_full::operator=(const dir_sampled_full& that)
{
   if (this != &that)
   {
      cleanup();
      copy(that);
   }
   return *this;
}
// --------------------------------------------------------------------------
// -- copy
// --   returns a copy of this
// --------------------------------------------------------------------------
void dir_sampled_full::copy(const dir_sampled_full& that)
{
   gammaln = that.gammaln;
   N = that.N;
   totalCounts = that.totalCounts;
   Nthreads = that.Nthreads;
   D = that.D;
   alphah = that.alphah;
   indices_counts = that.indices_counts;
   param = that.param;
   data_terms = that.data_terms;

   rArr = allocate_memory<gsl_rng*>(Nthreads);
   for (int t=0; t<Nthreads; t++)
      rArr[t] = initialize_gsl_rand(rand());
}

// --------------------------------------------------------------------------
// -- dir_sampled_full
// --   constructor; intializes to all the values given
// --------------------------------------------------------------------------
dir_sampled_full::dir_sampled_full(int _Nthreads, int _D, double _alphah) :
   N(0), totalCounts(0), Nthreads(_Nthreads), D(_D), alphah(_alphah), param(_D), data_terms(0),
   indices_counts(_D,0)
{
   if (fabs(alphah - (double)((int)alphah)) < 1e-30)
      gammaln = &gammalnint;
   else
      gammaln = &gsl_sf_lngamma;
   rArr = allocate_memory<gsl_rng*>(Nthreads);
   for (int t=0; t<Nthreads; t++)
      rArr[t] = initialize_gsl_rand(rand());
   clear();
}




// --------------------------------------------------------------------------
// -- ~dir_sampled_full
// --   destructor
// --------------------------------------------------------------------------
dir_sampled_full::~dir_sampled_full()
{
   cleanup();
}
// --------------------------------------------------------------------------
// -- ~cleanup
// --   deletes all the memory allocated by this
// --------------------------------------------------------------------------
void dir_sampled_full::cleanup()
{
   if (rArr)
   {
      for (int t=0; t<Nthreads; t++)
         if (rArr[t]!=NULL)
            gsl_rng_free(rArr[t]);
      deallocate_memory(rArr);
   }
}

// --------------------------------------------------------------------------
// -- clear
// --   Empties out the statistics and posterior hyperparameters and the
// -- sampled parameters. This basically deletes all notions of data from
// -- the node.
// --------------------------------------------------------------------------
void dir_sampled_full::clear()
{
   empty();
   update_posteriors();
}
// --------------------------------------------------------------------------
// -- empty
// --   Empties out the statistics and posterior hyperparameters, but not
// -- the sampled parameters
// --------------------------------------------------------------------------
void dir_sampled_full::empty()
{
   N = 0;
   totalCounts = 0;
   memset(indices_counts.data(),0,sizeof(int)*D);
   data_terms = 0;
}

bool dir_sampled_full::isempty() const               { return (N==0); }
int dir_sampled_full::getN() const                   { return N;}
int dir_sampled_full::getD() const                   { return D;}
arr(double) dir_sampled_full::get_logpi() const      { return param.logpi;}
gsl_rng* dir_sampled_full::get_r()                   { return rArr[0];}

void dir_sampled_full::set_multinomial(multinomial &other)
{
   param = other;
}
void dir_sampled_full::set_multinomial(arr(double) _logpi)
{
   memcpy(param.logpi, _logpi, sizeof(double)*D);
}

multinomial* dir_sampled_full::get_multinomial()
{
   return &param;
}



// --------------------------------------------------------------------------
// -- update_posteriors
// --   Updates the posterior hyperparameters
// --------------------------------------------------------------------------
void dir_sampled_full::update_posteriors()
{
}

void dir_sampled_full::update_posteriors_sample()
{
   update_posteriors();
   sample();
}

// helper function to merge to indices_counts
void dir_sampled_full::merge_indices_counts(int* indices_counts2)
{
   #pragma omp parallel for
   for (int d=0; d<D; d++)
   {
      indices_counts[d] += indices_counts2[d];
   }
   memset(indices_counts2, 0, sizeof(int)*D);
}


void dir_sampled_full::merge_with(dir_sampled_full* other, bool doSample)
{
   if (other!=NULL)
      merge_with(*other, doSample);
}
void dir_sampled_full::merge_with(dir_sampled_full* other1, dir_sampled_full* other2, bool doSample)
{
   if (other1==NULL && other2!=NULL)
      merge_with(*other2,doSample);
   else if (other1!=NULL && other2==NULL)
      merge_with(*other1,doSample);
   else if (other1!=NULL && other2!=NULL)
      merge_with(*other1, *other2, doSample);
}
void dir_sampled_full::merge_with(dir_sampled_full other, bool doSample)
{
   N += other.N;
   totalCounts += other.totalCounts;
   data_terms += other.data_terms;
   merge_indices_counts(other.indices_counts.data());
   if (doSample)
      update_posteriors_sample();
   else
      update_posteriors();
}
void dir_sampled_full::merge_with(dir_sampled_full other1, dir_sampled_full other2, bool doSample)
{
   N += other1.N + other2.N;
   totalCounts += other1.totalCounts + other2.totalCounts;
   data_terms += other1.data_terms + other2.data_terms;
   merge_indices_counts(other1.indices_counts.data());
   merge_indices_counts(other2.indices_counts.data());
   if (doSample)
      update_posteriors_sample();
   else
      update_posteriors();
}
void dir_sampled_full::set_stats(int _N, int _totalCounts, double _data_terms, int* _indices_counts)
{
   N = _N;
   totalCounts = _totalCounts;
   data_terms = _data_terms;
   memcpy(indices_counts.data(), _indices_counts, sizeof(int)*D);
}




double dir_sampled_full::data_loglikelihood() const
{
   return param.data_loglikelihood(indices_counts.data());
}

double dir_sampled_full::data_loglikelihood_marginalized()
{
   double log_likelihood = 0;

   int temp_counts_N = 0;
   #pragma omp parallel for reduction(+:log_likelihood,temp_counts_N)
   for (int d=0; d<D; d++)
   {
      int count = indices_counts[d];
      if (count>0)
      {
         log_likelihood += gammaln(count + alphah);
         temp_counts_N++;
      }
   }
   log_likelihood += -temp_counts_N*gammaln(alphah) + gammaln(D*alphah) - gammaln(totalCounts+D*alphah);
   log_likelihood += data_terms;

   return log_likelihood;
}

double dir_sampled_full::data_loglikelihood_marginalized_testmerge(dir_sampled_full *other) const
{
   double log_likelihood = 0;

   int temp_counts_N = 0;
   int temp_totalCounts = 0;
   #pragma omp parallel for reduction(+:log_likelihood,temp_counts_N,temp_totalCounts)
   for (int d=0; d<D; d++)
   {
      int count = indices_counts[d] + other->indices_counts[d];
      if (count>0)
      {
         log_likelihood += gammaln(count + alphah);
         temp_counts_N++;
         temp_totalCounts += count;
      }
   }
   log_likelihood += -temp_counts_N*gammaln(alphah) + gammaln(D*alphah) - gammaln(temp_totalCounts+D*alphah);
   log_likelihood += data_terms + other->data_terms;

   return log_likelihood;
}


void dir_sampled_full::sample()
{
   double total = 0;
   arr(double) logpi = param.logpi;
   // first sample all assuming zero counts
   // now sample the nonzero ones
   #pragma omp parallel for reduction(+:total)
   for (int d=0; d<D; d++)
   {
      int count = indices_counts[d];
      logpi[d] = gsl_ran_gamma(rArr[omp_get_thread_num()], alphah+count, 1);
      total += logpi[d];
   }
   total = log(total);
   #pragma omp parallel for
   for (int d=0; d<D; d++)
   {
      logpi[d] = log(logpi[d]) - total;
   }
}


double dir_sampled_full::Jdivergence(const dir_sampled_full &other)
{
   // ignore the det(cov) because it will be cancelled out
   return param.Jdivergence(other.param);
}





void dir_sampled_full::rem_data(arr(long) indices, arr(double) values, long nnz)
{
   // update the sufficient stats and the N
   if (N<=0) mexErrMsgTxt("Removing from empty cluster!\n");
   N--;

   if (N==0)
      empty();
   else
   {
      int Ci = 0;
      double new_data_term = 0;
      for (long di=0; di<nnz; di++)
      {
         int count = values[di];
         totalCounts -= count;
         Ci += count;
         new_data_term -= gammalnint(count+1);

         // find it
         indices_counts[indices[di]] -= count;
      }
      new_data_term += gammalnint(Ci+1);
      data_terms -= new_data_term;
   }
}
void dir_sampled_full::add_data_init(arr(long) indices, arr(double) values, long nnz)
{
   N++;

   int Ci = 0;
   double new_data_term = 0;
   for (long di=0; di<nnz; di++)
   {
      int count = values[di];
      totalCounts += count;
      Ci += count;
      new_data_term -= gammalnint(count+1);
      indices_counts[indices[di]] += count;
   }
   new_data_term += gammalnint(Ci+1);
   data_terms += new_data_term;
}
void dir_sampled_full::add_data(arr(long) indices, arr(double) values, long nnz)
{
   add_data_init(indices, values, nnz);
}
