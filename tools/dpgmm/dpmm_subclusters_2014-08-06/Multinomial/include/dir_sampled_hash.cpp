// =============================================================================
// == dir_sampled_hash.cpp
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

#include "dir_sampled_hash.h"

#ifndef pi
#define pi 3.14159265
#endif

#ifndef clogpi
#define clogpi 1.144729885849
#endif

// --------------------------------------------------------------------------
// -- dir_sampled_hash
// --   constructor; initializes to empty
// --------------------------------------------------------------------------
dir_sampled_hash::dir_sampled_hash() :
   D(0), Nthreads(0), alphah(0), N(0), totalCounts(0), param(0), rArr(NULL), data_terms(0),
   temp_counts_itr(NULL), temp_indices_itr(NULL), temp_counts_N(0), temp_counts_invalid(true)
{
   gammaln = &gsl_sf_lngamma;
}

// --------------------------------------------------------------------------
// -- dir_sampled_hash
// --   copy constructor;
// --------------------------------------------------------------------------
dir_sampled_hash::dir_sampled_hash(const dir_sampled_hash& that)
{
   copy(that);
}
// --------------------------------------------------------------------------
// -- operator=
// --   assignment operator
// --------------------------------------------------------------------------
dir_sampled_hash& dir_sampled_hash::operator=(const dir_sampled_hash& that)
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
void dir_sampled_hash::copy(const dir_sampled_hash& that)
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

   temp_counts_itr = NULL;
   temp_indices_itr = NULL;
   temp_counts_N = 0;
   temp_counts_invalid = true;
}

// --------------------------------------------------------------------------
// -- dir_sampled_hash
// --   constructor; intializes to all the values given
// --------------------------------------------------------------------------
dir_sampled_hash::dir_sampled_hash(int _Nthreads, int _D, double _alphah) :
   N(0), totalCounts(0), Nthreads(_Nthreads), D(_D), alphah(_alphah), param(_D), data_terms(0),
   temp_counts_itr(NULL), temp_indices_itr(NULL), temp_counts_N(0), temp_counts_invalid(true)
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
// -- ~dir_sampled_hash
// --   destructor
// --------------------------------------------------------------------------
dir_sampled_hash::~dir_sampled_hash()
{
   cleanup();
}
// --------------------------------------------------------------------------
// -- ~cleanup
// --   deletes all the memory allocated by this
// --------------------------------------------------------------------------
void dir_sampled_hash::cleanup()
{
   indices_counts.clear();
   if (rArr)
   {
      for (int t=0; t<Nthreads; t++)
         if (rArr[t]!=NULL)
            gsl_rng_free(rArr[t]);
      deallocate_memory(rArr);
   }

   if (temp_counts_itr!=NULL && temp_counts_N>0)
   {
      delete[] temp_counts_itr;
      delete[] temp_indices_itr;
      temp_counts_itr = NULL;
      temp_indices_itr = NULL;
      temp_counts_N = 0;
      temp_counts_invalid = true;
   }
}

// --------------------------------------------------------------------------
// -- clear
// --   Empties out the statistics and posterior hyperparameters and the
// -- sampled parameters. This basically deletes all notions of data from
// -- the node.
// --------------------------------------------------------------------------
void dir_sampled_hash::clear()
{
   empty();
   update_posteriors();
}
// --------------------------------------------------------------------------
// -- empty
// --   Empties out the statistics and posterior hyperparameters, but not
// -- the sampled parameters
// --------------------------------------------------------------------------
void dir_sampled_hash::empty()
{
   N = 0;
   totalCounts = 0;
   indices_counts.clear();
   data_terms = 0;
}

bool dir_sampled_hash::isempty() const               { return (N==0); }
int dir_sampled_hash::getN() const                   { return N;}
int dir_sampled_hash::getD() const                   { return D;}
arr(double) dir_sampled_hash::get_logpi() const      { return param.logpi;}
gsl_rng* dir_sampled_hash::get_r()                   { return rArr[0];}

void dir_sampled_hash::set_multinomial(multinomial &other)
{
   param = other;
}
void dir_sampled_hash::set_multinomial(arr(double) _logpi)
{
   memcpy(param.logpi, _logpi, sizeof(double)*D);
}

multinomial* dir_sampled_hash::get_multinomial()
{
   return &param;
}



// --------------------------------------------------------------------------
// -- update_posteriors
// --   Updates the posterior hyperparameters
// --------------------------------------------------------------------------
void dir_sampled_hash::update_posteriors()
{
   temp_counts_invalid = true;
}

void dir_sampled_hash::update_posteriors_sample()
{
   update_posteriors();
   sample();
}

// helper function to merge to indices_counts
void dir_sampled_hash::merge_indices_counts(unordered_map<int,int> &indices_counts2)
{
   //for (std::tr1::unordered_map<int,int>::iterator &itr : indices_counts2)
   for (unordered_map<int,int>::iterator itr=indices_counts2.begin(); itr!=indices_counts2.end(); itr++)
   {
      int index = itr->first;
      int count = itr->second;
      totalCounts += count;

      // try to insert
      std::pair< unordered_map<int,int>::iterator, bool> itrFnd = indices_counts.emplace(index,count);
      // see if it was found.  if it was found, just add the counts
      if (!itrFnd.second)
         itrFnd.first->second += count;
   }
   indices_counts2.clear();
}


void dir_sampled_hash::merge_with(dir_sampled_hash* other, bool doSample)
{
   if (other!=NULL)
      merge_with(*other, doSample);
   temp_counts_invalid = true;
}
void dir_sampled_hash::merge_with(dir_sampled_hash* other1, dir_sampled_hash* other2, bool doSample)
{
   if (other1==NULL && other2!=NULL)
      merge_with(*other2,doSample);
   else if (other1!=NULL && other2==NULL)
      merge_with(*other1,doSample);
   else if (other1!=NULL && other2!=NULL)
      merge_with(*other1, *other2, doSample);
   temp_counts_invalid = true;
}
void dir_sampled_hash::merge_with(dir_sampled_hash other, bool doSample)
{
   N += other.N;
   data_terms += other.data_terms;
   merge_indices_counts(other.indices_counts);
   if (doSample)
      update_posteriors_sample();
   else
      update_posteriors();
   temp_counts_invalid = true;
}
void dir_sampled_hash::merge_with(dir_sampled_hash other1, dir_sampled_hash other2, bool doSample)
{
   N += other1.N + other2.N;
   data_terms += other1.data_terms + other2.data_terms;
   merge_indices_counts(other1.indices_counts);
   merge_indices_counts(other2.indices_counts);
   if (doSample)
      update_posteriors_sample();
   else
      update_posteriors();
   temp_counts_invalid = true;
}
void dir_sampled_hash::set_stats(int _N, int _totalCounts, double _data_terms, unordered_map<int,int> &_indices_counts)
{
   N = _N;
   totalCounts = _totalCounts;
   data_terms = _data_terms;
   indices_counts = _indices_counts;
   temp_counts_invalid = true;
}




double dir_sampled_hash::data_loglikelihood() const
{
   return param.data_loglikelihood(indices_counts);
}

double dir_sampled_hash::data_loglikelihood_marginalized()
{
   double log_likelihood = 0;

   prepare_indices_counts_itr();
   #pragma omp parallel for reduction(+:log_likelihood)
   for (int i=0; i<temp_counts_N; i++)
   {
      int count = *temp_counts_itr[i];
      log_likelihood += gammaln(count + alphah);
   }
   log_likelihood += -temp_counts_N*gammaln(alphah) + gammaln(D*alphah) - gammaln(totalCounts+D*alphah);
   log_likelihood += data_terms;

   return log_likelihood;
}

double dir_sampled_hash::data_loglikelihood_marginalized_testmerge(dir_sampled_hash *other) const
{
   dir_sampled_hash temp = *this;
   temp.merge_with(other,false);
   return temp.data_loglikelihood_marginalized();
}


void dir_sampled_hash::sample()
{
   double total = 0;
   arr(double) logpi = param.logpi;
   // first sample all assuming zero counts
   #pragma omp parallel for reduction(+:total)
   for (int d=0; d<D; d++)
   {
      logpi[d] = gsl_ran_gamma(rArr[omp_get_thread_num()], alphah, 1);
      total += logpi[d];
   }
   // now sample the nonzero ones
   prepare_indices_counts_itr();
   #pragma omp parallel for reduction(+:total)
   for (int i=0; i<temp_counts_N; i++)
   {
      int d = temp_indices_itr[i];
      int count = *(temp_counts_itr[i]);
      total -= logpi[d];
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


double dir_sampled_hash::Jdivergence(const dir_sampled_hash &other)
{
   // ignore the det(cov) because it will be cancelled out
   return param.Jdivergence(other.param);
}





void dir_sampled_hash::rem_data(arr(long) indices, arr(double) values, long nnz)
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
         unordered_map<int,int>::iterator itr = indices_counts.find(indices[di]);
         if (itr == indices_counts.end())
         {
            mexPrintf("N=%d, d=%d, v=%d\n", N, (int)indices[di], (int)values[di]);
            mexErrMsgTxt("Removing something that didn't exist\n");
         }
         if ((itr->second -= count) == 0) // subtract
            indices_counts.erase(itr); // remove if zero
      }
      new_data_term += gammalnint(Ci+1);
      data_terms -= new_data_term;
   }
   temp_counts_invalid = true;
}
void dir_sampled_hash::add_data_init(arr(long) indices, arr(double) values, long nnz)
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

      // try to insert
      std::pair< unordered_map<int,int>::iterator, bool> itrFnd = indices_counts.emplace(indices[di],values[di]);
      // see if it was found.  if it was found, just add the counts
      if (!itrFnd.second)
         itrFnd.first->second += count;
   }
   new_data_term += gammalnint(Ci+1);
   data_terms += new_data_term;

   temp_counts_invalid = true;
}
void dir_sampled_hash::add_data(arr(long) indices, arr(double) values, long nnz)
{
   add_data_init(indices, values, nnz);
}



void dir_sampled_hash::prepare_indices_counts_itr()
{
   if (temp_counts_invalid)
   {
      if (temp_counts_N != indices_counts.size())
      {
         if (temp_counts_itr!=NULL) delete[] temp_counts_itr;
         if (temp_indices_itr!=NULL) delete[] temp_indices_itr;
         temp_counts_N = indices_counts.size();
         temp_counts_itr = new int*[temp_counts_N];
         temp_indices_itr = new int[temp_counts_N];
      }

      int i = 0;
      for (unordered_map<int,int>::iterator itr=indices_counts.begin(); itr!=indices_counts.end(); itr++, i++)
      {
         temp_indices_itr[i] = itr->first;
         temp_counts_itr[i] = &(itr->second);
      }
      temp_counts_invalid = false;
   }
}

