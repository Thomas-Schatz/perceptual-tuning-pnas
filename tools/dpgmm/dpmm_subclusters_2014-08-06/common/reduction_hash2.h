// =============================================================================
// == reduction_hash2.h
// == --------------------------------------------------------------------------
// == A reduction array class that can be used with OpenMP. Current OpenMP
// == software only allows the reduction of an array into a scalar. This class
// == allows one to reduce an array into an array. Also, since the code here
// == implements a reduction similar to OpenMP without actually using OpenMP's
// == reduction clause, the reduction_sparray class allows users to define their
// == own reduction functions and reductions on user-specified classes.
// ==
// == An array of size numThreads x D is created. Reduction is performed so
// == that each thread only has write accesses a unique spot in the array.
// ==
// == Notation is as follows:
// ==   numThreads - The number of OpenMP threads that will be running
// ==   D - The dimension of the array to reduce to (if scalar, should be 1)
// ==
// == General usage:
// ==   (1) Before the parallel environment, initialize the reduction array:
// ==     >> reduction_sparray<double> my_arr(numThreads,D,initial_value);
// ==
// ==   (2) Inside the parallel for loop, you can reduce with data x with:
// ==     >> my_arr.reduce_XXX(omp_get_thread_num(), bin, x);
// ==       "XXX" can be the predefined "add", "multiply", "max", or "min"
// ==       or you can specify a function pointer to your own function with
// ==     >> my_arr.reduce_function(omp_get_thread_num(), bin, x, func_ptr);
// ==
// ==   (3) After the parallel environment, reduce on the separate threads:
// ==     >> double* output = my_arr.final_reduce_XXX();
// ==       Again, "XXX" can be any of the predefined functions, or you can use
// ==     >> double* output = my_arr.final_reduce_function(func_ptr);
// ==
// ==   (4) output now is an array of length D with the reduced values.
// ==       Do *NOT* attempt to deallocate output or my_arr as they have their
// ==       own destructors.
// ==
// == Notes:
// ==   (1) Because of possible "false sharing", the array size here is actually
// ==       numThreads x D x cache_line. We pad the array so that different
// ==       threads will not access the same cache_line. If the cache line is
// ==       not large enough, please increase ie manually.
// == --------------------------------------------------------------------------
// == Written by Jason Chang 04-14-2013 - jchang7@csail.mit.edu
// =============================================================================

#ifndef _REDUCTION_HASH2
#define _REDUCTION_HASH2

#include <string.h>
#include <cstdlib>
#include <stdlib.h>
#include <iostream>
#include <algorithm>
#include "array.h"
#include "helperMEX.h"
#include <boost/unordered_map.hpp>

using boost::unordered_map;
// this should be the cache line size.  set it to 4, might need to be higher to
// avoid false sharing
//#ifndef cache_line
//#define cache_line 4
//#endif
// not needed because of the sparse linked list

template <typename T>
class reduction_hash2
{
private:
   unordered_map< int, T >* data;
   //arr(arr(linkedList< indval<T> >)) data;
   int numThreads;
   int K;

public:
   // --------------------------------------------------------------------------
   // -- reduction_hash2
   // --   constructors; initializes the reduction array with a number of
   // -- threads, each with a D dimensional vector. The third parameter can be
   // -- specified to give the initializing value.
   // --------------------------------------------------------------------------
   reduction_hash2();
   reduction_hash2(int thenumThreads, int theK);
   virtual ~reduction_hash2();

   // --------------------------------------------------------------------------
   // -- clear_values
   // --   Clears the arrays
   // --------------------------------------------------------------------------
   void clear_values();

   // --------------------------------------------------------------------------
   // -- reduce_XXX
   // --   Performs the reduction "XXX" on the d^th dimension with value
   // --------------------------------------------------------------------------
   void reduce_inc(int t, int k, long d);
   void reduce_inc(int t, int k, long* ds, int nnz);
   template <typename T2>
   void reduce_add(int t, int k, long d, T2 value);
   template <typename T2>
   void reduce_add(int t, int k, long* ds, T2* values, int nnz);

   // --------------------------------------------------------------------------
   // -- final_reduce_XXX
   // --   Performs the reduction "XXX" on the threads and returns result.
   // --------------------------------------------------------------------------
   unordered_map<int,T>* final_reduce_add();
};

#endif

// --------------------------------------------------------------------------
// -- reduction_sparray
// --   constructors; initializes the reduction array with a number of
// -- threads, each with a D dimensional vector. The third parameter can be
// -- specified to give the initializing value.
// --------------------------------------------------------------------------
template <typename T>
reduction_hash2<T>::reduction_hash2() :
   numThreads(0), data(NULL)
{
}
template <typename T>
reduction_hash2<T>::reduction_hash2(int thenumThreads, int theK) :
   numThreads(thenumThreads), K(theK)
{
   data = allocate_memory< unordered_map< int, T > >(numThreads*K);

   //for (int t=0; t<numThreads*K; t++)
      //data[t].rehash(100/data[t].max_load_factor());
   /*data = allocate_memory< arr(linkedList< indval<T> >) >(numThreads);
   for (int t=0; t<numThreads; t++)
      data[t] = allocate_memory< linkedList< indval<T> > >(K);*/
}
template <typename T>
reduction_hash2<T>::~reduction_hash2()
{
   if (data!=NULL) deallocate_memory(data);

   /*if (data!=NULL)
   {
      for (int t=0; t<numThreads; t++)
         if (data[t]!=NULL) deallocate_memory(data[t]);
      deallocate_memory(data);
   }*/
}


// --------------------------------------------------------------------------
// -- clear_values
// --   Clears the arrays
// --------------------------------------------------------------------------
template <typename T>
void reduction_hash2<T>::clear_values()
{
   for (int t=0; t<numThreads*K; t++)
      data[t].clear();
   //for (int t=0; t<numThreads; t++) for (int k=0; k<K; k++)
      //data[t][k]->clear();
}


// --------------------------------------------------------------------------
// -- reduce_XXX
// --   Performs the reduction "XXX" on the d^th dimension with value
// --------------------------------------------------------------------------
template <typename T>
inline void reduction_hash2<T>::reduce_inc(int t, int k, long d)
{
   std::pair< typename unordered_map<int, T>::iterator, bool> itrFnd = data[t*K+k].emplace(d,1);
   if (!itrFnd.second)
      itrFnd.first->second++;
}
template <typename T>
inline void reduction_hash2<T>::reduce_inc(int t, int k, long* ds, int nnz)
{
   int bin = t*K+k;
   for (int i=0; i<nnz; i++)
   {
      std::pair< typename unordered_map<int, T>::iterator, bool> itrFnd = data[bin].emplace(ds[i],1);
      if (!itrFnd.second)
         itrFnd.first->second++;
   }
}
template <typename T> template <typename T2>
inline void reduction_hash2<T>::reduce_add(int t, int k, long d, T2 value)
{
   std::pair< typename unordered_map<int, T>::iterator, bool> itrFnd = data[t*K+k].emplace(d,value);
   if (!itrFnd.second)
      itrFnd.first->second += value;
}
template <typename T> template <typename T2>
inline void reduction_hash2<T>::reduce_add(int t, int k, long* ds, T2* values, int nnz)
{
   int bin = t*K+k;
   for (int i=0; i<nnz; i++)
   {
      std::pair< typename unordered_map<int, T>::iterator, bool> itrFnd = data[bin].emplace(ds[i], values[i]);
      if (!itrFnd.second)
         itrFnd.first->second += values[i];
   }
}


// --------------------------------------------------------------------------
// -- final_reduce_XXX
// --   Performs the reduction "XXX" on the threads and returns result.
// --------------------------------------------------------------------------
template <typename T>
inline unordered_map<int,T>* reduction_hash2<T>::final_reduce_add()
{
   for (int t=1; t<numThreads; t++)
      for (int k=0; k<K; k++)
      {
         for (unordered_map<int,int>::iterator itr=data[t*K+k].begin(); itr!=data[t*K+k].end(); itr++)
         {
            int index = itr->first;
            int count = itr->second;

            // try to insert
            std::pair< typename unordered_map<int,int>::iterator, bool> itrFnd = data[k].emplace(index,count);
            // see if it was found.  if it was found, just add the counts
            if (!itrFnd.second)
               itrFnd.first->second += count;
         }
      }
   return data;
}



