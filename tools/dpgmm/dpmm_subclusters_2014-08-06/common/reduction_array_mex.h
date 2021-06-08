#ifndef REDUCTION_ARRAY
#define REDUCTION_ARRAY

#include "assert.h"
#include "string.h"
#include <cstdlib>
#include <stdlib.h>
#include <iostream>
#include "mex.h"

#define _min_D_cache_line 32

template <typename T>
class reduction_array
{
private:
   arr(T) data;
   int numThreads;
   int D;

public:
   // --------------------------------------------------------------------------
   // -- array
   // --   constructor; initializes the array to nothing
   // --------------------------------------------------------------------------
   reduction_array();
   reduction_array(int thenumThreads, int theD);
   reduction_array(int thenumThreads, int theD, T value);
   virtual ~reduction_array();
   
   void init_values(T value);

   // --------------------------------------------------------------------------
   // -- operator[]
   // --   returns the value at nIndex
   // --------------------------------------------------------------------------
   arr(T)& operator[](const int nIndex) const;
   
   void reduce(int t, int d, T value);
   
   arr(T) final_reduce();
};


// --------------------------------------------------------------------------
// -- array
// --   constructor; initializes the array to nothing
// --------------------------------------------------------------------------
template <typename T>
reduction_array<T>::reduction_array() :
   numThreads(0), D(0), data(NULL)
{
}

template <typename T>
reduction_array<T>::reduction_array(int thenumThreads, int theD) :
   numThreads(thenumThreads)
{
   D = max(D, _min_D_cache_line);
   data = allocate_memory<T>(numThreads*D);
}

template <typename T>
reduction_array<T>::reduction_array(int thenumThreads, int theD, T value) :
   numThreads(thenumThreads), D(theD)
{
   D = max(D, _min_D_cache_line);
   data = allocate_memory<T>(numThreads*D);
   memset(data, value, sizeof(T)*numThreads*D);
}

template <typename T>
reduction_array<T>::~reduction_array()
{
   deallocate_memory(data);
}

template <typename T>
void reduction_array<T>::init_values(T value)
{
   memset(data, value, sizeof(T)*numThreads*D);
}

// --------------------------------------------------------------------------
// -- operator[]
// --   returns the value at nIndex
// --------------------------------------------------------------------------
template <typename T>
arr(T)& reduction_array<T>::operator[](const int nIndex) const
{
   return data[nIndex*D];
}


template <typename T>
void reduction_array<T>::reduce(int t, int d, T value)
{
   data[t*D+d] += value;
}

template <typename T>
arr(T) reduction_array<T>::final_reduce()
{
   for (int d=0; d<D; d++)
   {
      double temp = 0;
      for (int t=1; t<numThreads; t++)
         temp += data[t*D+d];
      data[d] += temp;
   }
   return data;
}


#endif
