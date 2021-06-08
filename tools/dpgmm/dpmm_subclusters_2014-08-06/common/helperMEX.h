// =============================================================================
// == helperMEX.h
// == --------------------------------------------------------------------------
// == Functions to help in MEX interfaces
// == --------------------------------------------------------------------------
// == Copyright 2011. MIT. All Rights Reserved.
// == Written by Jason Chang 06-13-2011
// =============================================================================

#ifndef HELPERMEX
#define HELPERMEX
enum inputType {SCALAR, VECTOR, MATRIX, MATRIX3, NSCALAR, CELL, STRUCT};

#include "mex.h"
#include <math.h>

//#define DEBUG // define by compiler
// compile with "mex -O -DDEBUG ..." for debugging"
// compile with "mex -O ..." for debugging"

#include "array.h"
#include "gsl/gsl_rng.h"

#ifdef DEBUG_ARR
   #define arr(type)    array<type>

   template <typename T>
   array<T> allocate_memory(unsigned int length)
   {
      return array<T>(length);
   }
   template <typename T>
   array<T> allocate_memory(unsigned int length, T value)
   {
      return array<T>(length, value);
   }
   template <typename T>
   array<T> assign_memory(const mxArray *pm)
   {
      return array<T>(mxGetNumberOfElements(pm), (T*)mxGetData(pm));
   }
   template <typename T>
   void deallocate_memory(array<T> &ptr)
   {
      delete[] (ptr.getData());
      //mxFree(ptr.getData());
   }
   template <typename T>
   void copy_memory(array<T> &ptr1, const T* ptr2, size_t num)
   {
      ptr1.copy_memory(ptr2, num);
   }
   template <typename T>
   void copy_memory(array<T> &ptr1, const array<T> ptr2, size_t num)
   {
      ptr1.copy_memory(ptr2.getData(), num);
   }
   template <typename T>
   void set_memory(array<T> &ptr1, const T value, size_t num)
   {
      ptr1.set_memory(value, num);
   }
   template <typename T>
   array<T> getArrayInput(const mxArray *prhs[], int index)
   {
      return array<T>(mxGetNumberOfElements(prhs[index]), (T *) mxGetData(prhs[index]));
   }
   template <typename T>
   array<T> getArrayInput(const mxArray* pm)
   {
      if (pm==NULL)
         mexErrMsgTxt("Getting array input of NULL!");
      return array<T>(mxGetNumberOfElements(pm), (T *) mxGetPr(pm));
   }
#else
   #define arr(type)    type*

   template <typename T>
   T* allocate_memory(unsigned int length)
   {
      return new T[length];
      //return (T*)mxCalloc(length, sizeof(T));
   }
   template <typename T>
   T* allocate_memory(unsigned int length, T value)
   {
      T* ptr = new T[length];
      //T* ptr = (T*)mxCalloc(length, sizeof(T));
      for (int i=length-1; i>=0; i--)
         ptr[i] = value;
      return ptr;
   }
   template <typename T>
   T* assign_memory(const mxArray *pm)
   {
      return (T*)mxGetData(pm);
   }
   template <typename T>
   void deallocate_memory(T* ptr)
   {
      delete[] ptr;
      //mxFree(ptr);
   }
   template <typename T>
   void copy_memory(T* ptr1, const T* ptr2, size_t num)
   {
      memcpy(ptr1, ptr2, num);
   }
   template <typename T>
   void copy_memory(T* ptr1, const array<T> ptr2, size_t num)
   {
      memcpy(ptr1, ptr2.getData(), num);
   }
   template <typename T>
   void set_memory(T* ptr1, const T value, size_t num)
   {
      memset(ptr1, value, num);
   }
   template <typename T>
   T* getArrayInput(const mxArray *prhs[], int index)
   {
      return (T *) mxGetData(prhs[index]);
   }
   template <typename T>
   T* getArrayInput(const mxArray* pm)
   {
      if (pm==NULL)
         mexErrMsgTxt("Getting array input of NULL!");
      return (T *) mxGetPr(pm);
   }
#endif

inline mxArray* getField(const mxArray *pm, mwIndex index, const char *fieldname)
{
   if (mxGetFieldNumber(pm, fieldname)<0)
   {
      mexPrintf("%s --\n", fieldname);
      mexErrMsgTxt("Not a field\n");
   }
   return mxGetField(pm, index, fieldname);
}

// output stuff
template <typename T>
mxArray *mxCreateArray(int mrows, int ncols, arr(T) input)
{
   mxArray *result;
   result = mxCreateDoubleMatrix(mrows,ncols,mxREAL);
   double* array = mxGetPr(result);
   for (int i=0; i<mrows*ncols; i++)
      array[i] = input[i];
   return result;
}
inline mxArray* mxCreateScalar(double var)
{
   mxArray *result;
   result = mxCreateDoubleMatrix(1,1,mxREAL);
   *mxGetPr(result) = var;
   return result;
}
inline mxArray* mxCreateScalar(bool var)
{
   mxArray *result;
   result = mxCreateLogicalMatrix(1,1);
   *mxGetPr(result) = var;
   return result;
}
inline void mxWriteField(mxArray *mstruct,int ind,const char *fieldstr,mxArray *mfield)
{
   if (mxGetField(mstruct,ind,fieldstr) != NULL)
      mxDestroyArray(mxGetField(mstruct,ind,fieldstr));
   mxSetField(mstruct,ind,fieldstr,mfield);
}




template <typename T>
T min(T v1, T v2)
{
   return (v1<v2 ? v1 : v2);
}

template <typename T>
T max(T v1, T v2)
{
   return (v1>v2 ? v1 : v2);
}


template <typename T>
T getInput(const mxArray *prhs[], int index)
{
   return *( (T *) mxGetData(prhs[index]));
}
template <typename T>
T getInput(const mxArray *pm)
{
   if (pm==NULL)
      mexErrMsgTxt("Getting input of NULL!");
   return *( (T *) mxGetData(pm));
}


inline void dbl_memset(double* ptr, double val, int N)
{
   for (int i=N-1; i>=0; i--)
      ptr[i] = val;
}

inline double mx_rand(void)
{
  mxArray *plhs[1];
  if(mexCallMATLAB(1,plhs,0,NULL,"rand")) {
    mexErrMsgTxt("mexCallMATLAB(rand) failed");
  }
  return mxGetPr(plhs[0])[0];
}


inline gsl_rng* initialize_gsl_rand()
{
   const gsl_rng_type * T;
   gsl_rng * r;
   gsl_rng_env_setup();

   T = gsl_rng_default;
   r = gsl_rng_alloc (T);
   gsl_rng_set(r, (unsigned long)(mx_rand()*pow(2,32)));
   return r;
}
inline gsl_rng* initialize_gsl_rand(int seed)
{
   const gsl_rng_type * T;
   gsl_rng * r;
   gsl_rng_env_setup();

   T = gsl_rng_default;
   r = gsl_rng_alloc (T);
   gsl_rng_set(r, (unsigned long)(seed));
   return r;
}
inline gsl_rng* initialize_gsl_rand(double seed)
{
   const gsl_rng_type * T;
   gsl_rng * r;
   gsl_rng_env_setup();

   T = gsl_rng_default;
   r = gsl_rng_alloc (T);
   gsl_rng_set(r, (unsigned long)(seed*pow(2,32)));
   return r;
}
inline double my_rand(gsl_rng* r)
{
   return gsl_rng_uniform_pos(r);
}
inline unsigned long int my_randi(gsl_rng* r)
{
   return gsl_rng_get(r);
}


// --------------------------------------------------------------------------
// -- checkInput
// --   Checks the input to be the correct type and/or dimensions.  If
// -- incorrect, this will throw an error.
// --
// --   parameters:
// --     - prhs : the parameters passed into the MEX function
// --     - index : the index of the parameter to check
// --     - flag : a character indicating what the parameter must be
// --         's' : scalar
// --         'v' : vector
// --         'm' : matrix
// --         'n' : not scalar (vector or matrix)
// --         'c' : cell
// --------------------------------------------------------------------------
inline void checkInput(const mxArray *prhs, inputType type)
{
   // flag is either 's'-scalar, 'v'-vector, 'm'-matrix
   int mrows = mxGetM(prhs);
   int ncols = mxGetN(prhs);
   int ndims = mxGetNumberOfDimensions(prhs);

   switch (type)
   {
      case SCALAR:
         if (!(mrows==1 && ncols==1))
         {
            std::cerr << " ->Error with input\n";
            mexErrMsgTxt("    Should be a scalar");
         }
         break;
      case VECTOR:
         if ((mrows==1 && ncols==1) || (mrows!=1 && ncols!=1))
         {
            std::cerr << " ->Error with input\n";
            mexErrMsgTxt("    Should be a vector");
         }
         break;
      case MATRIX:
         if (mrows==1 || ncols==1)
         {
            std::cerr << " ->Error with input\n";
            mexErrMsgTxt("    Should be a matrix");
         }
         break;
      case MATRIX3:
         if (ndims!=3)
         {
            std::cerr << " ->Error with input\n";
            mexErrMsgTxt("    Should be a 3D array");
         }
         break;
      case NSCALAR:
         if (mrows==1 && ncols==1)
         {
            std::cerr << " ->Error with input\n";
            mexErrMsgTxt("    Should not be a scalar");
         }
         break;
      case CELL:
         if (!mxIsCell(prhs))
         {
            std::cerr << " ->Error with input\n";
            mexErrMsgTxt("    Should be a cell array");
         }
         break;
      case STRUCT:
         if (!mxIsStruct(prhs))
         {
            std::cerr << " ->Error with input\n";
            mexErrMsgTxt("    Should be a structure");
         }
   }
}
inline void checkInput(const mxArray *prhs[], int index, inputType type)
{
   checkInput(prhs[index], type);
}
inline void checkInput(const mxArray prhs[], const char* classname)
{
   if (!mxIsClass(prhs, classname))
   {
      std::cerr << " ->Error with input\n";
      mexErrMsgTxt("    Class error");
   }
}
inline void checkInput(const mxArray prhs[], inputType type, const char* classname)
{
   checkInput(prhs, type);
   checkInput(prhs, classname);
}
inline void checkInput(const mxArray *prhs[], int index, inputType type, const char* classname)
{
   checkInput(prhs[index], type, classname);
}


// --------------------------------------------------------------------------
// -- checkInput
// --   Checks the input to be the correct type and/or dimensions.  If
// -- incorrect, this will throw an error.
// --
// --   parameters:
// --     - prhs : the parameters passed into the MEX function
// --     - index : the index of the parameter to check
// --     - flag : a character indicating what the parameter must be
// --         's' : scalar
// --         'v' : vector
// --         'm' : matrix
// --         'n' : not scalar (vector or matrix)
// --         'c' : cell
// --------------------------------------------------------------------------
inline void checkInput(const mxArray *prhs[], int index, char flag)
{
   // flag is either 's'-scalar, 'v'-vector, 'm'-matrix
   int mrows = mxGetM(prhs[index]);
   int ncols = mxGetN(prhs[index]);
   int ndims = mxGetNumberOfDimensions(prhs[index]);

   switch (flag)
   {
      case 's':
         if (!(mrows==1 && ncols==1))
         {
            puts(" ->Error with input!");
            puts("   Input number:");
            char str[3];
            sprintf(str, "%d", index);
            puts(str);
            mexErrMsgTxt("    Should be a scalar");
         }
         break;
      case 'v':
         if ((mrows==1 && ncols==1) || (mrows!=1 && ncols!=1))
         {
            puts(" ->Error with input!");
            puts("   Input number:");
            char str[3];
            sprintf(str, "%d", index);
            puts(str);
            mexErrMsgTxt("    Should be a vector");
         }
         break;
      case 'm':
         if (mrows==1 || ncols==1)
         {
            puts(" ->Error with input!");
            puts("   Input number:");
            char str[3];
            sprintf(str, "%d", index);
            puts(str);
            mexErrMsgTxt("    Should be a matrix");
         }
         break;
      case 'n':
         if (mrows==1 && ncols==1)
         {
            puts(" ->Error with input!");
            puts("   Input number:");
            char str[3];
            sprintf(str, "%d", index);
            puts(str);
            mexErrMsgTxt("    Should not be a scalar");
         }
         break;
      case 'c':
         if (!mxIsCell(prhs[index]))
         {
            puts(" ->Error with input!");
            puts("   Input number:");
            char str[3];
            sprintf(str, "%d", index);
            puts(str);
            mexErrMsgTxt("    Should be a cell array");
         }
         break;
      case '3':
         if (ndims!=3)
         {
            puts(" ->Error with input!");
            puts("   Input number:");
            char str[3];
            sprintf(str, "%d", index);
            puts(str);
            mexErrMsgTxt("    Should be a 3D array");
         }
         break;
   }
}

// --------------------------------------------------------------------------
// -- findCoordinates
// --   converts 1D coordinates to 2D coordinates
// --
// --   parameters:
// --     - (X,Y) : the dimensions
// --     - index : the 1D coordinate
// --
// --   return parameters:
// --     - (x,y) : the 2D coordinates
// --------------------------------------------------------------------------
inline void findCoordinates(int X, int Y, int index, int &x, int &y)
{
   // index first does x's then does y's
   x = index%Y;
   y = index/Y;
}

// --------------------------------------------------------------------------
// -- findCoordinates
// --   converts 2D coordinates to 1D coordinate
// --
// --   parameters:
// --     - (X,Y) : the dimensions
// --     - (x,y) : the 2D coordinates
// --
// --   return value:
// --     - index : the 1D coordinates
// --------------------------------------------------------------------------
inline int findIndex(int X, int Y, int x, int y)
{
   return y*X + x;
}
// --------------------------------------------------------------------------
// -- findCoordinates
// --   converts 3D coordinates to 1D coordinate
// --
// --   parameters:
// --     - (X,Y,Z) : the dimensions
// --     - (x,y,z) : the 3D coordinates
// --
// --   return value:
// --     - index : the 1D coordinates
// --------------------------------------------------------------------------
inline int findIndex(int X, int Y, int Z, int x, int y, int z)
{
   return z*X*Y + y*X + x;
}


#endif
