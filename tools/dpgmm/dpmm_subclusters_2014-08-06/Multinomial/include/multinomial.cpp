// =============================================================================
// == multinomial.cpp
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

#include "multinomial.h"

// --------------------------------------------------------------------------
// -- multinomial
// --   constructor; initializes to empty
// --------------------------------------------------------------------------
multinomial::multinomial() : D(0), logpi(0)
{
}

// --------------------------------------------------------------------------
// -- multinomial
// --   copy constructor;
// --------------------------------------------------------------------------
multinomial::multinomial(const multinomial& that)
{
   copy(that);
}
// --------------------------------------------------------------------------
// -- operator=
// --   assignment operator
// --------------------------------------------------------------------------
multinomial& multinomial::operator=(const multinomial& that)
{
   if (this != &that)
   {
      if (D==that.D)
         memcpy(logpi, that.logpi, sizeof(double)*D);
      else
      {
         cleanup();
         copy(that);
      }
   }
   return *this;
}
// --------------------------------------------------------------------------
// -- copy
// --   returns a copy of this
// --------------------------------------------------------------------------
void multinomial::copy(const multinomial& that)
{
   D = that.D;
   logpi = allocate_memory<double>(D);
   memcpy(logpi, that.logpi, sizeof(double)*D);
}

// --------------------------------------------------------------------------
// -- multinomial
// --   constructor; intializes to all the values given
// --------------------------------------------------------------------------
multinomial::multinomial(int _D)
{
   D = _D;
   logpi = allocate_memory<double>(D);
}

// --------------------------------------------------------------------------
// -- ~multinomial
// --   destructor
// --------------------------------------------------------------------------
multinomial::~multinomial()
{
   cleanup();
}
// --------------------------------------------------------------------------
// -- ~cleanup
// --   deletes all the memory allocated by this
// --------------------------------------------------------------------------
void multinomial::cleanup()
{
   if (logpi)   deallocate_memory(logpi);   logpi = NULL;
}

