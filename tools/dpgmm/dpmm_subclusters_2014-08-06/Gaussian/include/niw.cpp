// =============================================================================
// == niw.cpp
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

#include "niw.h"

#ifndef pi
#define pi 3.14159265
#endif

#ifndef logpi
#define logpi 1.144729885849
#endif

// --------------------------------------------------------------------------
// -- niw
// --   constructor; initializes to empty
// --------------------------------------------------------------------------
niw::niw() :
   D(0), D2(0), kappah(0), nuh(0), thetah(NULL), Deltah(NULL), t(NULL), T(NULL), N(0),
   kappa(0), nu(0), theta(NULL), Delta(NULL), mean(NULL), prec(NULL), logDetCov(0),
   marginalize(true)
{
   r = initialize_gsl_rand(rand());
   logprior = 0;
   logmu = 0;
}

// --------------------------------------------------------------------------
// -- niw
// --   copy constructor;
// --------------------------------------------------------------------------
niw::niw(const niw& that)
{
   copy(that);
   r = initialize_gsl_rand(rand());
   logprior = that.logprior;
   logmu = that.logmu;
}
// --------------------------------------------------------------------------
// -- operator=
// --   assignment operator
// --------------------------------------------------------------------------
niw& niw::operator=(const niw& that)
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
void niw::copy(const niw& that)
{
   marginalize = that.marginalize;
   r = initialize_gsl_rand(rand());
   D = that.D;
   D2 = that.D2;
   kappah = that.kappah;
   nuh = that.nuh;
   N = that.N;
   kappa = that.kappa;
   nu = that.nu;
   logDetCov = that.logDetCov;
   thetah = allocate_memory<double>(D);
   Deltah = allocate_memory<double>(D2);
   t = allocate_memory<double>(D);
   T = allocate_memory<double>(D2);
   theta = allocate_memory<double>(D);
   Delta = allocate_memory<double>(D2);
   mean = allocate_memory<double>(D);
   prec = allocate_memory<double>(D2);
   logprior = that.logprior;
   logmu = that.logmu;

   for (int d=0; d<D; d++)
   {
      thetah[d] = that.thetah[d];
      t[d] = that.t[d];
      theta[d] = that.theta[d];
      mean[d] = that.mean[d];
   }
   for (int d=0; d<D2; d++)
   {
      Deltah[d] = that.Deltah[d];
      T[d] = that.T[d];
      Delta[d] = that.Delta[d];
      prec[d]= that.prec[d];
   }
}

// --------------------------------------------------------------------------
// -- niw
// --   constructor; intializes to all the values given
// --------------------------------------------------------------------------
niw::niw(bool _marginalize, int _D, double _kappah, double _nuh, arr(double) _thetah, arr(double) _Deltah)
{
   marginalize = _marginalize;
   D = _D;
   D2 = D*D;
   kappah = _kappah;
   nuh = _nuh;
   thetah = allocate_memory<double>(D);
   Deltah = allocate_memory<double>(D2);
   t = allocate_memory<double>(D);
   T = allocate_memory<double>(D2);
   theta = allocate_memory<double>(D);
   Delta = allocate_memory<double>(D2);
   mean = allocate_memory<double>(D);
   prec = allocate_memory<double>(D2);
   logprior = 0;
   logmu = 0;

   memcpy(thetah, _thetah, sizeof(double)*D);
   memcpy(Deltah, _Deltah, sizeof(double)*D2);

   r = initialize_gsl_rand(mx_rand());

   clear();
}




// --------------------------------------------------------------------------
// -- ~niw
// --   destructor
// --------------------------------------------------------------------------
niw::~niw()
{
   cleanup();
}
// --------------------------------------------------------------------------
// -- ~cleanup
// --   deletes all the memory allocated by this
// --------------------------------------------------------------------------
void niw::cleanup()
{
   if (thetah) deallocate_memory(thetah);
   if (Deltah) deallocate_memory(Deltah);
   if (t)      deallocate_memory(t);
   if (T)      deallocate_memory(T);
   if (theta)  deallocate_memory(theta);
   if (Delta)  deallocate_memory(Delta);
   if (mean)   deallocate_memory(mean);
   if (prec)   deallocate_memory(prec);
   gsl_rng_free(r);
}


// --------------------------------------------------------------------------
// -- empty
// --   Empties out the niw. Does not update the posterior hyperparameters.
// --------------------------------------------------------------------------
void niw::clear()
{
   empty();
   memset(mean, 0, sizeof(double)*D);
   memset(prec, 0, sizeof(double)*D2);

   update(true);
}
void niw::empty()
{
   N = 0;
   memset(t, 0, sizeof(double)*D);
   memset(T, 0, sizeof(double)*D2);
   memset(theta, 0, sizeof(double)*D);
   memset(Delta, 0, sizeof(double)*D2);
   kappa = kappah;
   nu = nuh;
}

bool niw::isempty()
{
   return (N==0);
}
int niw::getN()
{
   return N;
}
arr(double) niw::getMean()
{
   return mean;
}
arr(double) niw::getPrec()
{
   return prec;
}
double niw::getLogprior()
{
   return logprior;
}
double niw::get_logmu()
{
   return logmu;
}
void niw::set_marginalize(bool _marginalize)
{
   marginalize = _marginalize;
   update_predictive();
}

// --------------------------------------------------------------------------
// -- update_posteriors
// --   Updates the posterior hyperparameters
// --------------------------------------------------------------------------
void niw::update_posteriors()
{
   kappa = kappah + N;
   nu = nuh + N;
   for (int d=0; d<D; d++)
      theta[d] = (thetah[d]*kappah + t[d]) / kappa;
   for (int d=0; d<D2; d++)
      Delta[d] = (Deltah[d]*nuh + T[d] + kappah*thetah[d%D]*thetah[d/D] - kappa*theta[d%D]*theta[d/D]) / nu;
}

// --------------------------------------------------------------------------
// -- update_predictive
// --   Updates the parameters for the predictive moment-matched gaussian
// --------------------------------------------------------------------------
void niw::update_predictive()
{
   if (marginalize)
   {
      predictive_Z = myloggamma(0.5*(nu+1)) - myloggamma(0.5*(nu-D+1)) - 0.5*logDetCov - 0.5*D*log(nu-D+1);
      memcpy(mean, theta, sizeof(double)*D);
      for (int d=0; d<D2; d++)
         prec[d] = Delta[d] * (kappa+1)*nu / (kappa*(nu-D-1));
      logDetCov = log(det(prec,D));
      InvMat(prec,D);
   }
   else
      sample();
}

void niw::update(bool force_predictive)
{
   update_posteriors();
   if (force_predictive || marginalize)
      update_predictive();
}

// --------------------------------------------------------------------------
// -- add_data
// --   functions to add an observation to the niw. Updates the sufficient
// -- statistics, posterior hyperparameters, and predictive parameters
// --
// --   parameters:
// --     - data : the new observed data point of size [1 D]
// --------------------------------------------------------------------------
void niw::rem_data(arr(double) data)
{
   // update the sufficient stats and the N
   if (N<=0) mexErrMsgTxt("Removing from empty cluster!\n");
   N--;
   for (int d1=0; d1<D; d1++)
   {
      double temp_data = data[d1];
      t[d1] -= temp_data;
      for (int d2=0; d2<D; d2++)
         T[d1+d2*D] -= temp_data*data[d2];
   }
   update();
}
void niw::add_data_init(arr(double) data)
{
   // update the sufficient stats and the N
   N++;
   for (int d1=0; d1<D; d1++)
   {
      double temp_data = data[d1];
      t[d1] += temp_data;
      for (int d2=0; d2<D; d2++)
         T[d1+d2*D] += temp_data*data[d2];
   }
}
void niw::add_data(arr(double) data)
{
   add_data_init(data);
   update();
}
void niw::merge_with(niw* &other)
{
   if (other!=NULL)
   {
      N += other->N;
      for (int d=0; d<D; d++)
         t[d] += other->t[d];
      for (int d=0; d<D2; d++)
         T[d] += other->T[d];
      update(true);
   }
}
void niw::merge_with(niw* &other1, niw* &other2)
{
   if (other1==NULL && other2!=NULL)
      merge_with(other2);
   else if (other1!=NULL && other2==NULL)
      merge_with(other1);
   else if (other1!=NULL && other2!=NULL)
   {
      N += other1->N + other2->N;
      for (int d=0; d<D; d++)
         t[d] += other1->t[d] + other2->t[d];
      for (int d=0; d<D2; d++)
         T[d] += other1->T[d] + other2->T[d];
      update(true);
   }
}
void niw::merge_with(niw &other)
{
   N += other.N;
   for (int d=0; d<D; d++)
      t[d] += other.t[d];
   for (int d=0; d<D2; d++)
      T[d] += other.T[d];
   update(true);
}
void niw::merge_with(niw &other1, niw &other2)
{
   N += other1.N + other2.N;
   for (int d=0; d<D; d++)
      t[d] += other1.t[d] + other2.t[d];
   for (int d=0; d<D2; d++)
      T[d] += other1.T[d] + other2.T[d];
   update(true);
}


double niw::predictive_loglikelihood(arr(double) data) const
{
   // exact student-t
   // predictive_Z = myloggamma(0.5*(nu+1)) - myloggamma(0.5*nu) - 0.5*logDetCov - 0.5*D*log(nu-D+1);
   return predictive_Z - 0.5*(nu+1)*log(1 + (xmut_A_xmu(data,mean,prec,D))/(nu-D+1.0));

   // normal approximation
   return -0.5*xmut_A_xmu(data, mean, prec, D) - 0.5*logDetCov;
}


double niw::data_loglikelihood()
{
   return                          -0.5*N*D*logpi + mylogmgamma05(nu,D) - mylogmgamma05(nuh,D) + (0.5*nuh)*(D*log(nuh)+log(det(Deltah,D))) - (0.5*nu)*(D*log(nu)+log(det(Delta,D))) + 0.5*D*log(kappah/kappa);
//   return -0.5*N*D*1.144729885849 - 0.5*N*D*logpi + mylogmgamma05(nu,D) - mylogmgamma05(nuh,D) + (0.5*nuh)*(D*log(nuh)+log(det(Deltah,D))) - (0.5*nu)*(D*log(nu)+log(det(Delta,D))) + 0.5*D*log(kappah/kappa);
}

/*double niw::data_loglikelihood_sample_niw()
{
   arr(double) smean = allocate_memory<double>(D);
   arr(double) sprec = allocate_memory<double>(D2);

   double prob = 0;
   prob =

   double logdet = log(det(scov,D));
   prob += -0.5*(nuh+D+1)*logdet - 0.5*traceAxB(Deltah,sprec,D);



   deallocate_memory(smean);
   deallocate_memory(sprec);
}*/

void niw::sample()
{
   // do the cholesky decomposition
   Eigen::Map<Eigen::MatrixXd> sigma(Delta, D, D);
   Eigen::LLT<Eigen::MatrixXd> myLlt(sigma);
   Eigen::MatrixXd chol = myLlt.matrixL();

   for (int d=0; d<D; d++)
   {
      prec[d+d*D] = sqrt(gsl_ran_chisq(r, nu-d));
      for (int d2=d+1; d2<D; d2++)
      {
         prec[d2+d*D] = 0;
         prec[d+d2*D] = gsl_ran_gaussian(r, 1);
      }
   }

   // diag * cov^-1
   Eigen::Map<Eigen::MatrixXd> temp(prec, D, D);
   temp = sqrt(nu)*chol*temp.inverse();
   temp = temp*temp.transpose();

   // temp now contains the covariance
   myLlt.compute(temp);
   chol = myLlt.matrixL();

   // populate the mean
   for (int d=0; d<D; d++)
      mean[d] = gsl_ran_gaussian(r,1);
   Eigen::Map<Eigen::VectorXd> emean(mean, D);
   emean = chol*emean / sqrt(kappa);
   for (int d=0; d<D; d++)
      mean[d] += theta[d];

   logDetCov = log(det(prec,D));
   // convert to a precision
   temp = temp.inverse();

   logprior = log(N);

   // 0.5*pow(nuh,D+1)*log(det(Deltah)) -
   logmu = 0.5*(nuh+D+1)*logDetCov - 0.5*nuh*traceAxB(Deltah, prec, D);
   // log |cov/kappa| = log kappa^D |cov| = D*log(kappa) + logdetcov
   logmu += -0.5*logDetCov - 0.5*D*log(kappah) - 0.5*xmut_A_xmu(mean, theta, prec, D);
}
