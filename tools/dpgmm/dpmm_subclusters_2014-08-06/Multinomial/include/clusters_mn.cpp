// =============================================================================
// == clusters_mn.cpp
// == --------------------------------------------------------------------------
// == A class for all multinomial clusters with sub-clusters
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

#include "clusters_mn.h"
#include "reduction_array.h"
#include "reduction_array2.h"
#include "reduction_hash2.h"
#include "sample_categorical.h"

#ifndef pi
#define pi 3.14159265
#endif

// --------------------------------------------------------------------------
// -- clusters_mn
// --   constructor; initializes to empty
// --------------------------------------------------------------------------
clusters_mn::clusters_mn() :
   N(0), params(0), sticks(0), k2z(0), z2k(0), Nthreads(0), probsArr(0),
   rArr(0), superclusters(0), supercluster_labels(0), supercluster_labels_count(0),
   likelihoodOld(0), likelihoodDelta(0), splittable(0), randomSplitIndex(0)
{
   rand_gen = initialize_gsl_rand(rand());
   useSuperclusters = false;
   always_splittable = false;
}

// --------------------------------------------------------------------------
// -- clusters_mn
// --   copy constructor;
// --------------------------------------------------------------------------
clusters_mn::clusters_mn(const clusters_mn& that)
{
   copy(that);
}
// --------------------------------------------------------------------------
// -- operator=
// --   assignment operator
// --------------------------------------------------------------------------
clusters_mn& clusters_mn::operator=(const clusters_mn& that)
{
   if (this != &that)
   {
      if (N>0)
      {
         for (int i=0; i<N; i++)
            if (params[i]!=NULL)
               delete params[i];
      }
      if (Nthreads>0)
      {
         for (int t=0; t<Nthreads; t++)
            if (rArr[t]!=NULL) gsl_rng_free(rArr[t]);
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
void clusters_mn::copy(const clusters_mn& that)
{
   N = that.N;
   D = that.D;
   Nthreads = that.Nthreads;

   data = that.data;
   phi = that.phi;

   alpha = that.alpha;
   logalpha = log(alpha);
   params.assign(N,NULL);
   alive = that.alive;
   alive_ptrs.assign(N,NULL);
   
   randomSplitIndex.resize(N);

   for (int i=0; i<N; i++)
      if (that.params[i]!=NULL)
         params[i] = new cluster_sampledT<dir_sampled>(*(that.params[i]));

   sticks = that.sticks;
   k2z = that.k2z;
   z2k = that.z2k;
   likelihoodOld = that.likelihoodOld;
   likelihoodDelta = that.likelihoodDelta;
   splittable = that.splittable;

   linkedListNode<int>* node = alive.getFirst();
   while (node!=NULL)
   {
      alive_ptrs[node->getData()] = node;
      node = node->getNext();
   }

   rand_gen = initialize_gsl_rand(mx_rand());

   probsArr.resize(Nthreads);
   rArr.resize(Nthreads);
   for (int t=0; t<Nthreads; t++)
   {
      probsArr[t].resize(N);
      rArr[t] = initialize_gsl_rand(mx_rand());
   }

   useSuperclusters = that.useSuperclusters;
   always_splittable = that.always_splittable;
}

// --------------------------------------------------------------------------
// -- clusters_mn
// --   constructor; intializes to all the values given
// --------------------------------------------------------------------------
clusters_mn::clusters_mn(int _N, int _D, const mxArray* _data, arr(double) _phi,
   double _alpha, dir_sampled &_hyper, int _Nthreads, bool _useSuperclusters,
   bool _always_splittable) :
   N(_N), D(_D), data(_data), phi(_phi), alpha(_alpha), hyper(_hyper),
   Nthreads(_Nthreads), useSuperclusters(_useSuperclusters),
   superclusters(0), supercluster_labels(0), supercluster_labels_count(0),
   always_splittable(_always_splittable)
{
   params.assign(N,NULL);
   sticks.resize(N);
   k2z.assign(N,-1);
   z2k.assign(N,-1);
   alive_ptrs.assign(N,NULL);
   likelihoodOld.resize(N);
   likelihoodDelta.resize(N);
   splittable.resize(N);
   
   randomSplitIndex.resize(N);

   logalpha = log(alpha);

   rand_gen = initialize_gsl_rand(mx_rand());

   probsArr.resize(Nthreads);
   rArr.resize(Nthreads);
   for (int t=0; t<Nthreads; t++)
   {
      probsArr[t].resize(N);
      rArr[t] = initialize_gsl_rand(mx_rand());
   }
}


// --------------------------------------------------------------------------
// -- ~clusters_mn
// --   destructor
// --------------------------------------------------------------------------
clusters_mn::~clusters_mn()
{
   if (N>0)
   {
      for (int i=0; i<N; i++)
         if (params[i]!=NULL)
            delete params[i];
   }
   if (Nthreads>0)
   {
      for (int t=0; t<Nthreads; t++)
         if (rArr[t]!=NULL) gsl_rng_free(rArr[t]);
   }

   gsl_rng_free(rand_gen);
}


// --------------------------------------------------------------------------
// -- initialize
// --   populates the initial statistics
// --------------------------------------------------------------------------
void clusters_mn::initialize(const mxArray* cluster_params)
{
   // assume that the phi's are compressed between [0,K-1)
   K = mxGetNumberOfElements(cluster_params);
   for (int m=0; m<K; m++)
   {
      alive_ptrs[m] = alive.addNodeEnd(m);
      params[m] = new cluster_sampledT<dir_sampled>(hyper,alpha);

      sticks[m] = getInput<double>(mxGetField(cluster_params,m,"logpi"));
      double* logpi_mn = getArrayInput<double>(mxGetField(cluster_params,m,"logpi_mn"));
      double* logpi_mn_l = getArrayInput<double>(mxGetField(cluster_params,m,"logpi_mn_l"));
      double* logpi_mn_r = getArrayInput<double>(mxGetField(cluster_params,m,"logpi_mn_r"));
      likelihoodOld[m] = getInput<double>(mxGetField(cluster_params,m,"logsublikelihood"));
      likelihoodDelta[m] = getInput<double>(mxGetField(cluster_params,m,"logsublikelihoodDelta"));
      splittable[m] = getInput<bool>(mxGetField(cluster_params,m,"splittable"));

      params[m]->get_params()->set_multinomial(logpi_mn);
      params[m]->get_paramsl()->set_multinomial(logpi_mn_l);
      params[m]->get_paramsr()->set_multinomial(logpi_mn_r);
   }

   // calculate the initial statistics since they aren't explicitly stored
   reduction_array<int> NArr(Nthreads, K*2, 0);
   reduction_array<double> datatermsArr(Nthreads, K*2, 0);
   reduction_array<int> totalCountsArr(Nthreads, K*2, 0);
#ifdef USEFULL
   reduction_array2<int> countsArr(Nthreads, K*2, hyper.getD(), 0);
#else
   reduction_hash2<int> countsArr(Nthreads, K*2);
#endif

   #pragma omp parallel for schedule(dynamic)
   for (int i=0; i<N; i++)
   {
      // assume it's packed
      int m = phi[i];
      int proc = omp_get_thread_num();
      int bin = m*2 + (int)(2*(phi[i] - m));

      // accumulate the stats
      NArr.reduce_inc(proc, bin);
      long* colInds; double* colVals; long nnz;
      data.getColumn(i, colInds, colVals, nnz);
      countsArr.reduce_add(proc, bin, colInds, colVals, nnz);

      double new_dataterm = 0;
      int Ci = 0;
      for (int di=0; di<nnz; di++)
      {
         Ci += colVals[di];
         new_dataterm -= gammalnint(colVals[di]+1);
      }
      new_dataterm += gammalnint(Ci+1);
      totalCountsArr.reduce_add(proc, bin, Ci);
      datatermsArr.reduce_add(proc, bin, new_dataterm);
   }

   int* fNArr = NArr.final_reduce_add();
   double* fdatatermsArr = datatermsArr.final_reduce_add();
   int* ftotalCountsArr = totalCountsArr.final_reduce_add();
#ifdef USEFULL
   int* fcountsArr = countsArr.final_reduce_add();
#else
   unordered_map<int,int>* fcountsArr = countsArr.final_reduce_add();
#endif

   for (int k=0; k<K; k++)
   {
#ifdef USEFULL
      params[k]->get_paramsl()->set_stats(fNArr[k*2],   ftotalCountsArr[k*2],   fdatatermsArr[k*2],   fcountsArr+(k*2)*hyper.getD());
      params[k]->get_paramsr()->set_stats(fNArr[k*2+1], ftotalCountsArr[k*2+1], fdatatermsArr[k*2+1], fcountsArr+(k*2+1)*hyper.getD());
#else
      params[k]->get_paramsl()->set_stats(fNArr[k*2],   ftotalCountsArr[k*2],   fdatatermsArr[k*2],   fcountsArr[k*2]);
      params[k]->get_paramsr()->set_stats(fNArr[k*2+1], ftotalCountsArr[k*2+1], fdatatermsArr[k*2+1], fcountsArr[k*2+1]);
#endif
      params[k]->update_upwards_posterior();
   }
}

// --------------------------------------------------------------------------
// -- write_output
// --   creates and writes the output cluster structure
// --------------------------------------------------------------------------
void clusters_mn::write_output(mxArray* &plhs)
{
   linkedListNode<int>* node = alive.getFirst();
   populate_k2z_z2k();

   #pragma omp parallel for
   for (int i=0; i<N; i++)
   {
      int m = phi[i];
      int k = z2k[m];
      phi[i] += k-m;
   }

   const char* names[7] = {"logpi","logpi_mn","logpi_mn_l","logpi_mn_r","logsublikelihood","logsublikelihoodDelta","splittable"};
   plhs = mxCreateStructMatrix(K,1,7,names);
   mxArray* cluster_params = plhs;
   for (int k=0; k<K; k++)
   {
      int m = k2z[k];
      mxWriteField(cluster_params, k, "logpi", mxCreateScalar(sticks[m]));

      mxWriteField(cluster_params, k, "logpi_mn"   , mxCreateArray(D,1,params[m]->get_params()->get_logpi()));
      mxWriteField(cluster_params, k, "logpi_mn_l" , mxCreateArray(D,1,params[m]->get_paramsl()->get_logpi()));
      mxWriteField(cluster_params, k, "logpi_mn_r" , mxCreateArray(D,1,params[m]->get_paramsr()->get_logpi()));

      mxWriteField(cluster_params, k, "logsublikelihood",      mxCreateScalar(likelihoodOld[m]));
      mxWriteField(cluster_params, k, "logsublikelihoodDelta", mxCreateScalar(likelihoodDelta[m]));
      mxWriteField(cluster_params, k, "splittable",            mxCreateScalar(splittable[m]));
   }
}

// --------------------------------------------------------------------------
// -- populate_k2z_z2k
// --   Populates the k and z mappings
// --------------------------------------------------------------------------
void clusters_mn::populate_k2z_z2k()
{
   K = alive.getLength();
   linkedListNode<int>* node = alive.getFirst();
   for (int k=0; k<K; k++)
   {
      int m = node->getData();
      k2z[k] = m;
      z2k[m] = k;
      node = node->getNext();
   }
}

// --------------------------------------------------------------------------
// -- sample_params
// --   Samples the parameters for each cluster and the mixture weights
// --------------------------------------------------------------------------
void clusters_mn::sample_params()
{
   // populate the mapping and the stats for the dirichlet
   linkedListNode<int>* node = alive.getFirst();
   populate_k2z_z2k();

   double total = 0;
   #pragma omp parallel for reduction(+:total)
   for (int k=0; k<K; k++)
   {
      int proc = omp_get_thread_num();
      int m = k2z[k];
      sticks[m] = gsl_ran_gamma(rArr[proc], params[m]->getN(), 1);
      total += sticks[m];
   }

   // posterior
   // sample the cluster parameters and the gamma distributions
   //#pragma omp parallel for reduction(+:total)
   for (int k=0; k<K; k++)
   {
      int m = k2z[k];
      params[m]->sample_param(); // this is done in parallel instead of the outer loop
   }

   // store the log of the stick lenghts
   total = log(total);
   #pragma omp parallel for
   for (int k=0; k<K; k++)
   {
      int m = k2z[k];
      sticks[m] = log(sticks[m]) - total;
   }
}


// --------------------------------------------------------------------------
// -- sample_superclusters
// --   Samples the supercluster assignments
// --------------------------------------------------------------------------
void clusters_mn::sample_superclusters()
{
   if (!useSuperclusters)
      return;

   superclusters.assign(K,-1);
   supercluster_labels.assign(K*K,-1);
   supercluster_labels_count.assign(K,0);

   if (K<=3)
   {
      for (int k=0; k<K; k++)
      {
         superclusters[k] = 0;
         supercluster_labels[k] = k;
      }
      supercluster_labels_count[0] = K;
      return;
   }

   vector<double> adjMtx(K*K);
   #pragma omp parallel for
   for (int k=0; k<K*K; k++)
   {
      if (k%K!=k/K)
      {
         int k1 = k%K;
         int k2 = k/K;
         if (k1>k2)
         {
            int m1 = k2z[k1];
            int m2 = k2z[k2];
            adjMtx[k2*K+k1] = 1.0/exp(4*params[m1]->Jdivergence(params[m2]));
            //adjMtx[k2*K+k1] = pow(adjMtx[k2*K+k1],4);
            adjMtx[k1*K+k2] = adjMtx[k2*K+k1];
         }
      }
      else
         adjMtx[k] = 0;
   }

   vector<linkedList<int> > neighbors(K);
   double* tprobabilities = probsArr[0].data();
   for (int k=0; k<K; k++)
   {
      double totalProb = 0;
      for (int k2=0; k2<K; k2++)
      {
         double temp = adjMtx[k+k2*K] + adjMtx[k2+k*K];
         tprobabilities[k2] = temp;
         totalProb += temp;
      }
      int k2 = sample_categorical(tprobabilities, K, totalProb, rArr[0]);
      neighbors[k].addNodeEnd(k2);
      neighbors[k2].addNodeEnd(k);
   }

   vector<bool> done(K,false);
   int count = 0;
   for (int k=0; k<K; k++) if (!done[k])
   {
      int label_count = 0;
      done[k] = true;
      superclusters[k] = count;
      int* supercluster_labels_k = supercluster_labels.data()+count*K;
      supercluster_labels_k[label_count++] = (k);
      while (!neighbors[k].isempty())
      {
         int k1 = neighbors[k].popFirst();
         if (!done[k1])
         {
            done[k1] = true;
            superclusters[k1] = count;
            supercluster_labels_k[label_count++] = (k1);
            neighbors[k].merge_with(neighbors[k1]);
         }
      }
      supercluster_labels_count[count] = label_count;
      count++;
   }
}

// --------------------------------------------------------------------------
// -- sample_labels
// --   Samples the label assignments for each data point
// --------------------------------------------------------------------------
void clusters_mn::sample_labels()
{
   linkedListNode<int>* node;

   // =================================================================================
   // propose local label changes within superclusters
   // =================================================================================

   // temporary reduction array variables
   reduction_array<int> NArr(Nthreads, K*2, 0);
   reduction_array<int> totalCountsArr(Nthreads, K*2, 0);
   reduction_array<double> datatermsArr(Nthreads, K*2, 0);
   reduction_array<double> likelihoodArr(Nthreads, K, 0);
#ifdef USEFULL
   reduction_array2<int> countsArr(Nthreads, K*2, hyper.getD(), 0);
#else
   reduction_hash2<int> countsArr(Nthreads, K*2);
#endif

   // loop through points and sample labels
   int k0 = z2k[(int)phi[0]];
   //#pragma omp parallel for schedule(dynamic,100)
   #pragma omp parallel for schedule(dynamic)
   for (int i=0; i<N; i++)
   {
      int proc = omp_get_thread_num();
      double* tprobabilities = probsArr[proc].data();
      long* colInds; double* colVals; long nnz;
      data.getColumn(i, colInds, colVals, nnz);

      double tphi;
      double phii = phi[i];
      int k;
      // find the distribution over possible ones
      if (K==1)
      {
         k = k0;
      }
      else if (useSuperclusters)
      {
         double maxProb = -mxGetInf();
         int ki = z2k[(int)phii];
         int sci = superclusters[ki];
         for (int k2i=0; k2i<supercluster_labels_count[sci]; k2i++)
         {
            int k2 = supercluster_labels[sci*K + k2i];
            int m = k2z[k2];

            // find the probability of the data belonging to this component
            double prob = params[m]->get_params()->predictive_loglikelihood(colInds, colVals, nnz) + sticks[m];

            maxProb = max(maxProb, prob);
            tprobabilities[k2i] = prob;
         }

         // sample a new cluster label
         double totalProb = total_logcategorical(tprobabilities, supercluster_labels_count[sci], maxProb);
         int k2i = sample_logcategorical(tprobabilities, supercluster_labels_count[sci], maxProb, totalProb, rArr[proc]);
         k = supercluster_labels[sci*K+k2i];
      }
      else
      {
         double maxProb = -mxGetInf();
         for (int k2=0; k2<K; k2++)
         {
            int m = k2z[k2];

            // find the probability of the data belonging to this component
            double prob = params[m]->get_params()->predictive_loglikelihood(colInds, colVals, nnz) + sticks[m];

            maxProb = max(maxProb, prob);
            tprobabilities[k2] = prob;
         }

         // sample a new cluster label
         double totalProb = total_logcategorical(tprobabilities, K, maxProb);
         k = sample_logcategorical(tprobabilities, K, maxProb, totalProb, rArr[proc]);
      }
      int m = k2z[k];

      double loglikelihood;
      double logpl = params[m]->get_paramsl()->predictive_loglikelihood(colInds, colVals, nnz);
      double logpr = params[m]->get_paramsr()->predictive_loglikelihood(colInds, colVals, nnz);
      tphi = params[m]->sample_subcluster_label(logpl, logpr, rArr[proc], loglikelihood);

      // update changes
      likelihoodArr.reduce_add(proc, k, loglikelihood);

      // update stats
      phi[i] = m + tphi;
      
      // accumulate
      int bin = k*2 + (int)(tphi*2);
      NArr.reduce_inc(proc, bin);
      countsArr.reduce_add(proc, bin, colInds, colVals, nnz);

      double new_dataterm = 0;
      int Ci = 0;
      for (int di=0; di<nnz; di++)
      {
         Ci += colVals[di];
         new_dataterm -= gammalnint(colVals[di]+1);
      }
      new_dataterm += gammalnint(Ci+1);
      totalCountsArr.reduce_add(proc, bin, Ci);
      datatermsArr.reduce_add(proc, bin, new_dataterm);
   }

   // accumulate cluster statistics
   int* fNArr = NArr.final_reduce_add();
   double* fdatatermsArr = datatermsArr.final_reduce_add();
   double* flikelihood = likelihoodArr.final_reduce_add();
   int* ftotalCountsArr = totalCountsArr.final_reduce_add();
#ifdef USEFULL
   int* fcountsArr = countsArr.final_reduce_add();
#else
   unordered_map<int,int>* fcountsArr = countsArr.final_reduce_add();
#endif

   #pragma omp parallel for
   for (int k=0; k<K; k++)
   {
      int m = k2z[k];
      params[m]->empty();
#ifdef USEFULL
      params[m]->get_paramsl()->set_stats(fNArr[k*2],   ftotalCountsArr[k*2],   fdatatermsArr[k*2],   fcountsArr+(k*2)*hyper.getD());
      params[m]->get_paramsr()->set_stats(fNArr[k*2+1], ftotalCountsArr[k*2+1], fdatatermsArr[k*2+1], fcountsArr+(k*2+1)*hyper.getD());
#else
      params[m]->get_paramsl()->set_stats(fNArr[k*2],   ftotalCountsArr[k*2],   fdatatermsArr[k*2],   fcountsArr[k*2]);
      params[m]->get_paramsr()->set_stats(fNArr[k*2+1], ftotalCountsArr[k*2+1], fdatatermsArr[k*2+1], fcountsArr[k*2+1]);
#endif

      params[m]->update_upwards_posterior();
      //params[m]->sample_subclusters();

      double newlikelihoodDelta = (flikelihood[k]/params[m]->getN()) - likelihoodOld[m];
      if (newlikelihoodDelta<=0)// && likelihoodDelta[m]<0)
         splittable[m] = true;
      likelihoodDelta[m] = newlikelihoodDelta;
      likelihoodOld[m] = (flikelihood[k]/params[m]->getN());
   }

   // get rid of dead nodes
   bool deleted = false;
   node = alive.getFirst();

   while (node!=NULL)
   {
      int m = node->getData();
      node = node->getNext();
      if (params[m]->isempty())
      {
         alive.deleteNode(alive_ptrs[m]);
         alive_ptrs[m] = NULL;
         delete params[m];
         params[m] = NULL;
         deleted = true;
      }
   }
   if (deleted)
   {
      populate_k2z_z2k();
      // resample superclusters because the mapping is all broken
      sample_superclusters();
   }
}

// DO NOT USE
// THIS FUNCTION DOES NOT PRESERVE THE CORRECT STATIONARY DISTRIBUTION
// --------------------------------------------------------------------------
// -- propose_merges
// --   Samples the label assignments for each data point
// --------------------------------------------------------------------------
void clusters_mn::propose_merges()
{
   vector<int> merge_with(K, -1);
   int numMerges = 0;
   for (int km=0; km<K; km++) if (splittable[k2z[km]])
   {
      int zm = k2z[km];
      if (merge_with[km]<0)
         for (int kn=km+1; kn<K; kn++) if (splittable[k2z[kn]])
         {
            int zn = k2z[kn];
            if ((!useSuperclusters || superclusters[km]==superclusters[kn]) && merge_with[km]<0 && merge_with[kn]<0)
            {
               int Nm = params[zm]->getN();
               int Nn = params[zn]->getN();
               int Nkh = Nm+Nn;

               cluster_sampledT<dir_sampled>* paramsm = params[zm];
               cluster_sampledT<dir_sampled>* paramsn = params[zn];

               double HR = -logalpha - gammalnint(Nm) - gammalnint(Nn) + gammalnint(Nkh) - Nkh*log(2.0);
               HR += paramsm->data_loglikelihood_marginalized_testmerge(paramsn);
               HR += -paramsm->data_loglikelihood_marginalized() - paramsn->data_loglikelihood_marginalized();

               if ((HR>0 || my_rand(rand_gen) < exp(HR)))
               {
                  merge_with[km] = km;
                  merge_with[kn] = km;
                  numMerges++;

                  // move on to the next one
                  break;
               }
            }
         }
   }
   if (numMerges>0)
   {
      //mexPrintf("NumMerges=%d\n",numMerges);
      // fix the phi's
      #pragma omp parallel for
      for (int i=0; i<N; i++)
      {
         int zi = phi[i];
         int ki = z2k[zi];
         if (merge_with[ki]>=0)
            phi[i] = k2z[merge_with[ki]] + ((merge_with[ki]==ki) ? 0.25 : 0.75);
      }
      for (int kn=0; kn<K; kn++) if (merge_with[kn]>=0 && merge_with[kn]!=kn)
      {
         int zn = k2z[kn];
         int km = merge_with[kn];
         int zm = k2z[km];

         // mark them as done
         merge_with[km] = -1;
         merge_with[kn] = -1;

         splittable[zm] = false;
         likelihoodOld[zm] = -mxGetInf();
         likelihoodDelta[zm] = mxGetInf();

         params[zm]->merge_with(params[zn]);
         params[zm]->setstickl(sticks[zm]);
         params[zm]->setstickr(sticks[zn]);

         sticks[zm] = logsumexp(sticks[zm], sticks[zn]);
         sticks[zn] = -mxGetInf();

         // sample a new set of parameters for the highest level
         //params[zm]->sample_highest();

         alive.deleteNode(alive_ptrs[zn]);
         alive_ptrs[zn] = NULL;
         delete params[zn];
         params[zn] = NULL;
         numMerges--;
         if (numMerges==0)
            break;
      }
      populate_k2z_z2k();
   }
}



// --------------------------------------------------------------------------
// -- propose_splits
// --   Samples the label assignments for each data point
// --------------------------------------------------------------------------
void clusters_mn::propose_splits()
{
   vector<bool> do_split(K, false);
   int num_splits = 0;
   //#pragma omp parallel for reduction(+:num_splits)
   for (int k=0; k<K; k++)
   {
      // check to see if the subclusters have converged
      if (always_splittable || splittable[k2z[k]])//paramsChanges[k2z[k]]/((double)params[k2z[k]]->getN())<=0.1)
      {
         int proc = omp_get_thread_num();
         int kz = k2z[k];
         cluster_sampledT<dir_sampled>* paramsk = params[kz];
         int Nk = paramsk->getN();
         int Nmh = paramsk->getlN();
         int Nnh = paramsk->getrN();

         // full ratios
         double HR = logalpha + gammalnint(Nmh) + gammalnint(Nnh) - gammalnint(Nk);
         HR += paramsk->data_lloglikelihood_marginalized() + paramsk->data_rloglikelihood_marginalized();
         HR += -paramsk->data_loglikelihood_marginalized();

         if ((HR>0 || my_rand(rArr[proc]) < exp(HR)))
         {
            do_split[k] = true;
            num_splits++;
         }
      }
   }

   if (num_splits>0)
   {
      int temp_num_splits = num_splits;
      // figure out which labels to split into and prepare simple split stuff
      vector<int> split_into(K);
      vector<int> split_into_k(K);
      int k = 0;
      for (k; k<K; k++)
         if (do_split[k])
            break;
      for (int nzh=0; nzh<N; nzh++)
      {
         if (params[nzh]==NULL)
         {
            // found an empty one
            int kz = k2z[k];
            split_into[k] = nzh;
            split_into_k[k] = K+(num_splits-temp_num_splits);

            // mark it as unsplittable for now
            splittable[kz] = false;
            likelihoodOld[kz] = -mxGetInf();
            likelihoodDelta[kz] = mxGetInf();
            splittable[nzh] = false;
            likelihoodOld[nzh] = -mxGetInf();
            likelihoodDelta[nzh] = mxGetInf();

            alive_ptrs[nzh] = alive.addNodeEnd(nzh);
            params[nzh] = new cluster_sampledT<dir_sampled>(hyper,alpha);
            temp_num_splits--;

            // sample a new set of pi's
            sticks[kz] = params[kz]->getstickl();
            sticks[nzh] = params[kz]->getstickr();

            // transfer over the theta's
            params[nzh]->split_from(params[kz]);
            params[kz]->split_fix();
            params[nzh]->sample_param_subclusters();
            params[kz]->sample_param_subclusters();

            // do the next ones if there are any
            if (temp_num_splits>0)
            {
               for (k=k+1; k<K; k++)
                  if (do_split[k])
                     break;
            }
            else
               break;
         }
      }

      // correct the labels
      // temporary reduction array variables
      reduction_array<int> NArr(Nthreads, (K+num_splits)*2, 0);
      reduction_array<int> totalCountsArr(Nthreads, (K+num_splits)*2, 0);
      reduction_array<double> datatermsArr(Nthreads, (K+num_splits)*2, 0);
#ifdef USEFULL
      reduction_array2<int> countsArr(Nthreads, (K+num_splits)*2, hyper.getD(), 0);
#else
      reduction_hash2<int> countsArr(Nthreads, (K+num_splits)*2);
#endif


      #pragma omp parallel for schedule(dynamic)
      for (int i=0; i<N; i++) if (do_split[z2k[(int)(phi[i])]])
      {
         int proc = omp_get_thread_num();
         long* colInds; double* colVals; long nnz;
         data.getColumn(i, colInds, colVals, nnz);
	 
         int kz = phi[i];
         int newk;
         if (phi[i]-kz<0.5)
         {
            phi[i] = kz;
            newk = z2k[kz];
         }
         else
         {
            phi[i] = split_into[z2k[kz]];
            newk = split_into_k[z2k[kz]];
         }
         double tphi = (my_rand(rArr[proc])<0.5) ? 0.25 : 0.75;
         phi[i] += tphi;

         // updates stats
         int bin = newk*2 + (int)(tphi*2);
         NArr.reduce_inc(proc, bin);
         countsArr.reduce_add(proc, bin, colInds, colVals, nnz);

         double new_dataterm = 0;
         int Ci = 0;
         for (int di=0; di<nnz; di++)
         {
            Ci += colVals[di];
            new_dataterm -= gammalnint(colVals[di]+1);
         }
         totalCountsArr.reduce_add(proc, bin, Ci);
         new_dataterm += gammalnint(Ci+1);
         datatermsArr.reduce_add(proc, bin, new_dataterm);
      }
      
      // accumulate cluster statistics
      int* fNArr = NArr.final_reduce_add();
      int* ftotalCountsArr = totalCountsArr.final_reduce_add();
      double* fdatatermsArr = datatermsArr.final_reduce_add();
#ifdef USEFULL
      int* fcountsArr = countsArr.final_reduce_add();
#else
      unordered_map<int,int>* fcountsArr = countsArr.final_reduce_add();
#endif

      #pragma omp parallel for
      for (int k=0; k<K; k++) if (do_split[k])
      {
         int m = k2z[k];
         params[m]->empty();
#ifdef USEFULL
         params[m]->get_paramsl()->set_stats(fNArr[k*2],   ftotalCountsArr[k*2],   fdatatermsArr[k*2],   fcountsArr+(k*2)*hyper.getD());
         params[m]->get_paramsr()->set_stats(fNArr[k*2+1], ftotalCountsArr[k*2+1], fdatatermsArr[k*2+1], fcountsArr+(k*2+1)*hyper.getD());
#else
         params[m]->get_paramsl()->set_stats(fNArr[k*2],   ftotalCountsArr[k*2],   fdatatermsArr[k*2],   fcountsArr[k*2]);
         params[m]->get_paramsr()->set_stats(fNArr[k*2+1], ftotalCountsArr[k*2+1], fdatatermsArr[k*2+1], fcountsArr[k*2+1]);
#endif
         params[m]->update_upwards_posterior();

         m = split_into[k];
         int newk = split_into_k[k];
         params[m]->empty();
#ifdef USEFULL
         params[m]->get_paramsl()->set_stats(fNArr[newk*2],   ftotalCountsArr[newk*2],   fdatatermsArr[newk*2],   fcountsArr+(newk*2)*hyper.getD());
         params[m]->get_paramsr()->set_stats(fNArr[newk*2+1], ftotalCountsArr[newk*2+1], fdatatermsArr[newk*2+1], fcountsArr+(newk*2+1)*hyper.getD());
#else
         params[m]->get_paramsl()->set_stats(fNArr[newk*2],   ftotalCountsArr[newk*2],   fdatatermsArr[newk*2],   fcountsArr[newk*2]);
         params[m]->get_paramsr()->set_stats(fNArr[newk*2+1], ftotalCountsArr[newk*2+1], fdatatermsArr[newk*2+1], fcountsArr[newk*2+1]);
#endif
         params[m]->update_upwards_posterior();
      }

      for (int k=0; k<K; k++) if (do_split[k])
      {
         params[k2z[k]]->sample_param_subclusters();
         params[split_into[k]]->sample_param_subclusters();
      }

      populate_k2z_z2k();
   }
}


void clusters_mn::propose_random_split_assignments()
{
   // temporary reduction array variables
   reduction_array<int> randomNArr(Nthreads, K*2, 0);
   reduction_array<int> randomtotalCountsArr(Nthreads, K*2, 0);
   reduction_array<double> randomdatatermsArr(Nthreads, K*2, 0);
#ifdef USEFULL
   reduction_array2<int> randomcountsArr(Nthreads, K*2, hyper.getD(), 0);
#else
   reduction_hash2<int> randomcountsArr(Nthreads, K*2);
#endif


   // sample new pi's for the splits
   // store the actual stick breaks for now... will convert to log later
   #pragma omp parallel for
   for (int k=0; k<K; k++)
   {
      int m = k2z[k];
      double stickm = gsl_ran_gamma(rArr[omp_get_thread_num()], 0.5*alpha, 1);
      double stickn = gsl_ran_gamma(rArr[omp_get_thread_num()], 0.5*alpha, 1);
      double total = stickm + stickn;
      stickm /= total;
      stickn /= total;
      params[m]->setrandomstickl(stickm);
      params[m]->setrandomstickr(stickn);
   }

   // loop through points and sample labels
   #pragma omp parallel for schedule(dynamic)
   for (int i=0; i<N; i++)
   {
      int proc = omp_get_thread_num();
      long* colInds; double* colVals; long nnz;
      data.getColumn(i, colInds, colVals, nnz);
      int m = (int)phi[i];
      int k = z2k[m];

      double new_dataterm = 0;
      int Ci = 0;
      for (int di=0; di<nnz; di++)
      {
         Ci += colVals[di];
         new_dataterm -= gammalnint(colVals[di]+1);
      }
      new_dataterm += gammalnint(Ci+1);
      
      //int ri = (rand()%2);
      int ri = my_rand(rArr[proc]) < params[m]->getrandomstickl();
      randomSplitIndex[i] = ri;
      
      // accumulate random stuff
      int bin = k*2 + (int)ri;
      randomNArr.reduce_inc(proc,bin);
      randomtotalCountsArr.reduce_add(proc,bin, Ci);
      randomcountsArr.reduce_add(proc, bin, colInds, colVals, nnz);
      randomdatatermsArr.reduce_add(proc, bin, new_dataterm);
   }

   // accumulate cluster statistics
   int* frandomNArr = randomNArr.final_reduce_add();
   int* frandomtotalCountsArr = randomtotalCountsArr.final_reduce_add();
   double* frandomdatatermsArr = randomdatatermsArr.final_reduce_add();
#ifdef USEFULL
   int* frandomcountsArr = randomcountsArr.final_reduce_add();
#else
   unordered_map<int,int>* frandomcountsArr = randomcountsArr.final_reduce_add();
#endif
   
   #pragma omp parallel for
   for (int k=0; k<K; k++)
   {
      int m = k2z[k];
#ifdef USEFULL
      params[m]->get_randoml()->set_stats(frandomNArr[k*2],   frandomtotalCountsArr[k*2],   frandomdatatermsArr[k*2],   frandomcountsArr+(k*2)*hyper.getD());
      params[m]->get_randomr()->set_stats(frandomNArr[k*2+1], frandomtotalCountsArr[k*2+1], frandomdatatermsArr[k*2+1], frandomcountsArr+(k*2+1)*hyper.getD());
#else
      params[m]->get_randoml()->set_stats(frandomNArr[k*2],   frandomtotalCountsArr[k*2],   frandomdatatermsArr[k*2],   frandomcountsArr[k*2]);
      params[m]->get_randomr()->set_stats(frandomNArr[k*2+1], frandomtotalCountsArr[k*2+1], frandomdatatermsArr[k*2+1], frandomcountsArr[k*2+1]);
#endif
      // convert to log
      params[m]->setrandomstickl( log(params[m]->getrandomstickl()) );
      params[m]->setrandomstickr( log(params[m]->getrandomstickr()) );
   }
}


void clusters_mn::propose_random_splits()
{
   propose_random_split_assignments();

   vector<bool> do_split(K, false);
   int num_splits = 0;
   //#pragma omp parallel for reduction(+:num_splits)
   for (int k=0; k<K; k++)
   {
      int proc = omp_get_thread_num();
      int kz = k2z[k];
      cluster_sampledT<dir_sampled>* paramsk = params[kz];
      int Nk = paramsk->getN();

      // full ratios
      double HR = logalpha + 2*gsl_sf_lngamma(0.5*alpha) - gsl_sf_lngamma(alpha) + gsl_sf_lngamma(alpha + Nk) - gammalnint(Nk) + log(100.0);
      HR += paramsk->data_randomlloglikelihood_marginalized() + paramsk->data_randomrloglikelihood_marginalized();
      HR += -paramsk->data_loglikelihood_marginalized();

      if ((HR>0 || my_rand(rArr[proc]) < exp(HR)) && paramsk->getrandomlN()>0 && paramsk->getrandomrN()>0)
      {
         do_split[k] = true;
         num_splits++;
      }
   }
   if (num_splits>0)
   {
      //mexPrintf("NumRandomSplits=%d\n", num_splits);drawnow();
      int temp_num_splits = num_splits;
      // figure out which labels to split into and prepare simple split stuff
      vector<int> split_into(K);
      vector<int> split_into_k(K);
      int k = 0;
      for (k; k<K; k++)
         if (do_split[k])
            break;
      for (int nzh=0; nzh<N; nzh++)
      {
         if (params[nzh]==NULL)
         {
            // found an empty one
            int kz = k2z[k];
            split_into[k] = nzh;
            split_into_k[k] = K+(num_splits-temp_num_splits);

            // mark it as unsplittable for now
            splittable[kz] = false;
            likelihoodOld[kz] = -mxGetInf();
            likelihoodDelta[kz] = mxGetInf();
            splittable[nzh] = false;
            likelihoodOld[nzh] = -mxGetInf();
            likelihoodDelta[nzh] = mxGetInf();

            alive_ptrs[nzh] = alive.addNodeEnd(nzh);
            params[nzh] = new cluster_sampledT<dir_sampled>(hyper,alpha);
            temp_num_splits--;

            // sample a new set of pi's
            sticks[kz] = params[kz]->getrandomstickl();
            sticks[nzh] = params[kz]->getrandomstickr();

            // transfer over the theta's
            params[nzh]->split_from_random(params[kz]);
            params[kz]->split_fix_random();
            params[nzh]->sample_param_subclusters();
            params[kz]->sample_param_subclusters();

            // do the next ones if there are any
            if (temp_num_splits>0)
            {
               for (k=k+1; k<K; k++)
                  if (do_split[k])
                     break;
            }
            else
               break;
         }
      }

      // correct the labels
      // temporary reduction array variables
      reduction_array<int> NArr(Nthreads, (K+num_splits)*2, 0);
      reduction_array<int> totalCountsArr(Nthreads, (K+num_splits)*2, 0);
      reduction_array<double> datatermsArr(Nthreads, (K+num_splits)*2, 0);
#ifdef USEFULL
      reduction_array2<int> countsArr(Nthreads, (K+num_splits)*2, hyper.getD(), 0);
#else
      reduction_hash2<int> countsArr(Nthreads, (K+num_splits)*2);
#endif


      #pragma omp parallel for schedule(dynamic)
      for (int i=0; i<N; i++) if (do_split[z2k[(int)(phi[i])]])
      {
         int proc = omp_get_thread_num();
         long* colInds; double* colVals; long nnz;
         data.getColumn(i, colInds, colVals, nnz);

         int kz = phi[i];
         int newk;
         if (!randomSplitIndex[i])
         {
            phi[i] = kz;
            newk = z2k[kz];
         }
         else
         {
            phi[i] = split_into[z2k[kz]];
            newk = split_into_k[z2k[kz]];
         }
         double tphi = (my_rand(rArr[proc])<0.5) ? 0.25 : 0.75;
         phi[i] += tphi;

         // updates stats
         int bin = newk*2 + (int)(tphi*2);
         NArr.reduce_inc(proc, bin);
         countsArr.reduce_add(proc, bin, colInds, colVals, nnz);

         double new_dataterm = 0;
         int Ci = 0;
         for (int di=0; di<nnz; di++)
         {
            Ci += colVals[di];
            new_dataterm -= gammalnint(colVals[di]+1);
         }
         totalCountsArr.reduce_add(proc, bin, Ci);
         new_dataterm += gammalnint(Ci+1);
         datatermsArr.reduce_add(proc, bin, new_dataterm);
      }

      // accumulate cluster statistics
      int* fNArr = NArr.final_reduce_add();
      int* ftotalCountsArr = totalCountsArr.final_reduce_add();
      double* fdatatermsArr = datatermsArr.final_reduce_add();
#ifdef USEFULL
      int* fcountsArr = countsArr.final_reduce_add();
#else
      unordered_map<int,int>* fcountsArr = countsArr.final_reduce_add();
#endif

      #pragma omp parallel for
      for (int k=0; k<K; k++) if (do_split[k])
      {
         int m = k2z[k];
         params[m]->empty();
#ifdef USEFULL
         params[m]->get_paramsl()->set_stats(fNArr[k*2],   ftotalCountsArr[k*2],   fdatatermsArr[k*2],   fcountsArr+(k*2)*hyper.getD());
         params[m]->get_paramsr()->set_stats(fNArr[k*2+1], ftotalCountsArr[k*2+1], fdatatermsArr[k*2+1], fcountsArr+(k*2+1)*hyper.getD());
#else
         params[m]->get_paramsl()->set_stats(fNArr[k*2],   ftotalCountsArr[k*2],   fdatatermsArr[k*2],   fcountsArr[k*2]);
         params[m]->get_paramsr()->set_stats(fNArr[k*2+1], ftotalCountsArr[k*2+1], fdatatermsArr[k*2+1], fcountsArr[k*2+1]);
#endif
         params[m]->update_upwards_posterior();

         m = split_into[k];
         int newk = split_into_k[k];
         params[m]->empty();
#ifdef USEFULL
         params[m]->get_paramsl()->set_stats(fNArr[newk*2],   ftotalCountsArr[newk*2],   fdatatermsArr[newk*2],   fcountsArr+(newk*2)*hyper.getD());
         params[m]->get_paramsr()->set_stats(fNArr[newk*2+1], ftotalCountsArr[newk*2+1], fdatatermsArr[newk*2+1], fcountsArr+(newk*2+1)*hyper.getD());
#else
         params[m]->get_paramsl()->set_stats(fNArr[newk*2],   ftotalCountsArr[newk*2],   fdatatermsArr[newk*2],   fcountsArr[newk*2]);
         params[m]->get_paramsr()->set_stats(fNArr[newk*2+1], ftotalCountsArr[newk*2+1], fdatatermsArr[newk*2+1], fcountsArr[newk*2+1]);
#endif
         params[m]->update_upwards_posterior();
      }

      for (int k=0; k<K; k++) if (do_split[k])
      {
         params[k2z[k]]->sample_param_subclusters();
         params[split_into[k]]->sample_param_subclusters();
      }
      populate_k2z_z2k();


      // make sure none of them are empty
      // none of them should be, but for some reason, once in awhile it is
      int numDeleted = 0;
      for (int k=0; k<K; k++)
      {
         int m = k2z[k];
         if (params[m]->getN()==0)
         {
            sticks[m] = -mxGetInf();
            alive.deleteNode(alive_ptrs[m]);
            alive_ptrs[m] = NULL;
            delete params[m];
            params[m] = NULL;
            numDeleted++;
         }
      }
      if (numDeleted>0)
         populate_k2z_z2k();
   }
}

void clusters_mn::propose_random_merges()
{
   vector<double> data_loglikelihood_marginalized(K);
   for (int k=0; k<K; k++)
      data_loglikelihood_marginalized[k] = params[k2z[k]]->data_loglikelihood_marginalized();

   vector<int> merge_with(K, -1);
   int numMerges = 0;
   for (int km=0; km<K; km++)
   {
      int zm = k2z[km];
      if (merge_with[km]<0)
         for (int kn=km+1; kn<K; kn++)
         {
            int zn = k2z[kn];
            if ((!useSuperclusters || superclusters[km]==superclusters[kn]) && merge_with[km]<0 && merge_with[kn]<0)
            {
               cluster_sampledT<dir_sampled>* paramsm = params[zm];
               cluster_sampledT<dir_sampled>* paramsn = params[zn];

               int Nm = paramsm->getN();
               int Nn = paramsn->getN();
               int Nkh = Nm+Nn;

               double HR = -logalpha + gsl_sf_lngamma(alpha) - 2*gsl_sf_lngamma(0.5*alpha) + gammalnint(Nkh) - gsl_sf_lngamma(Nkh+alpha) - log(100.0);
               HR += paramsm->data_loglikelihood_marginalized_testmerge(paramsn);
               HR += -data_loglikelihood_marginalized[km] - data_loglikelihood_marginalized[kn];

               if ((HR>0 || my_rand(rand_gen) < exp(HR)))
               {
                  merge_with[km] = km;
                  merge_with[kn] = km;
                  numMerges++;

                  // move on to the next one
                  break;
               }
            }
         }
   }
   if (numMerges>0)
   {
      //mexPrintf("NumRandomMerges=%d\n",numMerges);
      // fix the phi's
      #pragma omp parallel for
      for (int i=0; i<N; i++)
      {
         int zi = phi[i];
         int ki = z2k[zi];
         if (merge_with[ki]>=0)
            phi[i] = k2z[merge_with[ki]] + ((merge_with[ki]==ki) ? 0.25 : 0.75);
      }
      for (int kn=0; kn<K; kn++) if (merge_with[kn]>=0 && merge_with[kn]!=kn)
      {
         int zn = k2z[kn];
         int km = merge_with[kn];
         int zm = k2z[km];

         // mark them as done
         merge_with[km] = -1;
         merge_with[kn] = -1;

         splittable[zm] = false;
         likelihoodOld[zm] = -mxGetInf();
         likelihoodDelta[zm] = mxGetInf();
         params[zm]->merge_with(params[zn]);
         params[zm]->setstickl(sticks[zm]);
         params[zm]->setstickr(sticks[zn]);

         sticks[zm] = logsumexp(sticks[zm], sticks[zn]);
         sticks[zn] = -mxGetInf();

         // sample a new set of parameters for the highest level
         //params[zm]->sample_highest();

         alive.deleteNode(alive_ptrs[zn]);
         alive_ptrs[zn] = NULL;
         delete params[zn];
         params[zn] = NULL;
         numMerges--;
         if (numMerges==0)
            break;
      }
      //mexPrintf("aa\n");drawnow();
      populate_k2z_z2k();
   }
}

double clusters_mn::joint_loglikelihood()
{
   linkedListNode<int>* node = alive.getFirst();
   double loglikelihood = gsl_sf_lngamma(alpha) - gsl_sf_lngamma(alpha+N);
   while (node!=NULL)
   {
      int m = node->getData();
      if (!params[m]->isempty())
         loglikelihood += logalpha + gsl_sf_lngamma(params[m]->getN()) + params[m]->data_loglikelihood_marginalized();
      node = node->getNext();
   }
   return loglikelihood;
}


