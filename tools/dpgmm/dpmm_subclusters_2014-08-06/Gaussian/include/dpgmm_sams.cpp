// =============================================================================
// == dpgmm_sams.cpp
// == --------------------------------------------------------------------------
// == The MEX interface file to sample a DP Gaussian Mixture Model with
// == the sequentially allocate merge-split
// == --------------------------------------------------------------------------
// == Copyright 2013. MIT. All Rights Reserved.
// == Written by Jason Chang 11-03-2013
// == --------------------------------------------------------------------------
// == If this code is used, the following should be cited:
// == 
// == [1] J. Chang and J. W. Fisher II, "Parallel Sampling of DP Mixtures
// ==     Models using Sub-Cluster Splits". Neural Information Processing
// ==     Systems (NIPS 2013), Lake Tahoe, NV, USA, Dec 2013.
// == [2] D. B. Dahl. An Improved Merge-Split Sampler for Conjugate Dirichlet
// ==     Process Mixture Models. Technical Report, University of Wisconsin -
// ==     Madison Dept. of Statistics, 2003.
// =============================================================================

#include "mex.h"
#include <iostream>
#include <math.h>
#include <algorithm>
#include <time.h>
#include "debugMEX.h"
#include "helperMEX.h"
#include "matrix.h"
#include "linkedList.cpp"
#include "linear_algebra.h"
#include "gsl/gsl_rng.h"
#include "gsl/gsl_randist.h"
#include "gsl/gsl_permutation.h"
#include "gsl/gsl_cdf.h"
#include "myfuncs.h"
#include "sample_categorical.h"
#include "niw.h"
#include "cluster_single.h"
#include "stopwatch.h"

#define NUMARGS 3
#define NUMOUT 1


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
   // Check for proper number of arguments
   if (nrhs != NUMARGS) {
         mexErrMsgTxt("Incorrect number of input arguments required.");
   } else if (nlhs > NUMOUT) {
         mexErrMsgTxt("Too many output arguments expected.");
   }

   checkInput(prhs[0], "double"); //x
   checkInput(prhs[1], VECTOR,  "uint32"); //z
   checkInput(prhs[2], STRUCT); // params

   const int *dims = mxGetDimensions(prhs[0]);
   int N = mxGetNumberOfElements(prhs[0]);
   int D = 1;
   if (mxGetNumberOfDimensions(prhs[0])>1)
      D = dims[0];
   N /= D;
   int D2 = D*D;

   arr(double) x = getArrayInput<double>(prhs[0]);
   arr(unsigned int) z = getArrayInput<unsigned int>(prhs[1]);

   const mxArray* params = prhs[2];
   double logalphah = log(getInput<double>(getField(params,0,"alpha")));
   double kappah = getInput<double>(getField(params,0,"kappa"));
   double nuh = getInput<double>(getField(params,0,"nu"));
   arr(double) thetah = getArrayInput<double>(getField(params,0,"theta"));
   arr(double) deltah = getArrayInput<double>(getField(params,0,"delta"));
   int its_crp = getInput<double>(getField(params,0,"its_crp"));
   int its_ms = getInput<double>(getField(params,0,"its_ms"));
   int its = max(its_crp, its_ms);

   niw hyper(true, D, kappah, nuh, thetah, deltah);
   hyper.update();

   arr(cluster_single*) clusters = allocate_memory<cluster_single*>(N, NULL);
   arr(linkedListNode<int>*) alive_ptrs = allocate_memory< linkedListNode<int>* >(N,NULL);
   linkedList<int> alive;
   arr(double) probabilities = allocate_memory<double>(N);

   // populate initial statistics for left and right halves
   for (int i=0; i<N; i++)
   {
      int m = z[i];
      if (clusters[m]==NULL)
      {
         alive_ptrs[m] = alive.addNodeEnd(m);
         clusters[m] = new cluster_single(hyper);
      }
      clusters[m]->add_data_init(x+i*D);
   }

   // populate the statistics
   linkedListNode<int>* node = alive.getFirst();
   while (node!=NULL)
   {
      int m = node->getData();
      clusters[m]->update_upwards();
      node = node->getNext();
   }


   gsl_rng *r = initialize_gsl_rand(mx_rand());
   const size_t count_t = N;
   const gsl_rng_type *T;
   gsl_permutation *perm = gsl_permutation_alloc(count_t);
   T = gsl_rng_default;
   gsl_permutation_init(perm);


   arr(int) indices = allocate_memory<int>(N);

   stopwatch timer;
   timer.tic();
   for (int it=0; it<its; it++)
   {
      if (it<its_crp)
      {
         gsl_ran_shuffle(r, perm->data, count_t, sizeof(size_t));
         for (int permi=0; permi<N; permi++)
         {
            int i = perm->data[permi];
            i = permi;

            // remove it from the previous component
            int m = z[i];
            clusters[m]->rem_data(x+i*D);
            if (clusters[m]->isempty())
            {
               alive.deleteNode(alive_ptrs[m]);
               alive_ptrs[m] = NULL;
               delete clusters[m];
               clusters[m] = NULL;
            }

            // find the distribution over possible ones
            int j = 0;
            node = alive.getFirst();
            double maxProb = -mxGetInf();
            while (node!=NULL)
            {
               m = node->getData();

               // find the probability of the data belonging to this component
               double prob = clusters[m]->predictive_loglikelihood(x+i*D) + logint(clusters[m]->getN());
               maxProb = max(maxProb, prob);
               probabilities[j++] = prob;
               node = node->getNext();
            }

            // new component
            probabilities[j] = hyper.predictive_loglikelihood(x+i*D) + logalphah;
            maxProb = max(maxProb, probabilities[j]);

            double totalProb = convert_logcategorical(probabilities, j+1, maxProb);
            int k = sample_categorical(probabilities, j+1, totalProb, r);

            // find the m
            if (k==j) // new one
            {
               for (m=0; m<N; m++)
                  if (clusters[m]==NULL)
                     break;
               alive_ptrs[m] = alive.addNodeEnd(m);
               clusters[m] = new cluster_single(hyper);
            }
            else
            {
               node = alive.getFirst();
               for (int l=0; l<k; l++)
                  node = node->getNext();
               m = node->getData();
            }

            // update stats
            z[i] = m;
            clusters[m]->add_data(x+i*D);
         }
      }


      if (it<its_ms)
      {
         // select two random points
         int i = my_randi(r)%N;
         int j = my_randi(r)%(N-1);
         if (j>i) j = j+1;

         int mi = z[i];
         int mj = z[j];

         if (mi==mj) // propose a split
         {
            cluster_single* si = new cluster_single(hyper);
            cluster_single* sj = new cluster_single(hyper);
            linkedList<int> si_indices, sj_indices;
            si->add_data(x+i*D);
            sj->add_data(x+j*D);
            si_indices.addNode(i);
            sj_indices.addNode(j);
            int Nij = 0;
            for (int ind=0; ind<N; ind++)
               if (z[ind]==mi)
                  indices[Nij++] = ind;

            // propose a new split
            double prob_split = 0;
            gsl_permutation *perm = gsl_permutation_alloc((size_t)Nij);
            gsl_permutation_init(perm);
            gsl_ran_shuffle(r, perm->data, Nij, sizeof(size_t));
            for (int permi=0; permi<Nij; permi++)
            {
               int ind = indices[permi];
               if (ind!=i && ind!=j)
               {
                  probabilities[0] = si->predictive_loglikelihood(x+ind*D) + logint(si->getN());
                  probabilities[1] = sj->predictive_loglikelihood(x+ind*D) + logint(sj->getN());

                  double totalProb = convert_logcategorical(probabilities, 2);
                  int ij = sample_categorical(probabilities, 2, totalProb, r);

                  if (ij==0)
                  {
                     prob_split += log(probabilities[0] / totalProb);
                     si->add_data(x+ind*D);
                     si_indices.addNode(ind);
                  }
                  else
                  {
                     prob_split += log(probabilities[1] / totalProb);
                     sj->add_data(x+ind*D);
                     sj_indices.addNode(ind);
                  }
               }
            }

            double pnew = si->data_loglikelihood() + logalphah + myloggamma(si->getN())
                        + sj->data_loglikelihood() + logalphah + myloggamma(sj->getN());
            double pold = clusters[mi]->data_loglikelihood() + logalphah + myloggamma(clusters[mi]->getN());
            double hratio = pnew - pold + 0 - prob_split;
            //mexPrintf("split %e / %e   %e / %e\n", pnew, pold, 0.0, prob_split);
            if (my_rand(r)<exp(hratio))
            {
               // accept split
               for (mj=0; mj<N; mj++)
                  if (clusters[mj]==NULL)
                     break;
               delete clusters[mi];
               clusters[mi] = si;
               clusters[mj] = sj;
               alive_ptrs[mj] = alive.addNodeEnd(mj);

               node = sj_indices.getFirst();
               while (node!=NULL)
               {
                  int ind = node->getData();
                  z[ind] = mj;
                  node = node->getNext();
               }
            }

            gsl_permutation_free(perm);
         }
         else // propose a merge
         {
            cluster_single* si = new cluster_single(hyper);
            cluster_single* sj = new cluster_single(hyper);
            linkedList<int> si_indices, sj_indices;
            si->add_data(x+i*D);
            sj->add_data(x+j*D);
            si_indices.addNode(i);
            sj_indices.addNode(j);
            int Nij = 0;
            for (int ind=0; ind<N; ind++)
               if (z[ind]==mi || z[ind]==mj)
                  indices[Nij++] = ind;

            // propose a new split
            double prob_split = 0;
            gsl_permutation *perm = gsl_permutation_alloc((size_t)Nij);
            gsl_permutation_init(perm);
            gsl_ran_shuffle(r, perm->data, Nij, sizeof(size_t));
            for (int permi=0; permi<Nij; permi++)
            {
               int ind = indices[permi];
               if (ind!=i && ind!=j)
               {
                  probabilities[0] = si->predictive_loglikelihood(x+ind*D) + logint(si->getN());
                  probabilities[1] = sj->predictive_loglikelihood(x+ind*D) + logint(sj->getN());

                  double totalProb = convert_logcategorical(probabilities, 2);
                  int ij = z[ind]==mi ? 0 : 1;

                  if (ij==0)
                  {
                     prob_split += log(probabilities[0] / totalProb);

   /*               if (mxIsInf(prob_split))
                  {
                     mexPrintf("%e  %e\n", probabilities[0], probabilities[1]);
                     mexPrintf("%e  %e\n", si->predictive_loglikelihood(x+ind*D) + logint(si->getN()), sj->predictive_loglikelihood(x+ind*D) + logint(sj->getN()));
                     mexErrMsgTxt("a\n");
                  }*/

                     si->add_data(x+ind*D);
                     si_indices.addNode(ind);
                  }
                  else
                  {
                     prob_split += log(probabilities[1] / totalProb);

   /*               if (mxIsInf(prob_split))
                  {
                     mexPrintf("%e  %e\n", probabilities[0], probabilities[1]);
                     mexPrintf("%e  %e\n", si->predictive_loglikelihood(x+ind*D) + logint(si->getN()), sj->predictive_loglikelihood(x+ind*D) + logint(sj->getN()));
                     mexErrMsgTxt("a\n");
                  }*/

                     sj->add_data(x+ind*D);
                     sj_indices.addNode(ind);
                  }
               }
            }

            double pold = si->data_loglikelihood() + logalphah + myloggamma(si->getN())
                        + sj->data_loglikelihood() + logalphah + myloggamma(sj->getN());
            double pnew = si->data_loglikelihood_testmerge(sj) + logalphah + myloggamma(si->getN()+sj->getN());
            double hratio = pnew - pold + prob_split - 0;

            /*mexPrintf("merge %e / %e   %e / %e\n", pnew, pold, prob_split, 0.0);
            if (prob_split==0)
            {
               mexPrintf("m1=%d m2=%d\n", mi, mj);
               mexPrintf("%d\n", Nij);
               mexPrintf("si->getN()=%d  sj->getN()=%d\n", si->getN(), sj->getN());
               mexErrMsgTxt("huh\n");
            }*/
            if (my_rand(r)<exp(hratio))
            {
               // accept merge
               delete clusters[mj];
               clusters[mj] = NULL;
               alive.deleteNode(alive_ptrs[mj]);
               alive_ptrs[mj] = NULL;

               node = sj_indices.getFirst();
               while (node!=NULL)
               {
                  int ind = node->getData();
                  z[ind] = mi;
                  clusters[mi]->add_data(x+ind*D);
                  node = node->getNext();
               }
            }

            gsl_permutation_free(perm);
         }
      }
   }
   plhs[0] = mxCreateNumericMatrix(1,1,mxDOUBLE_CLASS,mxREAL);
   arr(double) time = getArrayInput<double>(plhs[0]);
   time[0] = timer.toc();


   gsl_permutation_free(perm);
   gsl_rng_free(r);
   deallocate_memory(alive_ptrs);
   deallocate_memory(probabilities);

   for (int i=0; i<N; i++)
      if (clusters[i])
         delete clusters[i];
   deallocate_memory(clusters);
}
