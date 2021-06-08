// =============================================================================
// == dpmnmm_sams.cpp
// == --------------------------------------------------------------------------
// == The MEX interface file to sample a DP Multinomial Mixture Model with
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
#include "gsl/gsl_rng.h"
#include "gsl/gsl_randist.h"
#include "gsl/gsl_permutation.h"
#include "gsl/gsl_cdf.h"
#include "tableFuncs.h"
#include "sample_categorical.h"
#include "dir_sampled_hash.h"
#include "dir_sampled_full.h"
#include "mxSparseInput.h"
#include "stopwatch.h"

#define NUMARGS 3
#define NUMOUT 2

#ifdef USEFULL
   typedef dir_sampled_full dir_sampled;
#else
   typedef dir_sampled_hash dir_sampled;
#endif


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
   // Check for proper number of arguments
   if (nrhs != NUMARGS) {
         mexErrMsgTxt("Incorrect number of input arguments required.");
   } else if (nlhs > NUMOUT) {
         mexErrMsgTxt("Too many output arguments expected.");
   }

   if (!mxIsSparse(prhs[0]))
      mexErrMsgTxt("Not sparse!\n");
   checkInput(prhs[0], "double"); //x
   checkInput(prhs[1], VECTOR,  "uint32"); //z
   checkInput(prhs[2], STRUCT); // params

   int N = mxGetN(prhs[0]);
   int D = mxGetM(prhs[0]);

   mxSparseInput x(prhs[0]);
   arr(unsigned int) z = getArrayInput<unsigned int>(prhs[1]);

   const mxArray* params = prhs[2];
   double alphah = getInput<double>(getField(params,0,"alpha"));
   double diralphah = getInput<double>(getField(params,0,"diralpha"));
   double logalphah = log(alphah);
   int its_crp = getInput<double>(getField(params,0,"its_crp"));
   int its_ms = getInput<double>(getField(params,0,"its_ms"));
   int its = max(its_crp, its_ms);

   omp_set_num_threads(1);
   dir_sampled hyper(1, D, diralphah);
   hyper.update_posteriors();

   arr(dir_sampled*) clusters = allocate_memory<dir_sampled*>(N, NULL);
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
         clusters[m] = new dir_sampled(hyper);
      }
      long* colInds; double* colVals; long nnz;
      x.getColumn(i, colInds, colVals, nnz);
      clusters[m]->add_data_init(colInds, colVals, nnz);
   }

   // populate the statistics
   linkedListNode<int>* node = alive.getFirst();
   while (node!=NULL)
   {
      int m = node->getData();
      clusters[m]->update_posteriors();
      node = node->getNext();
   }


   gsl_rng *r = initialize_gsl_rand(mx_rand());
   const size_t count_t = N;
   const gsl_rng_type *T;
   gsl_permutation *perm = gsl_permutation_alloc(count_t);
   T = gsl_rng_default;
   gsl_permutation_init(perm);


   arr(int) indices = allocate_memory<int>(N);

   plhs[0] = mxCreateNumericMatrix(its,1,mxDOUBLE_CLASS,mxREAL);
   plhs[1] = mxCreateNumericMatrix(its,1,mxDOUBLE_CLASS,mxREAL);
   arr(double) timeArr = getArrayInput<double>(plhs[0]);
   arr(double) EArr = getArrayInput<double>(plhs[1]);

   stopwatch timer;
   for (int it=0; it<its; it++)
   {
      timer.tic();
      //mexPrintf("it=%d \t K=%d\n", it, alive.getLength());
      if (it<its_crp)
      {
         gsl_ran_shuffle(r, perm->data, count_t, sizeof(size_t));
         for (int permi=0; permi<N; permi++)
         {
            int i = perm->data[permi];

            // get the data
            long* colInds; double* colVals; long nnz;
            x.getColumn(i, colInds, colVals, nnz);

            // remove it from the previous component
            int m = z[i];
            clusters[m]->rem_data(colInds, colVals, nnz);
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
               double prob = clusters[m]->predictive_loglikelihood_marginalized(colInds, colVals, nnz) + logint(clusters[m]->getN());
               maxProb = max(maxProb, prob);
               probabilities[j++] = prob;
               node = node->getNext();
            }

            // new component
            probabilities[j] = hyper.predictive_loglikelihood_marginalized(colInds, colVals, nnz) + logalphah;
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
               clusters[m] = new dir_sampled(hyper);
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
            clusters[m]->add_data(colInds, colVals, nnz);
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

         long* colInds_i; double* colVals_i; long nnz_i;
         long* colInds_j; double* colVals_j; long nnz_j;
         long* colInds; double* colVals; long nnz;
         x.getColumn(i, colInds_i, colVals_i, nnz_i);
         x.getColumn(j, colInds_j, colVals_j, nnz_j);

         if (mi==mj) // propose a split
         {
            dir_sampled* si = new dir_sampled(hyper);
            dir_sampled* sj = new dir_sampled(hyper);
            linkedList<int> si_indices, sj_indices;
            si->add_data(colInds_i, colVals_i, nnz_i);
            sj->add_data(colInds_j, colVals_j, nnz_j);
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
                  x.getColumn(ind, colInds, colVals, nnz);
                  probabilities[0] = si->predictive_loglikelihood(colInds, colVals, nnz) + logint(si->getN());
                  probabilities[1] = sj->predictive_loglikelihood(colInds, colVals, nnz) + logint(sj->getN());

                  double totalProb = convert_logcategorical(probabilities, 2);
                  int ij = sample_categorical(probabilities, 2, totalProb, r);

                  if (ij==0)
                  {
                     prob_split += log(probabilities[0] / totalProb);
                     si->add_data(colInds, colVals, nnz);
                     si_indices.addNode(ind);
                  }
                  else
                  {
                     prob_split += log(probabilities[1] / totalProb);
                     sj->add_data(colInds, colVals, nnz);
                     sj_indices.addNode(ind);
                  }
               }
            }

            double pnew = si->data_loglikelihood_marginalized() + logalphah + gammalnint(si->getN())
                        + sj->data_loglikelihood_marginalized() + logalphah + gammalnint(sj->getN());
            double pold = clusters[mi]->data_loglikelihood_marginalized() + logalphah + gammalnint(clusters[mi]->getN());
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
            dir_sampled* si = new dir_sampled(hyper);
            dir_sampled* sj = new dir_sampled(hyper);
            linkedList<int> si_indices, sj_indices;
            si->add_data(colInds_i, colVals_i, nnz_i);
            sj->add_data(colInds_j, colVals_j, nnz_j);
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
                  x.getColumn(ind, colInds, colVals, nnz);
                  probabilities[0] = si->predictive_loglikelihood(colInds, colVals, nnz) + logint(si->getN());
                  probabilities[1] = sj->predictive_loglikelihood(colInds, colVals, nnz) + logint(sj->getN());

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

                     si->add_data(colInds, colVals, nnz);
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

                     sj->add_data(colInds, colVals, nnz);
                     sj_indices.addNode(ind);
                  }
               }
            }

            double pold = si->data_loglikelihood_marginalized() + logalphah + gammalnint(si->getN())
                        + sj->data_loglikelihood_marginalized() + logalphah + gammalnint(sj->getN());
            double pnew = si->data_loglikelihood_marginalized_testmerge(sj) + logalphah + gammalnint(si->getN()+sj->getN());
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
                  x.getColumn(ind, colInds, colVals, nnz);
                  z[ind] = mi;
                  clusters[mi]->add_data(colInds, colVals, nnz);
                  node = node->getNext();
               }
            }

            gsl_permutation_free(perm);
         }
      }

      double time = timer.toc();
      timeArr[it] = time;
      node = alive.getFirst();
      double loglikelihood = gsl_sf_lngamma(alphah) - gsl_sf_lngamma(alphah+N);
      while (node!=NULL)
      {
         int m = node->getData();
         loglikelihood += logalphah + gsl_sf_lngamma(clusters[m]->getN()) + clusters[m]->data_loglikelihood_marginalized();
         node = node->getNext();
      }
      EArr[it] = loglikelihood;
   }
   /*plhs[0] = mxCreateNumericMatrix(1,1,mxDOUBLE_CLASS,mxREAL);
   arr(double) time = getArrayInput<double>(plhs[0]);
   time[0] = timer.toc();*/


   gsl_permutation_free(perm);
   gsl_rng_free(r);
   deallocate_memory(alive_ptrs);
   deallocate_memory(probabilities);

   for (int i=0; i<N; i++)
      if (clusters[i])
         delete clusters[i];
   deallocate_memory(clusters);
}
