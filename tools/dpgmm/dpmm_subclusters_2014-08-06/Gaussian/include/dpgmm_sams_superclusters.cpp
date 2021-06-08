// =============================================================================
// == dpgmm_sams_superclusters.cpp
// == --------------------------------------------------------------------------
// == The MEX interface file to sample a DP Gaussian Mixture Model with
// == superclusters and the sequentially allocate merge-split
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
// == [3] D. Lovell, J. Malmaud, R. P. Adams, and V. K. Mansinghka. Parallel
// ==     Markov Chain Monte Carlo for Dirichlet Process Mixtures. Workshop
// ==     on Big Learning, NIPS, 2012.
// == [4] S. A. Williamson, A. Dubey, and E. P. Xing. Parallel Markov
// ==     Chain Monte Carlo for Nonparametric Mixture Models. International
// ==     Conference on Machine Learning, 2013.
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

#define NUMARGS 4
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
   checkInput(prhs[2], VECTOR,  "int32"); // sc
   checkInput(prhs[3], STRUCT); // params

   const int *dims = mxGetDimensions(prhs[0]);
   int N = mxGetNumberOfElements(prhs[0]);
   int D = 1;
   if (mxGetNumberOfDimensions(prhs[0])>1)
      D = dims[0];
   N /= D;
   int D2 = D*D;

   arr(double) x = getArrayInput<double>(prhs[0]);
   arr(unsigned int) z = getArrayInput<unsigned int>(prhs[1]);
   arr(int) sc_clusters = getArrayInput<int>(prhs[2]);

   const mxArray* params = prhs[3];
   double logalphah = log(getInput<double>(getField(params,0,"alpha")));
   double kappah = getInput<double>(getField(params,0,"kappa"));
   double nuh = getInput<double>(getField(params,0,"nu"));
   arr(double) thetah = getArrayInput<double>(getField(params,0,"theta"));
   arr(double) deltah = getArrayInput<double>(getField(params,0,"delta"));
   int its_crp = getInput<double>(getField(params,0,"its_crp"));
   int its_ms = getInput<double>(getField(params,0,"its_ms"));
   int its = max(its_crp, its_ms);
   int Mproc = getInput<double>(getField(params,0,"Mproc"));

   omp_set_num_threads(Mproc);

   niw hyper(true, D, kappah, nuh, thetah, deltah);
   hyper.update();

   arr(cluster_single*) clusters = allocate_memory<cluster_single*>(N, NULL);
   arr(linkedListNode<int>*) alive_ptrs = allocate_memory< linkedListNode<int>* >(N,NULL);
   linkedList<int> alive;

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

   arr(arr(double)) probabilities = allocate_memory<arr(double)>(Mproc);
   arr(gsl_rng*) rArr = allocate_memory<gsl_rng*>(Mproc);
   for (int t=0; t<Mproc; t++)
   {
      probabilities[t] = allocate_memory<double>(N);
      rArr[t] = initialize_gsl_rand(mx_rand());
   }

   arr(int) indices = allocate_memory<int>(N);
   arr(linkedListNode<int>*) llNodePointers = allocate_memory<linkedListNode<int>*>(N,NULL);
   arr(linkedList<int>*) llpoints = allocate_memory< linkedList<int>* >(N,NULL);
   for (int i=0; i<N; i++)
   {
      if (llpoints[z[i]]==NULL)
         llpoints[z[i]] = new linkedList<int>();
      llNodePointers[i] = llpoints[z[i]]->addNodeEnd(i);
   }

   //arr(int) superclusters = NULL;
   arr(int) superclusters = allocate_memory<int>(N,0);
   arr(linkedList<int>*) supercluster_labels = allocate_memory< linkedList<int>* >(N,NULL);
   arr(int) oldLengths = allocate_memory<int>(N,0);

   stopwatch timer;
   timer.tic();
   double t;
   for (int it=0; it<its; it++)
   {
//mexPrintf("start\n");

      // get the supercluster stuff
      int K = alive.getLength();
//mexPrintf("populate sc\n");
      linkedList<int> temp;
      linkedListNode<int>* node = alive.getFirst();
      int SC = 0;
      while (node!=NULL)
      {
         int m = node->getData();
         int sc = sc_clusters[m];
         if (supercluster_labels[sc]==NULL)
         {
            supercluster_labels[sc] = new linkedList<int>();
            superclusters[SC] = sc;
            SC++;
         }
         supercluster_labels[sc]->addNodeEnd(m);
         node = node->getNext();
      }
//mexPrintf("sample sc\n");

/*mexPrintf("Before----------------------------\n");
for (int sci=0; sci<SC; sci++)
{
   int sc = superclusters[sci];
   mexPrintf("sc=%d\t", sc);
   linkedListNode<int>* temp = supercluster_labels[sc]->getFirst();
   while (temp!=NULL)
   {
      mexPrintf("%d\t", temp->getData());
      temp = temp->getNext();
   }
   mexPrintf("\n");
}*/


      // sample superclusters
      arr(double) probabilities0 = probabilities[0];
      node = alive.getFirst();
      while (node!=NULL)
      {
         int m = node->getData();
         int sc = sc_clusters[m];
         linkedListNode<int>* temp = supercluster_labels[sc]->getFirst();
         bool found = false;
         while (temp!=NULL)
         {
            if (temp->getData() == m)
            {
               supercluster_labels[sc]->deleteNode(temp);
               found = true;
               break;
            }
            temp = temp->getNext();
         }
         if (!found)
         {
            mexPrintf("%d does nto belong to sc %d\n", m, sc);
            mexErrMsgTxt("Broken!\n");
         }

         double maxProb = logalphah;
         // sample a new one
         for (int new_sci=0; new_sci<SC; new_sci++)
         {
            int new_sc = superclusters[new_sci];
            if (supercluster_labels[new_sc]->getLength()>0)
               probabilities0[new_sci] = log(supercluster_labels[new_sc]->getLength());
            else
               probabilities0[new_sci] = -mxGetInf();
            maxProb = max(maxProb, probabilities0[new_sci]);
         }
         probabilities0[SC] = logalphah;

         double totalProb = convert_logcategorical(probabilities0, SC+1, maxProb);
         int new_sci = sample_categorical(probabilities0, SC+1, totalProb, r);
         int new_sc;
         if (new_sci==SC) // new supercluster
         {
//mexPrintf("new m=%d \t sc=%d\tnew_sc=%d\n",m,sc,new_sc);
            for (new_sci=0; new_sci<SC; new_sci++)
               if (supercluster_labels[superclusters[new_sci]]->isempty())
                  break;
            if (new_sci==SC)
            {
               for (new_sc=0; new_sc<SC+1; new_sc++)
                  if (supercluster_labels[new_sc]==NULL)
                     break;
               supercluster_labels[new_sc] = new linkedList<int>();
               superclusters[SC] = new_sc;
               SC++;
            }
            else
               new_sc = superclusters[new_sci];
            supercluster_labels[new_sc]->addNodeEnd(m);
         }
         else
         {
//mexPrintf("old m=%d \t sc=%d\tnew_sc=%d\n",m,sc,new_sc);
            new_sc = superclusters[new_sci];
            supercluster_labels[new_sc]->addNodeEnd(m);
         }
         sc_clusters[m] = new_sc;
         node = node->getNext();
      }

/*mexPrintf("After----------------------------\n");
for (int sci=0; sci<SC; sci++)
{
   int sc = superclusters[sci];
   mexPrintf("sc=%d\t", sc);
   linkedListNode<int>* temp = supercluster_labels[sc]->getFirst();
   while (temp!=NULL)
   {
      mexPrintf("%d\t", temp->getData());
      temp = temp->getNext();
   }
   mexPrintf("\n");
}*/


//mexPrintf("CRP\n");

      if (it<its_crp)
      {
         // potential new clusters
         arr(arr(cluster_single*)) new_clusters = allocate_memory<arr(cluster_single*)>(SC,NULL);
         arr(arr(linkedList<int>*)) new_llpoints = allocate_memory<arr(linkedList<int>*)>(SC,NULL);
         for (int sc=0; sc<SC; sc++)
         {
            new_clusters[sc] = allocate_memory<cluster_single*>(N,NULL);
            new_llpoints[sc] = allocate_memory<linkedList<int>*>(N,NULL);
         }

         linkedListNode<int>* node = alive.getFirst();
         while (node!=NULL)
         {
            int m = node->getData();
            oldLengths[m] = llpoints[m]->getLength();
            node = node->getNext();
         }

         #pragma omp parallel for
         for (int sci=0; sci<SC; sci++)
         {
            //int proc = 0;
            int proc = omp_get_thread_num();

            int sc = superclusters[sci];
            arr(double) tprobabilities = probabilities[proc];
            arr(cluster_single*) tnew_clusters = new_clusters[sci];
            arr(linkedList<int>*) tnew_llpoints = new_llpoints[sci];
            int Nk_sc = supercluster_labels[sc]->getLength();

            linkedListNode<int>* node = supercluster_labels[sc]->getFirst();
            for (int ki=0; ki<Nk_sc; ki++)
            {
               int k = node->getData();
               if (llpoints[k]==NULL)
                  mexErrMsgTxt("llpoints[k]==NULL\n");
               int Nk = oldLengths[k];

               for (int ik=0; ik<Nk; ik++)
               {
                  int i = llpoints[k]->popFirst();
                  llNodePointers[i] = NULL;

                  // sample point i from within the supercluster labels
                  // remove it from the current component
                  int m = z[i];
                  //if (m!=k)
                     //mexPrintf("m=%d, k=%d\n",m,k);
                  clusters[m]->rem_data(x+i*D);

                  // find the distribution over possible ones
                  int j;
                  double maxProb = -mxGetInf();
                  linkedListNode<int>* testnode = supercluster_labels[sc]->getFirst();
                  for (j=0; j<Nk_sc; j++)
                  {
                     int m = testnode->getData();
                     // find the probability of the data belonging to this component
                     double prob;
                     if (clusters[m]==NULL || clusters[m]->isempty())
                        prob = -mxGetInf();
                     else
                        prob = clusters[m]->predictive_loglikelihood(x+i*D) + logint(clusters[m]->getN());
                     maxProb = max(maxProb, prob);
                     tprobabilities[j] = prob;
                     testnode = testnode->getNext();
                  }

                  // find the distribution over the possible new clusters
                  int nci = 0;
                  while (tnew_clusters[nci]!=NULL)
                  {
                     // find the probability of the data belonging to this component
                     double prob = tnew_clusters[nci]->predictive_loglikelihood(x+i*D) + logint(tnew_clusters[nci]->getN());
                     maxProb = max(maxProb, prob);
                     tprobabilities[j] = prob;
                     nci++;
                     j++;
                  }

                  // new component
                  tprobabilities[j] = hyper.predictive_loglikelihood(x+i*D) + logalphah;
                  maxProb = max(maxProb, tprobabilities[j]);

                  double totalProb = convert_logcategorical(tprobabilities, j+1, maxProb);
                  int newk = sample_categorical(tprobabilities, j+1, totalProb, rArr[proc]);
                  // find the m

                  if (newk==j) // new one
                  {
                     newk = 0;
                     while (tnew_clusters[newk]!=NULL)
                        newk++;
                     tnew_clusters[newk] = new cluster_single(hyper);
                     tnew_clusters[newk]->add_data(x+i*D);
                     tnew_llpoints[newk] = new linkedList<int>();
                     llNodePointers[i] = tnew_llpoints[newk]->addNodeEnd(i);
                  }
                  else if (newk>=Nk_sc) // assigned to a new cluster that was created already
                  {
                     newk = newk-Nk_sc;
                     tnew_clusters[newk]->add_data(x+i*D);
                     llNodePointers[i] = tnew_llpoints[newk]->addNodeEnd(i);
                     newk = N;
                  }
                  else
                  {
                     newk = supercluster_labels[sc]->operator[](newk);
                     clusters[newk]->add_data(x+i*D);
                     llNodePointers[i] = llpoints[newk]->addNodeEnd(i);
                     z[i] = newk;
                  }
               }
               node = node->getNext();
            }
         }

         // delete the old dead clusters;
         node = alive.getFirst();
         while (node!=NULL)
         {
            int m = node->getData();
            node = node->getNext();
            if (clusters[m]->isempty())
            {
               alive.deleteNode(alive_ptrs[m]);
               alive_ptrs[m] = NULL;
               delete clusters[m];
               clusters[m] = NULL;
               delete llpoints[m];
               llpoints[m] = NULL;
            }
         }


         // accumulate the new clusters
         for (int sci=0; sci<SC; sci++)
         {
            int sc = superclusters[sci];
            arr(cluster_single*) tnew_clusters = new_clusters[sci];
            arr(linkedList<int>*) tnew_llpoints = new_llpoints[sci];
            int Nk_sc = supercluster_labels[sc]->getLength();
            int k = 0;
            while (tnew_clusters[k]!=NULL)
            {
               // copy over the cluster
               // find an empty one
               int m;
               for (m=0; m<N; m++)
                  if (clusters[m]==NULL)
                     break;
               sc_clusters[m] = sc;
               clusters[m] = tnew_clusters[k];
               tnew_clusters[k] = NULL;
               if (llpoints[m]!=NULL)
                  mexErrMsgTxt("huh? why isn't this null\n");
               llpoints[m] = tnew_llpoints[k];
               tnew_llpoints[k] = NULL;

               alive_ptrs[m] = alive.addNodeEnd(m);

               // loop through and fix the labels
               if (llpoints[m]!=NULL)
               {
                  linkedListNode<int>* node = llpoints[m]->getFirst();
                  //linkedListNode<int>* node = tnew_llpoints[k]->getFirst();
                  while (node!=NULL)
                  {
                     z[node->getData()] = m;
                     node = node->getNext();
                  }
               }
               k++;
            }
         }

         for (int sc=0; sc<SC; sc++)
         {
            /*for (int i=0; i<N; i++)
            {
               if (new_clusters[sc][i]!=NULL) delete new_clusters[sc][i];
               if (new_llpoints[sc][i]!=NULL) delete new_llpoints[sc][i];
            }*/
            deallocate_memory(new_clusters[sc]);
            deallocate_memory(new_llpoints[sc]);
         }
         deallocate_memory(new_clusters);
         deallocate_memory(new_llpoints);

      }

//mexPrintf("M/S\n");

      arr(double) tprobabilities = probabilities[0];
      if (it<its_ms)
      {
         // select two random points
         int i = my_randi(r)%N;
         int j = my_randi(r)%(N-1);
         if (j>i) j = j+1;

         int mi = z[i];
         int mj = z[j];

         if (clusters[mi]==NULL)
            mexErrMsgTxt("mi null!\n");
         if (clusters[mj]==NULL)
            mexErrMsgTxt("mj null!\n");

         if (mi==mj) // propose a split
         {
//mexPrintf("split propose\n");
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
                  tprobabilities[0] = si->predictive_loglikelihood(x+ind*D) + logint(si->getN());
                  tprobabilities[1] = sj->predictive_loglikelihood(x+ind*D) + logint(sj->getN());

                  double totalProb = convert_logcategorical(tprobabilities, 2);
                  int ij = sample_categorical(tprobabilities, 2, totalProb, r);

                  if (ij==0)
                  {
                     prob_split += log(tprobabilities[0] / totalProb);
                     si->add_data(x+ind*D);
                     si_indices.addNode(ind);
                  }
                  else
                  {
                     prob_split += log(tprobabilities[1] / totalProb);
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
//mexPrintf("split cluster\n");
               // accept split
               for (mj=0; mj<N; mj++)
                  if (clusters[mj]==NULL)
                     break;
               delete clusters[mi];
               clusters[mi] = si;
               clusters[mj] = sj;
               sc_clusters[mj] = sc_clusters[mi];
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
//mexPrintf("merge propose\n");
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
                  tprobabilities[0] = si->predictive_loglikelihood(x+ind*D) + logint(si->getN());
                  tprobabilities[1] = sj->predictive_loglikelihood(x+ind*D) + logint(sj->getN());

                  double totalProb = convert_logcategorical(tprobabilities, 2);
                  int ij = z[ind]==mi ? 0 : 1;

                  if (ij==0)
                  {
                     prob_split += log(tprobabilities[0] / totalProb);
                     si->add_data(x+ind*D);
                     si_indices.addNode(ind);
                  }
                  else
                  {
                     prob_split += log(tprobabilities[1] / totalProb);
                     sj->add_data(x+ind*D);
                     sj_indices.addNode(ind);
                  }
               }
            }

            double pold = si->data_loglikelihood() + logalphah + myloggamma(si->getN())
                        + sj->data_loglikelihood() + logalphah + myloggamma(sj->getN());
            double pnew = si->data_loglikelihood_testmerge(sj) + logalphah + myloggamma(si->getN()+sj->getN());
            double hratio = pnew - pold + prob_split - 0;

            if (my_rand(r)<exp(hratio))
            {
//mexPrintf("merge cluster\n");
               // accept merge
               sc_clusters[mj] = sc_clusters[mi];
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
   t = timer.toc();
   plhs[0] = mxCreateNumericMatrix(1,1,mxDOUBLE_CLASS,mxREAL);
   arr(double) time = getArrayInput<double>(plhs[0]);
   time[0] = t;



   for (int i=0; i<N; i++)
      if (supercluster_labels[i]!=NULL)
         delete supercluster_labels[i];
   if (supercluster_labels!=NULL) deallocate_memory(supercluster_labels);
   if (superclusters!=NULL) deallocate_memory(superclusters);


   deallocate_memory(indices);
   for (int i=0; i<N; i++)
      if (llpoints[i]!=NULL)
         delete llpoints[i];
   deallocate_memory(llpoints);
   deallocate_memory(llNodePointers);

   deallocate_memory(oldLengths);



   gsl_permutation_free(perm);
   gsl_rng_free(r);
   deallocate_memory(alive_ptrs);
   for (int m=0; m<Mproc; m++)
   {
      deallocate_memory(probabilities[m]);
      gsl_rng_free(rArr[m]);
   }
   deallocate_memory(probabilities);
   deallocate_memory(rArr);

   for (int i=0; i<N; i++)
      if (clusters[i])
         delete clusters[i];
   deallocate_memory(clusters);
}
