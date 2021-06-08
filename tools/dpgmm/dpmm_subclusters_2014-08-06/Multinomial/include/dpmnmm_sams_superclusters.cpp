// =============================================================================
// == dpmnmm_sams_superclusters.cpp
// == --------------------------------------------------------------------------
// == The MEX interface file to sample a DP Multinomial Mixture Model with
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
#include "gsl/gsl_rng.h"
#include "gsl/gsl_randist.h"
#include "gsl/gsl_permutation.h"
#include "gsl/gsl_cdf.h"
#include "myfuncs.h"
#include "sample_categorical.h"
#include "dir_sampled_hash.h"
#include "dir_sampled_full.h"
#include "stopwatch.h"
#include "mxSparseInput.h"

#define NUMARGS 4
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

   checkInput(prhs[0], "double"); //x
   checkInput(prhs[1], VECTOR,  "uint32"); //z
   checkInput(prhs[2], VECTOR,  "int32"); // sc
   checkInput(prhs[3], STRUCT); // params

   int N = mxGetN(prhs[0]);
   int D = mxGetM(prhs[0]);

   mxSparseInput x(prhs[0]);
   arr(unsigned int) z = getArrayInput<unsigned int>(prhs[1]);
   arr(int) sc_clusters = getArrayInput<int>(prhs[2]);

   const mxArray* params = prhs[3];
   double alphah = getInput<double>(getField(params,0,"alpha"));
   double diralphah = getInput<double>(getField(params,0,"diralpha"));
   double logalphah = log(alphah);
   int its_crp = getInput<double>(getField(params,0,"its_crp"));
   int its_ms = getInput<double>(getField(params,0,"its_ms"));
   int its = max(its_crp, its_ms);
   int Mproc = getInput<double>(getField(params,0,"Mproc"));

   omp_set_num_threads(Mproc);
   arr(dir_sampled*) hypers = allocate_memory<dir_sampled*>(Mproc,NULL);
   for (int t=0; t<Mproc; t++)
      hypers[t] = new dir_sampled(1,D,diralphah);

   dir_sampled hyper(1, D, diralphah);
   hyper.update_posteriors();

   arr(dir_sampled*) clusters = allocate_memory<dir_sampled*>(N, NULL);
   arr(linkedListNode<int>*) alive_ptrs = allocate_memory< linkedListNode<int>* >(N,NULL);
   linkedList<int> alive;

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



   plhs[0] = mxCreateNumericMatrix(its,1,mxDOUBLE_CLASS,mxREAL);
   plhs[1] = mxCreateNumericMatrix(its,1,mxDOUBLE_CLASS,mxREAL);
   arr(double) timeArr = getArrayInput<double>(plhs[0]);
   arr(double) EArr = getArrayInput<double>(plhs[1]);



   node = alive.getFirst();
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




   stopwatch timer;
   double t;
   for (int it=0; it<its; it++)
   {
      //mexPrintf("it=%d\n",it);drawnow();
      timer.tic();
      // get the supercluster stuff
      int K = alive.getLength();
      linkedList<int> temp;


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
}
drawnow();*/



      if (it<its_crp)
      {
         // potential new clusters
         arr(arr(dir_sampled*)) new_clusters = allocate_memory<arr(dir_sampled*)>(SC,NULL);
         arr(arr(linkedList<int>*)) new_llpoints = allocate_memory<arr(linkedList<int>*)>(SC,NULL);
         for (int sc=0; sc<SC; sc++)
         {
            new_clusters[sc] = allocate_memory<dir_sampled*>(N,NULL);
            new_llpoints[sc] = allocate_memory<linkedList<int>*>(N,NULL);
         }

         linkedListNode<int>* node = alive.getFirst();
         while (node!=NULL)
         {
            int m = node->getData();
            oldLengths[m] = llpoints[m]->getLength();
            node = node->getNext();
         }

         double minusinf = -mxGetInf();

         #pragma omp parallel for
         for (int sci=0; sci<SC; sci++)
         {
            //int proc = 0;
            int proc = omp_get_thread_num();

            int sc = superclusters[sci];
            arr(double) tprobabilities = probabilities[proc];
            arr(dir_sampled*) tnew_clusters = new_clusters[sci];
            arr(linkedList<int>*) tnew_llpoints = new_llpoints[sci];
            int Nk_sc = supercluster_labels[sc]->getLength();
            dir_sampled* thyper = hypers[proc];

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
                  long* colInds; double* colVals; long nnz;
                  x.getColumn(i, colInds, colVals, nnz);

                  llNodePointers[i] = NULL;

                  // sample point i from within the supercluster labels
                  // remove it from the current component
                  int m = z[i];
                  //if (m!=k)
                     //mexPrintf("m=%d, k=%d\n",m,k);
                  clusters[m]->rem_data(colInds, colVals, nnz);

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
                        prob = minusinf;
                     else
                        prob = clusters[m]->predictive_loglikelihood_marginalized(colInds, colVals, nnz) + logint(clusters[m]->getN());
                     maxProb = max(maxProb, prob);
                     tprobabilities[j] = prob;
                     testnode = testnode->getNext();
                  }

                  // find the distribution over the possible new clusters
                  int nci = 0;
                  while (tnew_clusters[nci]!=NULL)
                  {
                     // find the probability of the data belonging to this component
                     double prob = tnew_clusters[nci]->predictive_loglikelihood_marginalized(colInds, colVals, nnz) + logint(tnew_clusters[nci]->getN());
                     maxProb = max(maxProb, prob);
                     tprobabilities[j] = prob;
                     nci++;
                     j++;
                  }

                  // new component
                  tprobabilities[j] = thyper->predictive_loglikelihood_marginalized_hyper(colInds, colVals, nnz) + logalphah;
                  maxProb = max(maxProb, tprobabilities[j]);

                  double totalProb = convert_logcategorical(tprobabilities, j+1, maxProb);
                  int newk = sample_categorical(tprobabilities, j+1, totalProb, rArr[proc]);
                  // find the m

                  if (newk==j) // new one
                  {
                     for (newk=0; newk<N; newk++)
                        if (tnew_clusters[newk]==NULL)
                           break;
                     if (newk>=N)
                        mexErrMsgTxt("too many new clusters\n");
                     tnew_clusters[newk] = new dir_sampled(hyper);
                     tnew_clusters[newk]->add_data(colInds, colVals, nnz);
                     tnew_llpoints[newk] = new linkedList<int>();
                     llNodePointers[i] = tnew_llpoints[newk]->addNodeEnd(i);
                  }
                  else if (newk>=Nk_sc) // assigned to a new cluster that was created already
                  {
                     newk = newk-Nk_sc;
                     tnew_clusters[newk]->add_data(colInds, colVals, nnz);
                     llNodePointers[i] = tnew_llpoints[newk]->addNodeEnd(i);
                     newk = N;
                  }
                  else
                  {
                     newk = supercluster_labels[sc]->operator[](newk);
                     clusters[newk]->add_data(colInds, colVals, nnz);
                     llNodePointers[i] = llpoints[newk]->addNodeEnd(i);
                     z[i] = newk;
                  }
               }
               node = node->getNext();
            }
         }
//mexPrintf("deleting\n");//drawnow();

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

               // find and delete the sc also
               int sc = sc_clusters[m];
               linkedListNode<int>* temp = supercluster_labels[sc]->getFirst();
               while (temp!=NULL)
               {
                  if (temp->getData() == m)
                  {
                     supercluster_labels[sc]->deleteNode(temp);
                     break;
                  }
                  temp = temp->getNext();
               }
            }

         }

//mexPrintf("accumulating\n");//drawnow();
         // accumulate the new clusters
         for (int sci=0; sci<SC; sci++)
         {
            int sc = superclusters[sci];
            arr(dir_sampled*) tnew_clusters = new_clusters[sci];
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
               supercluster_labels[sc]->addNodeEnd(m);
               //mexPrintf("adding sc=%d m=%d\n", sci, m);

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


/*mexPrintf("After Again----------------------------\n");
int count = 0;
for (int sci=0; sci<SC; sci++)
{
   int sc = superclusters[sci];
   mexPrintf("sc=%d\t", sc);
   linkedListNode<int>* temp = supercluster_labels[sc]->getFirst();
   while (temp!=NULL)
   {
      mexPrintf("%d\t", temp->getData());
      temp = temp->getNext();
      count++;
   }
   mexPrintf("\n");
}
if (count!=alive.getLength())
   mexErrMsgTxt("repeats!\n");
drawnow();*/


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

         long* colInds_i; double* colVals_i; long nnz_i;
         long* colInds_j; double* colVals_j; long nnz_j;
         long* colInds; double* colVals; long nnz;
         x.getColumn(i, colInds_i, colVals_i, nnz_i);
         x.getColumn(j, colInds_j, colVals_j, nnz_j);

         if (clusters[mi]==NULL)
            mexErrMsgTxt("mi null!\n");
         if (clusters[mj]==NULL)
            mexErrMsgTxt("mj null!\n");

         if (mi==mj) // propose a split
         {
//mexPrintf("split propose\n");
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
                  tprobabilities[0] = si->predictive_loglikelihood_marginalized(colInds, colVals, nnz) + logint(si->getN());
                  tprobabilities[1] = sj->predictive_loglikelihood_marginalized(colInds, colVals, nnz) + logint(sj->getN());

                  double totalProb = convert_logcategorical(tprobabilities, 2);
                  int ij = sample_categorical(tprobabilities, 2, totalProb, r);

                  if (ij==0)
                  {
                     prob_split += log(tprobabilities[0] / totalProb);
                     si->add_data(colInds, colVals, nnz);
                     si_indices.addNode(ind);
                  }
                  else
                  {
                     prob_split += log(tprobabilities[1] / totalProb);
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
                  tprobabilities[0] = si->predictive_loglikelihood_marginalized(colInds, colVals, nnz) + logint(si->getN());
                  tprobabilities[1] = sj->predictive_loglikelihood_marginalized(colInds, colVals, nnz) + logint(sj->getN());

                  double totalProb = convert_logcategorical(tprobabilities, 2);
                  int ij = z[ind]==mi ? 0 : 1;

                  if (ij==0)
                  {
                     prob_split += log(tprobabilities[0] / totalProb);
                     si->add_data(colInds, colVals, nnz);
                     si_indices.addNode(ind);
                  }
                  else
                  {
                     prob_split += log(tprobabilities[1] / totalProb);
                     sj->add_data(colInds, colVals, nnz);
                     sj_indices.addNode(ind);
                  }
               }
            }

            double pold = si->data_loglikelihood_marginalized() + logalphah + gammalnint(si->getN())
                        + sj->data_loglikelihood_marginalized() + logalphah + gammalnint(sj->getN());
            double pnew = si->data_loglikelihood_marginalized_testmerge(sj) + logalphah + gammalnint(si->getN()+sj->getN());
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
   


   for (int t=0; t<Mproc; t++)
      if (hypers[t]!=NULL)
         delete hypers[t];
   deallocate_memory(hypers);

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
