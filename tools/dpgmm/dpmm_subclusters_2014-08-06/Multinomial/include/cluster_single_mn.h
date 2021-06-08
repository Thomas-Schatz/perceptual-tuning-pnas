#ifndef _CLUSTER_SINGLE_H_INCLUDED_
#define _CLUSTER_SINGLE_H_INCLUDED_

#include "matrix.h"
#include "mex.h"
#include <math.h>
#include "array.h"
#include "dir_sampled_hash.h"
#include "dir_sampled_full.h"

#include "helperMEX.h"
#include "debugMEX.h"

#ifdef USEFULL
   typedef dir_sampled_full dir_sampled;
#else
   typedef dir_sampled_hash dir_sampled;
#endif

class cluster_single
{
   // prior hyperparameters
   dir_sampled params;

public:
   // --------------------------------------------------------------------------
   // -- cluster_single
   // --   constructor; initializes to empty
   // --------------------------------------------------------------------------
   cluster_single();
   // --------------------------------------------------------------------------
   // -- cluster_single
   // --   copy constructor;
   // --------------------------------------------------------------------------
   cluster_single(const cluster_single& that);
   // --------------------------------------------------------------------------
   // -- operator=
   // --   assignment operator
   // --------------------------------------------------------------------------
   cluster_single& operator=(const cluster_single& that);
   // --------------------------------------------------------------------------
   // -- copy
   // --   returns a copy of this
   // --------------------------------------------------------------------------
   void copy(const cluster_single& that);
   // --------------------------------------------------------------------------
   // -- cluster_single
   // --   copy that into all params
   // --------------------------------------------------------------------------
   cluster_single(const dir_sampled& that);

   // --------------------------------------------------------------------------
   // -- ~cluster_single
   // --   destructor
   // --------------------------------------------------------------------------
   virtual ~cluster_single();

public:
   // --------------------------------------------------------------------------
   // -- empty
   // --   Empties out the statistics of the cluster_single (i.e. no data).
   // --------------------------------------------------------------------------
   void empty();

   bool isempty();
   int getN();
   dir_sampled* get_params();
   void set_params(dir_sampled* _params);

   // --------------------------------------------------------------------------
   // -- update
   // --   Updates the posterior hyperparameters and the parameters for the
   // -- predictive moment-matched gaussian
   // --------------------------------------------------------------------------
   void update();

   // --------------------------------------------------------------------------
   // -- XXX_data
   // --   functions to remove or add an observation to the cluster_single. Updates
   // -- sufficient statistics, posterior hyperparameters, and predictive
   // -- parameters for all relevant NIWs. phi should be in [0,1), and indicates
   // -- the following:
   // --                phi range      NIWs to update
   // --                [0,0.25)       paramsll,paramsl,params
   // --                [0.25,0.5)     paramslr,paramsl,params
   // --                [0.5,0.75)     paramsrl,paramsr,params
   // --                [0.75,1)       paramsrr,paramsr,params
   // --
   // --   parameters:
   // --     - data : the new observed data point of size [1 D]
   // --     - phi : the ordering
   // --------------------------------------------------------------------------
   void rem_data(arr(double) data);
   void add_data(arr(double) data);

   // --------------------------------------------------------------------------
   // -- add_data_init
   // --   adds the data to only the ll,lr,rl,rr components. Should only be used
   // -- for initial adding of data. After all initial data is added, a call to
   // -- update_upwards() should be called
   // --                phi range      NIWs to update
   // --                [0,0.25)       paramsll,paramsl,params
   // --                [0.25,0.5)     paramslr,paramsl,params
   // --                [0.5,0.75)     paramsrl,paramsr,params
   // --                [0.75,1)       paramsrr,paramsr,params
   // --
   // --   parameters:
   // --     - data : the new observed data point of size [1 D]
   // --     - phi : the ordering
   // --------------------------------------------------------------------------
   void add_data_init(arr(double) data);
   // --------------------------------------------------------------------------
   // -- update_upwards
   // --   Updates params,paramsl,paramsr, based on the ll,lr,rl,rr components.
   // --------------------------------------------------------------------------
   void update_upwards();

   void clear_lowest();
   void update_lowest();
   void fix_highest();

   double predictive_loglikelihood(arr(double) data);

   double data_loglikelihood_testmerge(cluster_single* &other);
   double data_loglikelihood();
};


#endif
