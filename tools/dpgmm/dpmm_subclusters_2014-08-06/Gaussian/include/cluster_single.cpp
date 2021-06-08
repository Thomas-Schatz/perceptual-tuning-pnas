#include "cluster_single.h"

#ifndef pi
#define pi 3.14159265
#endif

// --------------------------------------------------------------------------
// -- cluster_single
// --   constructor; initializes to empty
// --------------------------------------------------------------------------
cluster_single::cluster_single()
{
}

// --------------------------------------------------------------------------
// -- cluster_single
// --   copy constructor;
// --------------------------------------------------------------------------
cluster_single::cluster_single(const cluster_single& that)
{
   copy(that);
}
// --------------------------------------------------------------------------
// -- operator=
// --   assignment operator
// --------------------------------------------------------------------------
cluster_single& cluster_single::operator=(const cluster_single& that)
{
   if (this != &that)
      copy(that);
   return *this;
}
// --------------------------------------------------------------------------
// -- copy
// --   returns a copy of this
// --------------------------------------------------------------------------
void cluster_single::copy(const cluster_single& that)
{
   params = that.params;
}

// --------------------------------------------------------------------------
// -- cluster_single
// --   constructor; intializes to all the values given
// --------------------------------------------------------------------------
cluster_single::cluster_single(int _D, double _kappah, double _nuh, arr(double) _thetah, arr(double) _Deltah) :
   params(true, _D, _kappah, _nuh, _thetah, _Deltah)
{
}

// --------------------------------------------------------------------------
// -- cluster_single
// --   copy that into all params
// --------------------------------------------------------------------------
cluster_single::cluster_single(const niw& that) :
   params(that)
{
}



// --------------------------------------------------------------------------
// -- ~cluster_single
// --   destructor
// --------------------------------------------------------------------------
cluster_single::~cluster_single()
{
}


// --------------------------------------------------------------------------
// -- empty
// --   Empties out the cluster_single. Does not update the posterior hyperparameters.
// --------------------------------------------------------------------------
void cluster_single::empty()
{
   params.empty();
}

bool cluster_single::isempty() { return params.isempty();}
int cluster_single::getN()     { return params.getN();}

niw* cluster_single::get_params()    { return &params;}
void cluster_single::set_params(niw* _params)   { params = *_params;}


// --------------------------------------------------------------------------
// -- update
// --   Updates the posterior hyperparameters and the parameters for the
// -- predictive moment-matched gaussian
// --------------------------------------------------------------------------
void cluster_single::update()
{
   params.update();
}


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
void cluster_single::rem_data(arr(double) data)
{
   params.rem_data(data);
}
void cluster_single::add_data(arr(double) data)
{
   params.add_data(data);
}

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
void cluster_single::add_data_init(arr(double) data)
{
   params.add_data_init(data);
}
// --------------------------------------------------------------------------
// -- update_upwards
// --   Updates params,paramsl,paramsr, based on the ll,lr,rl,rr components.
// --------------------------------------------------------------------------
void cluster_single::update_upwards()
{
   params.update();
}

void cluster_single::clear_lowest()
{
   params.empty();
}
void cluster_single::update_lowest()
{
   params.update();
}
void cluster_single::fix_highest()
{
   params.update();
}


double cluster_single::predictive_loglikelihood(arr(double) data)
{
   return params.predictive_loglikelihood(data);
}



double cluster_single::data_loglikelihood_testmerge(cluster_single* &other)
{
   niw temp = params;
   temp.merge_with(other->params);
   return temp.data_loglikelihood();
}

double cluster_single::data_loglikelihood()
{
   return params.data_loglikelihood();
}