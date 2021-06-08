# -*- coding: utf-8 -*-
"""
Created on Sat May 27 10:04:05 2017

@author: Thomas Schatz

Extract posteriors from Jason Chang's library DPGMM model.
The models are treated as GMM not as DPGMM, following Chen et al.
(we consider that what was learnt is the MAP mixture instead of a whole
distribution over mixtures)
Input:
    h5features file containing features in 'features' group
    GMM from Jason Chang's library in .mat format
Output:
    h5features file with posteriorgrams
"""

import os.path as path
import scipy
import scipy.io as io
import numpy as np
import h5features as h5f


#######################
# Auxiliary functions #
#######################
# Code adapted from megamix python module (https://github.com/14thibea/megamix) by Elina Thibeau-Sutre

def _compute_precisions_chol(cov):
    n_components, n_features, _ = cov.shape
    precisions_chol = np.empty((n_components, n_features, n_features))
    for k, covariance in enumerate(cov):
        try:
            cov_chol = scipy.linalg.cholesky(covariance, lower=True)
        except scipy.linalg.LinAlgError:
            raise ValueError(str(k) + \
                             "-th covariance matrix non positive definite")
        precisions_chol[k] = scipy.linalg.solve_triangular(cov_chol,
                                                           np.eye(n_features),
                                                           lower=True).T
    return precisions_chol


def _log_normal_matrix(points, means, cov):
    """
    This method computes the log of the density of probability of a normal
    law centered. Each line
    corresponds to a point from points.
    
    @param points: an array of points (n_points,dim)
    @param means: an array of k points which are the means of the clusters
                  (n_components,dim)
    @param cov: an array of k arrays which are the covariance matrices
                (n_components,dim,dim)
    @return: an array containing the log of density of probability of a normal
             law centered (n_points,n_components)
    """
    n_points,dim = points.shape
    n_components,_ = means.shape
    precisions_chol = _compute_precisions_chol(cov)
    log_det_chol = np.log(np.linalg.det(precisions_chol))    
    log_prob = np.empty((n_points,n_components))
    for k, (mu, prec_chol) in enumerate(zip(means,precisions_chol)):
        y = np.dot(points,prec_chol) - np.dot(mu,prec_chol)
        log_prob[:,k] = np.sum(np.square(y), axis=1)
    return -.5 * (dim * np.log(2*np.pi) + log_prob) + log_det_chol


#######
# API #
#######

def load_model(filename):
    """
    Load GMM model saved in .mat from Jason Chang's library
    Output: 
        Dictionary with entries:
            cov : array of floats (n_components,dim,dim)
                Contains the computed covariance matrices of the mixture.
            
            means : array of floats (n_components,dim)
                Contains the computed means of the mixture.
            
            log_weights : array of floats (n_components,)
    """
    # file format is not consistent between Chang's init and update steps...
    data = io.loadmat(filename)
    sh = data['clusters'].shape
    if sh[0] == 1:
        K = sh[1]
        #dt = data['clusters'].dtype.descr
        #keys = [dt[i][0] for i in range(len(dt))]  # names of various descriptors for clusters
        model = {}        
        logpi = [data['clusters'][0,i]['logpi'] for i in range(K)]
        model['log_weights'] = np.concatenate(logpi).reshape((K,))  # (K,)
        mu = [data['clusters'][0,i]['mu'] for i in range(K)]
        model['means'] = np.column_stack(mu).T  # (K,d)
        d = model['means'].shape[1]
        Sigma = [data['clusters'][0,i]['Sigma'].reshape((1,d,d)) for i in range(K)]
        model['cov'] = np.concatenate(Sigma, axis=0)  # (K,d,d)
    else:
        K = sh[0]
        model = {}        
        logpi = [data['clusters'][i,0]['logpi'] for i in range(K)]
        model['log_weights'] = np.concatenate(logpi).reshape((K,))  # (K,)
        mu = [data['clusters'][i,0]['mu'] for i in range(K)]
        model['means'] = np.column_stack(mu).T  # (K,d)
        d = model['means'].shape[1]
        Sigma = [data['clusters'][i,0]['Sigma'].reshape((1,d,d)) for i in range(K)]
        model['cov'] = np.concatenate(Sigma, axis=0)  # (K,d,d)
    return model


def compute_log_posteriors(points, model):
    """
    Returns responsibilities for each
    point in each cluster
    
    Parameters
    ----------
    points : an array (n_points,dim)
    model : GMM model, formatted like the output from load_model
    
    Returns
    -------
    log_resp: an array (n_points,n_components)
        an array containing the logarithm of the responsibilities.            
    """
    log_normal_matrix = _log_normal_matrix(points,
                                           model['means'],
                                           model['cov'])
    log_product = log_normal_matrix + model['log_weights'][:,np.newaxis].T
    log_prob_norm = scipy.misc.logsumexp(log_product,axis=1)
    log_resp = log_product - log_prob_norm[:,np.newaxis]       
    return log_resp


def extract_posteriors(features_file, model_file, output_file):
    print('Loading model')
    model = load_model(model_file)
    print('Model with {} clusters loaded'.format(len(model['log_weights'])))
    print('Loading input features')
    with h5f.Reader(features_file, 'features') as reader:
        data = reader.read()
    input_features = data.features()
    nbItems = len(input_features)
    post_features = []
    for i, feats in enumerate(input_features):
        print('Computing posteriors for item {} of {}'.format(i+1, nbItems))
        post_features.append(np.exp(compute_log_posteriors(feats, model)))
    post_data = h5f.Data(data.items(), data.labels(), post_features,
                         check=True)
    print('Writing posteriors to disk')
    with h5f.Writer(output_file) as writer:
        writer.write(post_data, 'features')


def extract_posteriors_lowmem(features_file, model_file, output_file):
    # just like extract_posteriors but writing item by item
    # if this is too slow, could do a version by blocks of items
    print('Loading model')
    model = load_model(model_file)
    print('Model with {} clusters loaded'.format(len(model['log_weights'])))
    print('Loading input features')
    with h5f.Reader(features_file, 'features') as reader:
        data = reader.read()
    input_features = data.features()
    nbItems = len(input_features)
    post_features = []
    assert not(path.exists(output_file))
    with h5f.Writer(output_file) as writer:
        for i, (item, label, feats) in enumerate(zip(data.items(),
                                                     data.labels(),
                                                     input_features)):
            print('Computing posteriors for item {} of {}'.format(i+1, nbItems))
            post_features = [np.exp(compute_log_posteriors(feats, model))]
            post_data = h5f.Data([item], [label], post_features, check=True)
            print('Writing posteriors to disk')
            writer.write(post_data, 'features', append=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('features_file', help = "h5features file" + \
                                                "containing input features")
    parser.add_argument('model_file', help = "mat file containing" + \
                                             "GMM model from Jason Chang's" + \
                                             "library")
    parser.add_argument('output_file', help = "h5features file where" + \
                                              "extracted posteriorgrams" + \
                                              "will be stored")                 
    args = parser.parse_args()
    extract_posteriors_lowmem(args.features_file, args.model_file, args.output_file)
