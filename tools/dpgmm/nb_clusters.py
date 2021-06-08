import scipy.io as io
import numpy as np


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


def get_nb_Gaussians(model_path):
    model = load_model(model_path)
    n = len(model['log_weights'])
    return n


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', help="path to model .mat file")                 
    args = parser.parse_args()
    n = get_nb_Gaussians(args.filename)
    print(n)