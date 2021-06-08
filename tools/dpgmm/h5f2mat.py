# -*- coding: utf-8 -*-
"""
Created on Wed May 24 09:10:24 2017

@author: Thomas Schatz

Export data from h5features file to Matlab .mat file.
All the data is randomly shuffled together, independently of the
original grouping and order in the h5features file. The random
order is not saved.

Loads the whole features file in RAM.

The vad_file option can be used to provide an abkhazia formatted 'segments.txt'
file, in which case only features with center times within the designated
segments are retained. Only segments file with a start and end time specified
on every line are supported at present.
"""

import numpy as np
import h5features
import scipy.io as sio
import io


def h5f2mat(h5feat_file, mat_file, vad_file=None):
    data = h5features.Reader(h5feat_file, 'features').read()
    if vad_file is None:
        data = np.concatenate(data.features())  # drop labels
    else:
        features = data.dict_features()
        times = data.dict_labels()
        data = []
        with io.open(vad_file, 'r', encoding='utf-8') as fh:
            for line in fh:
                utt_id, wav_id, start, stop = line.strip().split(u" ")
                assert wav_id[-4:] == '.wav'
                wav_id = wav_id[:-4]
                start, stop = float(start), float(stop)
                print(utt_id)
                t = times[wav_id]
                inds = np.where(np.logical_and(t>=start, t<=stop))[0]
                data.append(features[wav_id][inds,:])
        data = np.concatenate(data)
    np.random.shuffle(data)  # drop order
    # cutting data into a bunch of smaller variables (if needed) to accomodate
    # matlab file format size limitations
    max_size = 2**30  # in bytes
    max_nrow = max_size // (data.dtype.itemsize*data.shape[1])
    nb_iter = data.shape[0] // max_nrow
    remainder = data.shape[0] % max_nrow
    d = dict()
    for i in range(nb_iter):
        d['data'+str(i+1)] = data[i*max_nrow:(i+1)*max_nrow, :]
    if remainder > 0:
        d['data'+str(nb_iter+1)] = data[nb_iter*max_nrow:, :]
    sio.savemat(mat_file, d)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('h5feat_file', help="h5features file")
    parser.add_argument('mat_file', help="mat file")
    parser.add_argument('--vad_file', default=None,
                        help=("optional vad file in "
                              "abkhazia 'segments.txt' format"))                 
    args = parser.parse_args()
    h5f2mat(args.h5feat_file, args.mat_file, args.vad_file)