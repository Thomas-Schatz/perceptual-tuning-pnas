# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 14:28:04 2018

@author: Thomas Schatz

Compute distances, scores and results given task and features.

Usage: 
    python run_abx.py feat_file task_file res_folder res_id distance normalized
    
    where 'distance' is 'kl' or 'cos' and 'normalized' is a boolean
"""


import ABXpy.distances.distances as dis
import ABXpy.score as sco
import ABXpy.analyze as ana
import ABXpy.distances.metrics.dtw as dtw
import ABXpy.distances.metrics.kullback_leibler as kl
import ABXpy.distances.metrics.cosine as cos
import scipy.spatial.distance as euc
import numpy as np


def run_ABX(feat_file, task_file, dis_file, score_file, result_file, distance,
            normalized):
    """
    Run distances, scores and results ABXpy steps based on
    provided features and task files.
    Results are saved in:
        $res_folder/distances/'$res_id'.distances
        $res_folder/scores/'$res_id'.scores
        $res_folder/results/'$res_id'.txt
    """
    dis.compute_distances(feat_file, '/features/', task_file, dis_file,
                          distance, normalized=normalized, n_cpu=1)
    sco.score(task_file, dis_file, score_file)
    ana.analyze(task_file, score_file, result_file)


if __name__=='__main__':
    import argparse
    import os.path as path
    parser = argparse.ArgumentParser()
    parser.add_argument('feat_file', help='h5features file')
    parser.add_argument('task_file', help='ABXpy task file')
    parser.add_argument('res_folder', help=('Result folder (must contain'
                                            'distances, scores and results'
                                            'subfolders)'))
    parser.add_argument('res_id', help=('identifier for the results'
                                        '(model + task)'))
    parser.add_argument('distance', help='kl or cos or dur or euc')
    parser.add_argument('normalized', type=bool,
                        help=('if true, take mean distance along'
                              'dtw path length instead of sum'))
    args = parser.parse_args()
    assert path.exists(args.feat_file), ("No such file "
                                         "{}".format(args.feat_file))
    assert path.exists(args.task_file), ("No such file "
                                         "{}".format(args.task_file))
    dis_file = path.join(args.res_folder, 'distances',
                         args.res_id + '.distances')    
    score_file = path.join(args.res_folder, 'scores',
                           args.res_id + '.scores')
    result_file = path.join(args.res_folder, 'results',
                            args.res_id + '.txt')
    assert not(path.exists(dis_file)), \
        "{} already exists".format(dis_file)
    assert not(path.exists(score_file)), \
        "{} already exists".format(score_file)
    assert not(path.exists(result_file)), \
        "{} already exists".format(result_file)
    assert args.distance in ['kl', 'cos', 'dur', 'euc'], \
        "Distance function {} not supported".format(args.distance)
    if args.distance == 'kl':
        frame_dis = kl.kl_divergence
    elif args.distance == 'cos':
        frame_dis = cos.cosine_distance
    elif args.distance == 'euc':
        frame_dis = lambda x, y: euc.cdist(x, y, 'euclidean')
    if args.distance in ['kl', 'cos', 'euc']:
        distance = lambda x, y, normalized: dtw.dtw(x, y, frame_dis,
                                                    normalized=normalized)
    else:
        distance = lambda x, y, normalized: np.abs(x.shape[0]-y.shape[0])
    run_ABX(args.feat_file, args.task_file, dis_file, score_file, result_file,
            distance, args.normalized)

