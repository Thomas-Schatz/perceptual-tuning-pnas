# -*- coding: utf-8 -*-
"""
Created on Sun Dec 31 15:52:14 2017

@author: Thomas Schatz


Script converting h5features files indexed by recording to h5features files
indexed by utterance. It uses a spoCk formatted segments.txt file to locate
utterances inside recordings. All utterances start and end times must be
specified in the segments.txt file for this scrit to work properly.

Dependencies: python 2.7+ or 3.5+, numpy, h5features (available on pip)

Usage: python feats_rec2utt.py in_feats segments out_feats
"""


import h5features as h5f
import numpy as np
import io


def load_segments(segments):
    """
    Returns a dict indexed by recordings (without .wav) and containing list
    of associated utt_labels, utt_starts and utt_ends (ordered chronologically)
    
    This function will work only if utterance start and end times are specified
    in the segments file
    """

    segs = {}
    with io.open(segments, 'r', encoding='UTF-8') as fh:
        for line in fh:
            tokens = line.strip().split()
            utt_id = tokens[0]
            assert tokens[1][-4:] == ".wav"
            rec_id = tokens[1][:-4]
            assert len(tokens) == 4, ("Missing utterance start/end times " + \
                                      "for utterance {} in file " + \
                                      "{} ".format(utt_id, segments))
            tstart, tend = float(tokens[2]), float(tokens[3])
            try:
                segs[rec_id].append((utt_id, tstart, tend))
            except KeyError:
                segs[rec_id] = [(utt_id, tstart, tend)]
    return segs


def rec2utt_indexing(in_feats, segments, out_feats):
    # load segment file and get utterances by recording
    recordings = load_segments(segments)
    with h5f.Reader(in_feats, 'features') as reader:
        with h5f.Writer(out_feats) as writer:
            # read and write recording by recording
            for rec in recordings:
                data = reader.read(rec)
                times, feats = data.labels()[0], data.features()[0]
                # for each recording get items, labels, feats
                # lists indexed by utt 
                utt_items, utt_times, utt_feats = [], [], []                
                for utt, utt_start, utt_end in recordings[rec]:
                    utt_items.append(utt)
                    t = np.where(np.logical_and(times >= utt_start,
                                                times <= utt_end))[0]
                    utt_times.append(times[t]-utt_start)
                    utt_feats.append(feats[t,:])    
                out_data = h5f.Data(utt_items, utt_times, utt_feats, check=True)
                writer.write(out_data, 'features', append=True)


if __name__ == '__main__':
    import argparse
    import os.path as path
    parser = argparse.ArgumentParser()
    parser.add_argument('in_feats', help = "h5features file " + \
                                           "containing input features " + \
                                           "indexed by recording")
    parser.add_argument('segments', help = "spoCk-formatted " + \
                                           "'segments.txt' file " + \
                                           "giving utterance positions " + \
                                           "in recordings")
    parser.add_argument('out_feats', help = "h5features file where " + \
                                            "utterance-indexed features " + \
                                            "will be stored")                 
    args = parser.parse_args()
    assert path.isfile(args.in_feats), "No such file {}".format(args.in_feats)
    assert path.isfile(args.segments), "No such file {}".format(args.segments)
    assert not(path.exists(args.out_feats)), \
           "Output path {} already occupied".format(args.out_feats)        
    rec2utt_indexing(args.in_feats, args.segments, args.out_feats)