# -*- coding: utf-8 -*-
"""
Created on Wed May 17 08:42:42 2017

@author: Thomas Schatz

Convert a non-compressed text format kaldi ark file containing speech features
indexed by utt-id into an h5features file.

Could use abkhazia.utils.kaldi.{ark, scp}_to_h5f instead, which is more general
but for now this is intricated into the whole abkhazia which seems too heavy
duty (for example a kaldi install is not necessary to do the conversion).
"""

import io
import numpy as np
import h5features


def kaldif2h5f(in_file, out_file):
    """
    kaldi input features (mfcc, etc.) to h5features
    this loads everything into memory, but it would be easy to write
    an incremental version if this poses a problem
    Input features must be in a single archive text format, that can be
    obtained using the 'copy-feats' kaldi utility
    """
    # below is basically a parser for kaldi vector format for each line
    # parse input text file
    outside_utt = True
    features = []
    utt_ids = []
    times = []
    with io.open(in_file, 'r', encoding='utf8') as inp:
        for index, line in enumerate(inp):
            print("Processing line {0}".format(index+1))
            # / {1}".format(index+1, len(lines)))

            tokens = line.strip().split(u" ")
            if outside_utt:
                assert (len(tokens) == 3 and
                        tokens[1] == u"" and
                        tokens[2] == u"[")
                utt_id = tokens[0]
                outside_utt = False
                frames = []
            else:
                if tokens[-1] == u"]":
                    # end of utterance
                    outside_utt = True
                    tokens = tokens[:-1]
                frames.append(np.array(tokens, dtype=np.float))
                if outside_utt:
                    # end of utterance, continued
                    features.append(np.row_stack(frames))
                    # as in kaldi2abkhazia, this is ad hoc and has not
                    # been checked formally
                    times.append(0.0125 + 0.01*np.arange(len(frames)))
                    utt_ids.append(utt_id)
    h5features.write(out_file, 'features', utt_ids, times, features)

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('source_ark_file', help="non-compressed text " + \
                                                "kaldi ark file containing" + \
                                                " speech features indexed" + \
                                                " by utt-id")
    parser.add_argument('target_h5f_file', help='target h5 features file')
    args = parser.parse_args()
    kaldif2h5f(args.source_ark_file, args.target_h5f_file)
        