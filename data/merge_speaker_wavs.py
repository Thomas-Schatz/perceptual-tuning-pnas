# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 14:57:38 2017

@author: Thomas Schatz

Script to merge all wavs of a same speaker 
"""

from abkhazia.corpus.corpus import Corpus
import abkhazia.utils as utils
import os.path as path


def merge_speaker_wavs(in_path, out_path, padding):
    log = path.join(out_path, 'wav_merging.log')
    c = Corpus.load(in_path,
                    log=utils.logger.get_log(log_file=log, verbose=True))
    #FIXME remove unused corpus_dir arg from corpus.py and merge_wavs.py?
    c.merge_wavs(output_dir=out_path, padding=padding)
    #log = path.join(out_path, 'data_validation.log')
    #c = Corpus.load(out_path,
    #                log=utils.logger.get_log(log_file=log, verbose=True))    
    #c.validate()


if __name__ == '__main__':
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('in_path', help='path to corpus data folder')
    parser.add_argument('out_path', help=('path to merged wavs corpus'
                                          'data folder'))
    parser.add_argument('padding', type=float, default=0.,
                        help='silence amount between two merged file in sec.')
    args = parser.parse_args()
    assert path.exists(args.in_path)
    assert not(path.exists(args.out_path))
    merge_speaker_wavs(args.in_path, args.out_path, args.padding)