# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 12:59:50 2017

@author: Thomas Schatz

Given an abkhazia-formatted corpus and a corresponding alignment file,
creates a subcorpus containing only the utterances present in the alignment
and copy the alignment file within this subcorpus

Requires a working abkhazia install to load the corpus object
"""

import abkhazia.corpus
import os, shutil, io


def create_subcorpus(input_corpus, alignment_file, output_folder):
    # get set of utt_ids present in alignment file
    with io.open(alignment_file, mode='r', encoding='utf-8') as fh:
        utt_ids = {line.strip().split(u' ')[0] for line in fh}
    # get set of utt_ids in original corpus
    segments_file = os.path.join(input_corpus, 'segments.txt')
    with io.open(segments_file, mode='r', encoding='utf-8') as fh:
        all_utt_ids = {line.strip().split(u' ')[0] for line in fh}
    # get files to be removed (for logging purpose)
    unknown_utts = utt_ids.difference(utt_ids)
    removed_utts = all_utt_ids.difference(utt_ids)
    assert unknown_utts == set(), \
           ("Alignment contains utt_ids "
            "not present in corpus: {}").format(unknown_utts)
    # load corpus
    corpus = abkhazia.corpus.Corpus.load(input_corpus, validate=True)
    # instantiate subcorpus
    subcorpus = corpus.subcorpus(utt_ids, prune=True,
                                 name=corpus.meta.name+'-aligned',
                                 validate=True)
    # save subcorpus
    subcorpus.save(output_folder, copy_wavs=False)
    # copy alignment file to subcorpus folder
    shutil.copy(alignment_file, output_folder)
    # append removed files to log
    log_file = os.path.join(output_folder, 'log.txt')
    nb_removed = len(removed_utts)
    with io.open(log_file, mode='a', encoding='utf-8') as fh:
        fh.write(u'Removed {} utts without alignment: {}'.format(nb_removed,
                                                                 removed_utts))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('input_corpus', help='input corpus data folder path')
    parser.add_argument('alignment_file', help='alignment file')
    parser.add_argument('output_folder', help=('folder where to instantiate '
                                                'the aligned subcorpus'))
    args = parser.parse_args()
    assert os.path.exists(args.input_corpus), \
            "Invalid input corpus folder {}".format(args.input_corpus)
    assert os.path.exists(args.alignment_file), \
            "Invalid alignment file {}".format(args.alignment_file)
    assert not(os.path.exists(args.output_folder)), \
            "Output folder already exists: {}".format(args.output_folder)
    create_subcorpus(args.input_corpus, args.alignment_file,
                     args.output_folder)

