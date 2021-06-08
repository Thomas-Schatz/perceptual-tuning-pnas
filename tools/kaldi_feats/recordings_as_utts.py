# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 12:59:06 2017

@author: Thomas Schatz

Given an abkhazia formatted corpus, 
create utt2spk.txt and segments.txt
appropriate for kaldi to consider whole
recordings as utterances.
This puts spk_id as a prefix to recording_ids to allow
the use of whole recording as kaldi utterances.
"""

import os
import os.path as p
import io
import shutil
import subprocess


def cpp_sort(filename):
    # there is redundancy here but I didn't check which export can be 
    # safely removed, so better safe than sorry
    os.environ["LC_ALL"] = "C"
    subprocess.call("export LC_ALL=C; sort {0} -o {1}".format(filename,
                                                              filename),
                    shell=True, env=os.environ)
    # could use the following to do the sorting on python data structures:
    #   import locale; locale.setlocale(locale.LC_ALL, "C")
    #   sorted(list, cmp=locale.strcoll)


def recordings_as_utts(corpus_path, out_folder):
    seg_file = p.join(corpus_path, 'segments.txt')
    utt2spk_file = p.join(corpus_path, 'utt2spk.txt')
    # parse segments
    segs = {}
    with io.open(seg_file, 'r', encoding='utf-8') as fh:
        for line in fh:
            utt_id, seg_id = line.strip().split(u' ')[:2]
            assert seg_id[-4:] == '.wav'
            seg_id = seg_id[:-4]
            if seg_id in segs:
                segs[seg_id].append(utt_id)
            else:
                segs[seg_id] = [utt_id]
    # parse speakers
    with io.open(utt2spk_file, 'r', encoding='utf-8') as fh:
        utt2spk = {}
        for line in fh:
            utt_id, spk_id = line.strip().split(u' ')
            utt2spk[utt_id] = spk_id
    # check there is only one speaker per recording
    seg2spk = {}
    for seg_id in segs:
        speakers = {utt2spk[utt_id] for utt_id in segs[seg_id]}
        assert len(speakers) == 1, speakers
        seg2spk[seg_id] = speakers.pop()
    # check if recordings id have speaker id as prefix
    correct_prefix = True
    for seg_id in segs:
        spk = seg2spk[seg_id]
        correct_prefix = seg_id[:len(spk)] == spk
        if not(correct_prefix):
            break
    # create seg2spk file
    seg2spk_file = p.join(out_folder, 'utt2spk.txt')
    with io.open(seg2spk_file, 'w', encoding='utf-8') as fh:
        for seg_id in seg2spk:
            if correct_prefix:
                fh.write(u'{} {}\n'.format(seg_id, seg2spk[seg_id]))
            else:
                spk = seg2spk[seg_id]
                fh.write(u'{} {}\n'.format(spk + '_' + seg_id, spk))
    cpp_sort(seg2spk_file)
    # create segments file
    out_seg_file = p.join(args.out_folder, 'segments.txt')
    with io.open(out_seg_file, 'w', encoding='utf-8') as fh:
        for seg_id in segs:
            if correct_prefix:
                fh.write(u'{} {}.wav\n'.format(seg_id, seg_id))
            else:
                spk = seg2spk[seg_id]
                fh.write(u'{} {}.wav\n'.format(spk+ '_' + seg_id, seg_id))  
    cpp_sort(out_seg_file)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus_path", help=('Path to abkhazia formatted '
                                             'corpus data folder'))
    parser.add_argument("out_folder", help=('Folder where to save '
                                            'results'))
    args = parser.parse_args()
    assert p.exists(args.corpus_path), \
           "{} does not exist".format(args.corpus_path)
    f = p.join(args.out_folder, 'utt2spk.txt')
    assert not(p.exists(f)), \
           "{} already exists".format(f)
    f = p.join(args.out_folder, 'segments.txt')
    assert not(p.exists(f)), \
           "{} already exists".format(f) 
    recordings_as_utts(args.corpus_path, args.out_folder)