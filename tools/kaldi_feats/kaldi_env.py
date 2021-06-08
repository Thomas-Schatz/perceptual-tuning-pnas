# -*- coding: utf-8 -*-
"""
Created on Sun May 14 23:02:10 2017

@author: Thomas Schatz

Create an execution environment for running kaldi feature extraction routines.

This reuses code from old and current abkhazia and ideally should
be integrated within a low-level abkhazia2kaldi (or spoCk2kaldi) interface,
but for now there is only a monolithic abkhazia that requires transcription
files that are unnecessary here.

Creating features conf files (mfcc.conf, pitch.conf, etc.) is not the
responsibility of this module.
"""

import os
import os.path as p
import io
import shutil
import subprocess


def setup_wav(recipe_path, data_path, segments):
    # get list of wavs from segments.txt
    # this used with whole recordings as utterances
    # so there is only one entry in segments file per wav
    segs = []
    with io.open(segments, mode='r', encoding='utf8') as f:
        for line in f:
            segs.append(line.strip().split(u" "))
    # write wav.scp 
    with io.open(p.join(data_path, 'wav.scp'), 'w', encoding='utf8') as f:
        for record_id, wav_id in segs:
            wav_name = wav_id
            wav_full_path = p.join(p.abspath(recipe_path), 'wavs', wav_name)
            f.write(u"{0} {1}\n".format(record_id, wav_full_path))


def setup_segments(data_path, segments):
    with io.open(segments, 'r', encoding='utf8') as f:
        lines = f.readlines()
    # for now segments files where some beginning and endings of segments
    # are not explicitly specified is not supported
    # if this needs to be fixed, just add a module checking the wavs to
    # complete missing beginnings and endings
    missing_time_tags = [len(line.strip().split(u" ")) != 4 for line in lines]
    if any(missing_time_tags):
        raise NotImplementedError("Segments files with missing begin and " + \
                                  "end times are not yet supported : " + \
                                  segments)
    with io.open(p.join(data_path, 'segments'), 'w', encoding='utf8') as f:
        for line in lines:
            elements = line.strip().split(u" ")
            utt_id, wav_id = elements[:2]
            record_id = p.splitext(wav_id)[0]
            f.write(u" ".join([utt_id, record_id] + elements[2:]) + u"\n")


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


def setup_cmd(recipe_path):
    template = u"export train_cmd=\"queue.pl -q all.q@puck*.cm.cluster\"\n" + \
               u"export decode_cmd=\"queue.pl -q all.q@puck*.cm.cluster\"\n" + \
               u"export highmem_cmd=\"queue.pl -q all.q@puck*.cm.cluster\"\n"
    with io.open(p.join(recipe_path, 'cmd.sh'), 'w', encoding='utf8') as f:
        f.write(template)


def setup_path(recipe_path, kaldi_root):
    # in particular get kaldiroot + optional stuff from there
    # for oberon kaldi install 2017 I used a conda gcc, so I had to put
    # correct LD_LIBRARY_PATH in path.sh below
    template = \
u"""
export LC_ALL=C  # For expected sorting and joining behaviour
export LD_LIBRARY_PATH=/home/thomas/.conda/envs/gcc/lib:$LD_LIBRARY_PATH
KALDISRC=$KALDI_ROOT/src
KALDIBIN=$KALDISRC/bin:$KALDISRC/featbin:$KALDISRC/fgmmbin:$KALDISRC/fstbin
KALDIBIN=$KALDIBIN:$KALDISRC/gmmbin:$KALDISRC/latbin
KALDIBIN=$KALDIBIN:$KALDISRC/sgmmbin:$KALDISRC/lmbin
KALDIBIN=$KALDIBIN:$KALDISRC/nnetbin:$KALDISRC/nnet2bin
KALDIBIN=$KALDIBIN:$KALDISRC/kwsbin:$KALDISRC/ivectorbin
KALDIBIN=$KALDIBIN:$KALDISRC/online2bin:$KALDISRC/sgmm2bin

FSTBIN=$KALDI_ROOT/tools/openfst/bin

PLATFORM=i686-m64  # default path for Unix machines
[ $(uname) == "Darwin" ] && PLATFORM=macosx
LMBIN=$KALDI_ROOT/tools/irstlm/bin:$KALDI_ROOT/tools/srilm/bin/$PLATFORM
LMBIN=$LMBIN:$KALDI_ROOT/tools/srilm/bin/:$KALDI_ROOT/tools/sctk/bin/

#[ -d $PWD/local ] || { echo "$0: 'local' subdirectory not found."; }
[ -d $PWD/utils ] || { echo "$0: 'utils' subdirectory not found."; }
[ -d $PWD/steps ] || { echo "$0: 'steps' subdirectory not found."; }

export kaldi_local=$PWD/local
export kaldi_utils=$PWD/utils
export kaldi_steps=$PWD/steps
export IRSTLM=$KALDI_ROOT/tools/irstlm 	# for LM building
SCRIPTS=$kaldi_local:$kaldi_utils:$kaldi_steps

export PATH=$PATH:$KALDIBIN:$FSTBIN:$LMBIN:$SCRIPTS
"""
    path_script = "KALDI_ROOT={}\n".format(kaldi_root) + template
    with io.open(p.join(recipe_path, 'path.sh'), 'w', encoding='utf8') as f:
        f.write(path_script)


def setup_kaldi_env(kaldi_root, recipe_path, wav_path, utt2spk, segments):
    """
    Setup an execution environment for running kaldi feature extraction scripts
    using whole recordings as utts

    Parameters
    ----------
    kaldi_root : str
        path to kaldi install (folder containing the 'egs' folder) 
    recipe_path : str 
        path where to setup kaldi environment
    wav_path : str 
        path to folder containing wavefiles, in abkhazia format
    utt2spk : str
        path to utt2spk file in abkhazia format (containing whole recordings as utts)
    segments : str
        path to segments file in abkhazia format (containing whole recordings as utts)
    """
    # create recipe directory
    if p.exists(recipe_path):
        raise IOError("Cannot setup new kaldi environment in " + \
                      "{}, directory already exists.".format(recipe_path))
    else:
        os.mkdir(recipe_path)
    # symlink standard steps and utils folders (to access kaldi bash scripts)
    steps_dir = p.abspath(p.join(kaldi_root, 'egs', 'wsj', 's5', 'steps'))
    steps_link = p.abspath(p.join(recipe_path, 'steps'))
    os.symlink(steps_dir, steps_link)
    utils_dir = p.abspath(p.join(kaldi_root, 'egs', 'wsj', 's5', 'utils'))
    utils_link = p.abspath(p.join(recipe_path, 'utils'))
    os.symlink(utils_dir, utils_link)
    # symlink wav folder
    wav_link = p.join(recipe_path, 'wavs')
    os.symlink(wav_path, wav_link)
    # create data folder
    data_path = p.join(recipe_path, 'data', 'main')
    os.makedirs(data_path)
    # intantiate wav.scp file
    setup_wav(recipe_path, data_path, segments)
    # instantiate utt2spk file
    shutil.copy(utt2spk, p.join(data_path, 'utt2spk'))
    # Sort files in C++ order to comply with kaldi requirements  
    for f in ['wav.scp', 'segments', 'utt2spk']:
        path = p.join(data_path, f)
        if p.exists(path):
            cpp_sort(path)
    setup_cmd(recipe_path)
    setup_path(recipe_path, kaldi_root)


if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('kaldi_root')
    parser.add_argument('recipe_path')
    parser.add_argument('wav_path')
    parser.add_argument('utt2spk')
    parser.add_argument('segments')
    args = parser.parse_args()
    setup_kaldi_env(args.kaldi_root, args.recipe_path, args.wav_path,
                    args.utt2spk, args.segments)
        