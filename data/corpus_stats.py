# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 11:29:36 2017

@author: Thomas Schatz

Compute and report statistics on the final corpora.
Pretty ad hoc.
"""

import os.path as p
import match_corpora as mc
import matplotlib.pyplot as plt


def get_stats(corpora_folders, gender_files):
    data = {}
    for subcorpus in ['train', 'test']:
        folders = {c: corpora_folders[c] + '_' + subcorpus
                   for c in corpora_folders}
        data[subcorpus] = mc.load_spk_matching_data(folders, gender_files)
    return data

def print_stats(out_folder, data, prefix=None):
    if prefix is None:
        prefix=''
    txt = p.join(out_folder, prefix+'stats.txt')
    pdf = p.join(out_folder, prefix+'stats.pdf')
    nb_r = 4
    nb_c = 2
    fig, axes = plt.subplots(nrows=nb_r, ncols=nb_c)
    with open(txt, 'w') as fh:
        for i_sub, subcorpus in enumerate(data):
            for i_cor, corpus in enumerate(data[subcorpus]):
                for i_gen, gender in enumerate(data[subcorpus][corpus]):
                    datum = data[subcorpus][corpus][gender]
                    fh.write('{} {}: {} {}\n'.format(corpus, subcorpus,
                                                     len(datum), gender))
                    durs = [d[1] for d in datum]
                    dur = sum(durs)
                    fh.write('Total duration: {} s\n'.format(dur))
                    axis = axes[i_cor+2*i_sub, i_gen]
                    axis.hist(durs, bins=5)
                    axis.set_xlabel('Speaker amount (in seconds)')
                    axis.set_ylabel('Nb speakers')
                    axis.set_title("{} {} {}".format(corpus, subcorpus,
                                                     gender))
    plt.tight_layout()
    plt.savefig(pdf)
