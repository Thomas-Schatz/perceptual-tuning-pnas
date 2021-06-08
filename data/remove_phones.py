# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 13:39:10 2017

@author: Thomas Schatz

Script to remove infrequent GPJ phones
"""

from abkhazia.corpus.corpus import Corpus
import abkhazia.utils as utils
import os.path as path


def remove_phones(in_path, out_path, phones=None, silences=None):
    log = path.join(out_path, 'data_validation.log')
    c = Corpus.load(in_path,
                    log=utils.logger.get_log(log_file=log, verbose=True))
    c2 = c.remove_phones(phones=phones, silences=silences)
    c2.validate()
    c2.save(out_path, copy_wavs=False)


#root = '/scratch1/users/thomas/perceptual-tuning-data/corpora/WSJ'
#in_path = path.join(root, 'data')
#out_path = path.join(root, 'data2')
#remove_phones(in_path, out_path, silences=['NSN'])


root = '/scratch1/users/thomas/perceptual-tuning-data/corpora/GPJ/japanese'
in_path = path.join(root, 'data')
out_path = path.join(root, 'data2')
phones = ['Q+b', 'Q+z', 'Q+h', 'Q+d', 'Q+z+y', 'Q+g', 'Q+F', 'Q+c']
remove_phones(in_path, out_path, phones=phones)