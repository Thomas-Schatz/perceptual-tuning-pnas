# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 13:39:10 2017

@author: Thomas Schatz

Script to generate item files for ABX tasks
"""


from abkhazia.corpus.corpus import Corpus
import abkhazia.utils.abkhazia2abx as abk2abx
import os.path as path


root = '/scratch1/users/thomas/perceptual-tuning-data/'


# BUC
corpus = Corpus.load(path.join(root, 'corpora', 'BUC',
                               'CSJ_matched_data_test'))
alignment_file = path.join(root, 'corpora', 'BUC',
                           'aligned_data', 'alignment.txt')
item_file = path.join(root, 'eval', 'abx', 'BUC.item')
abk2abx.alignment2item(corpus, alignment_file, item_file, 'single_phone')

# CSJ
corpus = Corpus.load(path.join(root, 'corpora', 'CSJ',
                               'BUC_matched_data_test'))
alignment_file = path.join(root, 'corpora', 'CSJ',
                           'aligned_data', 'alignment.txt')
item_file = path.join(root, 'eval', 'abx', 'CSJ.item')
abk2abx.alignment2item(corpus, alignment_file, item_file, 'single_phone')

# GPJ
corpus = Corpus.load(path.join(root, 'corpora', 'WSJ',
                               'GPJ_matched_data_test'))
alignment_file = path.join(root, 'corpora', 'WSJ',
                           'aligned_data', 'alignment.txt')
item_file = path.join(root, 'eval', 'abx', 'WSJ.item')
abk2abx.alignment2item(corpus, alignment_file, item_file, 'single_phone')

# WSJ
corpus = Corpus.load(path.join(root, 'corpora', 'GPJ',
                               'WSJ_matched_data_test'))
alignment_file = path.join(root, 'corpora', 'GPJ',
                           'aligned_data', 'alignment.txt')
item_file = path.join(root, 'eval', 'abx', 'GPJ.item')
abk2abx.alignment2item(corpus, alignment_file, item_file, 'single_phone')
