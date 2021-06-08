# -*- coding: utf-8 -*-
# Copyright 2015, 2016 Thomas Schatz
#
# This file is part of abkhazia: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Abkhazia is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with abkhazia. If not, see <http://www.gnu.org/licenses/>.


import codecs
import pandas as pd
import numpy as np
import ABXpy.database.database as database


# Note that randomness is probably reproducible on the same machine
# with the same python version, but I'm not sure what would happen on
# different machines or with different python versions
def threshold_item(item_file, output_file, columns,
                   lower_threshold=1, upper_threshold=np.inf, seed=0):
    """Randomly sample items in item_file in order to limit the number of
    element in each cell to upper_threshold, where a cell is defined
    as a unique value of the specified columns
    """
    np.random.seed(seed)
    # read input file
    with codecs.open(item_file, mode='r', encoding='UTF-8') as inp:
        header = inp.readline()
    db, _, feat_db = database.load(item_file, features_info=True)
    db = pd.concat([feat_db, db],axis=1)
    # group and sample
    with codecs.open(output_file, mode='w', encoding='UTF-8') as out:
        out.write(header)
        for group, df in db.groupby(columns):
            if len(df) >= lower_threshold:
                # shuffle dataframe
                df = df.reindex(np.random.permutation(df.index))
                m = min(upper_threshold, len(df))
                df = df.iloc[:m]
                for i in range(m):
                    out.write(u" ".join([unicode(e) for e in
                                         df.iloc[i]]) + u"\n")


root='/scratch1/users/thomas/perceptual-tuning-data/eval/abx/'
corpora=['BUC','CSJ', 'WSJ', 'GPJ']


for corpus in corpora:
    item_file=root+corpus+'.item'
    lower_threshold = 2
    upper_threshold = 5
    out_file = root + corpus + '_threshold_' + str(lower_threshold) + '_' + str(upper_threshold) + '.item'
    columns = ['phone', 'prev-phone', 'next-phone', 'speaker']  # ['phone', 'talker']
    print "****"+item_file
    threshold_item(item_file, out_file, columns,
                 lower_threshold=lower_threshold,
                 upper_threshold=upper_threshold)

 
