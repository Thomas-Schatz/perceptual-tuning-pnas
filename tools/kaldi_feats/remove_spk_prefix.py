# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 22:49:33 2017

@author: Thomas Schatz

Script for removing speaker prefixes from
item names in an h5features file.

More specifically: removes anything up to first '_'
character in item names.

Load the whole file in RAM.
"""

import h5features
import os.path as p

def remove_spk_prefix(in_file, out_file):
    data = h5features.Reader(in_file, 'features').read()
    new_items = []
    for item in data.items():
        ind = item.index('_')
        assert len(item)-1 > ind, "nothing after '_' in {}".format(item)
        new_items.append(item[ind+1:])
    data = h5features.Data(new_items, data.labels(), data.features(),
                           check=True)
    with h5features.Writer(out_file) as writer:
        writer.write(data, 'features')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('in_file', help = "input h5features file")
    parser.add_argument('out_file', help = "output h5features file")                 
    args = parser.parse_args()
    assert p.exists(args.in_file), "{} does not exist".format(args.in_file)
    assert not(p.exists(args.out_file)), \
           "{} already exists".format(args.out_file)
    remove_spk_prefix(args.in_file, args.out_file)