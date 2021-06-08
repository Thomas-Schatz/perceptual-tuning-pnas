# -*- coding: utf-8 -*-
"""
Created on Thu Nov  23 06:38:47 2017

@author: Thomas Schatz

Ad-hoc functions to generate within speaker
ABX stats and task files for perceptual tuning project.
"""

import ABXpy.task
import os.path as path


def generate_task(item, out, stats_only=False):
    t = ABXpy.task.Task(item,
                        on='phone',
                        across=[],
                        by=['speaker', 'prev-phone', 'next-phone'],
                        verbose=1)
    if stats_only:
        t.compute_nb_levels()
        t.print_stats(out)
    else:
        t.generate_triplets(output=out)  


if __name__ == '__main__':
  
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('item_file')
  parser.add_argument('out_file', help='task or stat file')
  parser.add_argument('--stats', action='store_true',
                      help='get task stats instead of actual task file')
  args = parser.parse_args()
  assert path.exists(args.item_file)
  assert not(path.exists(args.out_file))
  generate_task(args.item_file, args.out_file, stats_only=args.stats)
