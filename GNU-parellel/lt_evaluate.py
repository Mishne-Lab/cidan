# -*- coding: utf-8 -*-
"""
Created on Tue May  7 17:00:20 2019
Modified from evaluate.py from neurofinder
Argumant1:ROIlist1 name string; Argument2:ROIlist2 name string
Save results as dictionary
@author: Lan Tang
"""

import click
from neurofinder import load, centers, shapes


@click.argument('files', nargs=2, metavar='<files: ground truth, estimate>',
                required=True)
@click.option('--threshold', default=5, help='threshold distance')
@click.command('evaluate', short_help='compare results of two algorithms',
               options_metavar='<options>')
def lt_evaluate(file1, file2, threshold=5):
    a = load(file1)
    b = load(file2)
    if a != 0 and b != 0:
        recall, precision = centers(a, b, threshold=threshold)
        inclusion, exclusion = shapes(a, b, threshold=threshold)

        if recall == 0 and precision == 0:
            combined = 0
        else:
            combined = 2 * (recall * precision) / (recall + precision)

        result = {'combined': round(combined, 4), 'inclusion': round(inclusion, 4),
                  'precision': round(precision, 4), 'recall': round(recall, 4),
                  'exclusion': round(exclusion, 4)}
        return (result)
    else:
        return {}
