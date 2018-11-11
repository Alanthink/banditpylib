# -*- coding: utf-8 -*-
"""
Figure generating methods
"""

import os

import matplotlib.pyplot as plt


def draw(results, out_file):
    """draw method"""
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    breakpoints = results['breakpoints']

    handlers = []
    legends = []
    for key in results:
        if key != 'breakpoints':
            handler, = plt.plot(breakpoints, results[key])
            handlers.append(handler)
            legends.append(key)

    plt.legend(handlers, legends)
    plt.ylabel('regret', fontweight='bold', fontsize=15)
    plt.xlabel('horizon', fontweight='bold', fontsize=15)
    print('Output figure to %s' % out_file)
    plt.savefig(out_file, format="pdf")
    plt.close()
