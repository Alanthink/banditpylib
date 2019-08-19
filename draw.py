"""
Figure generating methods
"""

import os

import matplotlib.pyplot as plt
from absl import logging


def draw(results, out_file):
  """draw method"""
  os.makedirs(os.path.dirname(out_file), exist_ok=True)

  breakpoints = results['breakpoints']

  with plt.xkcd():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    handlers = []
    legends = []
    for key in results:
      if key != 'breakpoints':
        handler, = ax.plot(breakpoints, results[key])
        handlers.append(handler)
        legends.append(key)

    ax.legend(handlers, legends)

    ax.set_ylabel('regret', fontweight='bold', fontsize=15)
    ax.set_xlabel('horizon', fontweight='bold', fontsize=15)
    logging.info('Output figure to %s' % out_file)
    plt.savefig(out_file, format="pdf")
    plt.close()
