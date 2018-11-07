import matplotlib
matplotlib.use('Agg')
import os

import matplotlib.pyplot as plt
import numpy as np


def draw(results, outFile):
    os.makedirs(os.path.dirname(outFile), exist_ok=True)

    matplotlib.rcParams['ps.useafm'] = True
    matplotlib.rcParams['pdf.use14corefonts'] = True
    matplotlib.rcParams['text.usetex'] = True

    x = results['breakpoints']

    handlers = []
    legends = []
    for key in results:
        if key != 'breakpoints':
            handler, = plt.plot(x, results[key])
            handlers.append(handler)
            legends.append(key)

    plt.legend(handlers, legends)
    plt.ylabel('regret', fontweight='bold', fontsize=15)
    plt.xlabel('horizon', fontweight='bold', fontsize=15)
    print('Output figure to %s' % outFile)
    plt.savefig(outFile, format="pdf")
    plt.close()
