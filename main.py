"""
A simple example.

To run, try `python3 main.py` under `banditpylib` root directory.
The result is output to `out/figure.pdf` by default.
"""

import os

from absl import app
from absl import logging
from absl import flags

from arm import BernoulliArm
from bandit import Bandit
from draw import draw_figure
from basiclearner import Uniform, UCB, MOSS, TS
from simulator import RegretMinimizationSimulator

FLAGS = flags.FLAGS

flags.DEFINE_string('dir', 'out', 'output directory')
flags.DEFINE_string('data_filename', 'data.out', 'output data filename')
flags.DEFINE_string('figure_filename', 'figure.pdf', 'output figure filename')
flags.DEFINE_enum('do', 'all', ['all', 'd', 'f', 'r'],
    'd:generate the data, f:generate the figure, r:remove the data, all:do everything')

# DEBUG, INFO, WARN, ERROR, FATAL
# For debugging purpose
logging.set_verbosity(logging.INFO)


def main(argv):
  del argv

  data_file = os.path.join(FLAGS.dir, FLAGS.data_filename)
  figure_file = os.path.join(FLAGS.dir, FLAGS.figure_filename)

  if FLAGS.do in ['d', 'all']:
    # data generation
    means = [0.3, 0.5, 0.7]
    arms = [BernoulliArm(mean) for mean in means]
    bandit = Bandit(arms)
    learners = [Uniform(), UCB(), MOSS(), TS()]
    simulator = RegretMinimizationSimulator(bandit, learners)

    horizon = 2000
    simulator.sim(horizon, data_file)
  if FLAGS.do in ['f', 'all']:
    # figure generation
    draw_figure(data_file, figure_file)
  if FLAGS.do in ['r', 'all']:
    # remove generated data
    os.remove(data_file)


if __name__ == '__main__':
  app.run(main)
