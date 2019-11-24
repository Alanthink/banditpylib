"""
Figure generating methods
"""

import json
import logging as log
import os

from absl import logging
from absl import flags

from matplotlib.collections import PolyCollection
from tabulate import tabulate

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns; sns.set()

FLAGS = flags.FLAGS

log.getLogger('matplotlib').setLevel(log.ERROR)

FIG_NAME = 'figure'
FIG_FMT = 'pdf'
ANA_NAME = 'analysis.txt'


def draw_regretmin(data_file):
  # files will be generated
  fig_file = os.path.join(FLAGS.dir, FIG_NAME+'.'+FIG_FMT)
  ana_file = os.path.join(FLAGS.dir, ANA_NAME)

  col_learners = []
  col_horizons = []
  col_regrets = []
  with open(data_file, 'r') as f:
    for line in f:
      one_trial = json.loads(line)
      (learner, regrets) = list(one_trial.items())[0]
      for horizon in regrets:
        col_learners.append(learner)
        col_horizons.append(int(horizon))
        col_regrets.append(regrets[horizon])

  df = pd.DataFrame({'learner':col_learners, 'horizon':col_horizons,
      'regret': col_regrets})

  with open(ana_file, 'w') as f:
    f.write(tabulate(df.groupby(['learner','horizon']).mean(),
        headers='keys', tablefmt='psql'))

  if FLAGS.novar:
    ci_val = None
  else:
    ci_val = 'sd'

  ax = sns.lineplot(
      x='horizon', y='regret', hue='learner', data=df, ci=ci_val)

  # hide edges of filled area
  for child in ax.findobj(PolyCollection):
    child.set_linewidth(0.0)

  plt.ylabel('regret', fontweight='bold', fontsize=15)
  plt.xlabel('horizon', fontweight='bold', fontsize=15)
  logging.info('output figure to %s' % fig_file)
  plt.savefig(fig_file, format=FIG_FMT)
  plt.close()


def draw_fixbudgetbai(data_file):
  # files will be generated
  fig_file = os.path.join(FLAGS.dir, FIG_NAME+'.'+FIG_FMT)
  ana_file = os.path.join(FLAGS.dir, ANA_NAME)

  col_learners = []
  col_budgets = []
  col_regrets = []

  with open(data_file, 'r') as f:
    lines = f.readlines()
    if not lines:
      logging.fatal('File is empty!')

    for line in lines:
      one_trial = json.loads(line)
      (learner, (budget, regret)) = list(one_trial.items())[0]
      col_learners.append(learner)
      col_budgets.append(int(budget))
      col_regrets.append(regret)

  df = pd.DataFrame({'learner':col_learners, 'budget': col_budgets,
      'fail_prob': col_regrets})

  with open(ana_file, 'w') as f:
    f.write(tabulate(df.groupby(['learner','budget']).mean(),
        headers='keys', tablefmt='psql'))

  sns.catplot(x='budget', y='fail_prob', hue='learner', kind='point', data=df)

  plt.ylabel('fail probability', fontweight='bold', fontsize=15)
  plt.xlabel('budget', fontweight='bold', fontsize=15)
  logging.info('output figure to %s' % fig_file)
  plt.savefig(fig_file, format=FIG_FMT)
  plt.close()


def draw_fixconfbai(data_file):
  # files will be generated
  fig_file = os.path.join(FLAGS.dir, FIG_NAME+'.'+FIG_FMT)
  ana_file = os.path.join(FLAGS.dir, ANA_NAME)

  col_learners = []
  col_fail_prob = []
  col_samples = []
  col_regrets = []

  with open(data_file, 'r') as f:
    lines = f.readlines()
    if not lines:
      logging.fatal('File is empty!')

    for line in lines:
      one_trial = json.loads(line)
      (learner, (fail_prob, samples, regret)) = list(one_trial.items())[0]
      col_learners.append(learner)
      col_fail_prob.append(fail_prob)
      col_samples.append(samples)
      col_regrets.append(regret)

  samples_df = pd.DataFrame({'learner':col_learners,
      'fix_fail_prob':col_fail_prob, 'samples':col_samples,
      'fail_prob':col_regrets})

  with open(ana_file, 'w') as f:
    f.write(tabulate(samples_df.groupby(['learner','fix_fail_prob']).mean(),
        headers='keys', tablefmt='psql'))

  if FLAGS.novar:
    ci_val = None
  else:
    ci_val = 'sd'

  sns.catplot(x='fix_fail_prob', y='samples',
      hue='learner', kind='bar', ci=ci_val, data=samples_df)

  plt.ylabel('samples', fontweight='bold', fontsize=15)
  plt.xlabel('fail probability', fontweight='bold', fontsize=15)
  logging.info('output figure to %s' % fig_file)
  plt.savefig(fig_file, format=FIG_FMT)
  plt.close()


def draw_figure(goal):
  data_file = os.path.join(FLAGS.dir, FLAGS.data_filename)

  if goal == 'regretmin':
    draw_regretmin(data_file)
  elif goal == 'bestarmid.fixbudget':
    draw_fixbudgetbai(data_file)
  elif goal == 'bestarmid.fixconf':
    draw_fixconfbai(data_file)
  else:
    logging.fatal('No analysis output for this goal!')
