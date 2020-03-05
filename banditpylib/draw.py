"""
Figure generating methods
"""
import logging as log

from absl import logging
from absl import flags

from matplotlib.collections import PolyCollection

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set()

FLAGS = flags.FLAGS

log.getLogger('matplotlib').setLevel(log.ERROR)

FIG_NAME = 'figure'
FIG_FMT = 'pdf'
ANA_NAME = 'analysis.txt'

__all__ = ['plot']


def draw_regretmin(data, save_fig):
  col_learners = []
  col_horizons = []
  col_regrets = []
  for one_trial in data:
    (learner, regrets) = list(one_trial.items())[0]
    for horizon in regrets:
      col_learners.append(learner)
      col_horizons.append(int(horizon))
      col_regrets.append(regrets[horizon])

  df = pd.DataFrame({'learner':col_learners, 'horizon':col_horizons,
                     'regret': col_regrets})
  analysis = df.groupby(['learner', 'horizon']).mean()

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

  if save_fig:
    logging.info('output figure to %s' % save_fig)
    plt.savefig(save_fig)
    plt.close()

  return analysis


def draw_fixbudgetbai(data, save_fig):
  col_learners = []
  col_budgets = []
  col_regrets = []

  for one_trial in data:
    (learner, (budget, regret)) = list(one_trial.items())[0]
    col_learners.append(learner)
    col_budgets.append(int(budget))
    col_regrets.append(regret)

  df = pd.DataFrame({'learner':col_learners,
                     'budget': col_budgets,
                     'fail_prob': col_regrets})
  analysis = df.groupby(['learner', 'budget']).mean()

  if FLAGS.novar:
    ci_val = None
  else:
    ci_val = 'sd'
  sns.catplot(x='budget', y='fail_prob', hue='learner',
              kind='point', ci=ci_val, data=df)
  plt.ylabel('fail probability', fontweight='bold', fontsize=15)
  plt.xlabel('budget', fontweight='bold', fontsize=15)

  if save_fig:
    logging.info('output figure to %s' % save_fig)
    plt.savefig(save_fig)
    plt.close()

  return analysis


def draw_fixconfbai(data, save_fig):
  col_learners = []
  col_fail_prob = []
  col_samples = []
  col_regrets = []

  for one_trial in data:
    (learner, (fail_prob, samples, regret)) = list(one_trial.items())[0]
    col_learners.append(learner)
    col_fail_prob.append(fail_prob)
    col_samples.append(samples)
    col_regrets.append(regret)

  samples_df = pd.DataFrame({'learner':col_learners,
                             'fix_fail_prob':col_fail_prob,
                             'samples':col_samples,
                             'fail_prob':col_regrets})
  analysis = samples_df.groupby(['learner', 'fix_fail_prob']).mean()

  if FLAGS.novar:
    ci_val = None
  else:
    ci_val = 'sd'
  sns.catplot(x='fix_fail_prob', y='samples',
              hue='learner', kind='bar', ci=ci_val, data=samples_df)
  plt.ylabel('samples', fontweight='bold', fontsize=15)
  plt.xlabel('fail probability', fontweight='bold', fontsize=15)

  if save_fig:
    logging.info('output figure to %s' % save_fig)
    plt.savefig(save_fig)
    plt.close()

  return analysis


def plot(data, goal=None, novar=False, save_fig=None):
  FLAGS.novar = novar

  if not data:
    logging.fatal('Data is empty!')

  if goal == 'regretmin':
    return draw_regretmin(data, save_fig)
  elif goal == 'bestarmid.fixbudget':
    return draw_fixbudgetbai(data, save_fig)
  elif goal == 'bestarmid.fixconf':
    return draw_fixconfbai(data, save_fig)
  else:
    logging.fatal('Please specify the goal of learners!')
