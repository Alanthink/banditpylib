"""
Figure generating methods
"""

import json
import os

from absl import logging

from matplotlib.collections import PolyCollection

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns; sns.set()


def check_total_runs(total_runs):
  if len(total_runs) > 1:
    logging.warn('Algorithms are not experimented with the same trials!')
  logging.info('%d independent trials totally' % total_runs[0])


def load_regrets(file_name):
  # compute horizons where regrets are recorded
  horizons = []
  with open(file_name, 'r') as f:
    lines = f.readlines()
    if not lines:
      logging.fatal('File is empty!')

    first_output = json.loads(lines[0])
    anss = list(first_output.values())[0]
    for horizon in anss:
      if int(horizon) not in horizons:
        horizons.append(int(horizon))
  horizons = sorted(horizons)

  # record regrets of different horizons for different learners
  trials_per_learner = dict()
  col_horizons = []
  col_regrets = []
  col_learners = []
  with open(file_name, 'r') as f:
    for line in f:
      one_trial = json.loads(line)
      (learner, regrets) = list(one_trial.items())[0]
      if learner not in trials_per_learner:
        # initialization
        trials_per_learner[learner] = 0
      trials_per_learner[learner] += 1
      for horizon in horizons:
        if str(horizon) not in regrets:
          logging.fatal('Regret of T=%d is not recorded for learner %s' %
              (horizon, learner))
        col_horizons.append(horizon)
        col_regrets.append(float(regrets[str(horizon)]))
        col_learners.append(learner)

  total_runs = []
  for learner in trials_per_learner:
    if trials_per_learner[learner] not in total_runs:
      total_runs.append(trials_per_learner[learner])
  check_total_runs(total_runs)

  results = pd.DataFrame({'learner':col_learners, 'horizon': col_horizons,
      'regret': col_regrets})

  return results


def draw_regretmin(data_file, out_file, novar):
  results = load_regrets(data_file)

  if novar:
    ci_val = None
  else:
    ci_val = 'sd'

  ax = sns.lineplot(
      x='horizon', y='regret', hue='learner', data=results, ci=ci_val)

  # hide edges of filled area
  for child in ax.findobj(PolyCollection):
    child.set_linewidth(0.0)

  plt.ylabel('regret', fontweight='bold', fontsize=15)
  plt.xlabel('horizon', fontweight='bold', fontsize=15)
  logging.info('output figure to %s' % out_file)
  plt.savefig(out_file, format='pdf')
  plt.close()


def draw_fixbudgetbai(data_file, out_file):
  data = dict()
  with open(data_file, 'r') as f:
    lines = f.readlines()
    if not lines:
      logging.fatal('File is empty!')

    for line in lines:
      one_trial = json.loads(line)
      (learner, regret) = list(one_trial.items())[0]
      if learner not in data:
        data[learner] = dict()
        data[learner][regret[0]] = (regret[1],1)
      elif regret[0] not in data[learner]:
        data[learner][regret[0]] = (regret[1],1)
      else:
        data[learner][regret[0]] = (regret[1]+data[learner][regret[0]][0],
            1+data[learner][regret[0]][1])

  col_learners = []
  col_budgets = []
  col_regrets = []
  total_runs = []

  for learner in data:
    for budget in data[learner]:
      col_learners.append(learner)
      col_budgets.append(budget)
      col_regrets.append(data[learner][budget][0] / \
          data[learner][budget][1])
      if data[learner][budget][1] not in total_runs:
        total_runs.append(data[learner][budget][1])
  check_total_runs(total_runs)

  df = pd.DataFrame({'learner':col_learners, 'budget': col_budgets,
      'fail_prob': col_regrets})

  sns.catplot(x='budget', y='fail_prob', hue='learner', kind='point', data=df)

  plt.ylabel('fail probability', fontweight='bold', fontsize=15)
  plt.xlabel('budget', fontweight='bold', fontsize=15)
  logging.info('output figure to %s' % out_file)
  plt.savefig(out_file, format='pdf')
  plt.close()


def draw_figure(data_file, out_file, goal, novar=True):
  os.makedirs(os.path.dirname(out_file), exist_ok=True)
  if goal == 'regretmin':
    draw_regretmin(data_file, out_file, novar)
  elif goal == 'bestarmid.fixbudget':
    draw_fixbudgetbai(data_file, out_file)
  else:
    logging.fatal('No figure output for this goal!')
