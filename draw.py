"""
Figure generating methods
"""

import json
import os

from absl import logging

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns; sns.set()


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
          logging.fatal('Regret of T=%d is not recorded for learner %s' % (horizon, learner))
        col_horizons.append(horizon)
        col_regrets.append(float(regrets[str(horizon)]))
        col_learners.append(learner)

  total_runs = []
  for learner in trials_per_learner:
    if trials_per_learner[learner] not in total_runs:
      total_runs.append(trials_per_learner[learner])
  if len(total_runs) > 1:
    logging.warn('Algorithms are not experimented with the same trials!')
  logging.info('%d independent runs totally' % total_runs[0])

  results = pd.DataFrame({'learners':col_learners, 'horizons': col_horizons, 'regrets': col_regrets})

  return results


def draw_figure(data_file, out_file):
  os.makedirs(os.path.dirname(out_file), exist_ok=True)

  results = load_regrets(data_file)

  sns.lineplot(x='horizons', y='regrets', hue='learners', data=results, ci='sd')

  plt.ylabel('regret', fontweight='bold', fontsize=15)
  plt.xlabel('horizon', fontweight='bold', fontsize=15)
  logging.info('Output figure to %s' % out_file)
  plt.savefig(out_file, format="pdf")
  plt.close()
