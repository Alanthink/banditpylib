"""
Figure generating methods
"""

import json
import os

import matplotlib.pyplot as plt
from absl import logging


def compute_avg_regret(file_name):
  """Compute average regret for every policy using given data file"""
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

  # compute aggregate regrets for different learners
  trials_per_learner = dict()
  agg_regrets = dict()
  with open(file_name, 'r') as f:
    for line in f:
      one_trial = json.loads(line)
      (learner, regrets) = list(one_trial.items())[0]
      if learner not in agg_regrets:
        # initialization
        trials_per_learner[learner] = 0
        agg_regrets[learner] = dict()
      trials_per_learner[learner] += 1
      for horizon in horizons:
        if str(horizon) not in regrets:
          logging.fatal('Regret of T=%d is not recorded for learner %s' % (horizon, learner))

        if int(horizon) not in agg_regrets[learner]:
          agg_regrets[learner][int(horizon)] = 0
        agg_regrets[learner][int(horizon)] += int(regrets[str(horizon)])

  # compute average regrets for different learners
  results = dict()
  for (learner, regrets) in agg_regrets.items():
    avg_regret = []
    for horizon in horizons:
      avg_regret.append(regrets[horizon]/trials_per_learner[learner])
    results[learner] = avg_regret

  total_runs = []
  for learner in trials_per_learner:
    if trials_per_learner[learner] not in total_runs:
      total_runs.append(trials_per_learner[learner])
  if len(total_runs) > 1:
    logging.warn('Algorithms are not experimented with the same trials!')

  logging.info('%d independent runs totally' % total_runs[0])

  return horizons, results


def draw_figure(data_file, out_file):
  os.makedirs(os.path.dirname(out_file), exist_ok=True)

  horizons, results = compute_avg_regret(data_file)

  with plt.xkcd():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    handlers = []
    legends = []
    for learner in results:
      handler, = ax.plot(horizons, results[learner])
      handlers.append(handler)
      legends.append(learner)

    ax.legend(handlers, legends)

    ax.set_ylabel('regret', fontweight='bold', fontsize=15)
    ax.set_xlabel('horizon', fontweight='bold', fontsize=15)
    logging.info('Output figure to %s' % out_file)
    plt.savefig(out_file, format="pdf")
    plt.close()
