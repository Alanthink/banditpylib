"""
banditpylib: a lightweight python library for bandit algorithms
"""
import json
import os

from importlib import import_module
import tempfile

from absl import app
from absl import logging
from absl import flags

from matplotlib.collections import PolyCollection
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set()

FLAGS = flags.FLAGS

flags.DEFINE_boolean('debug', False, 'output runtime debug info')
flags.DEFINE_string('out', 'data.out', 'file for generated data')
flags.DEFINE_boolean('novar', False, 'do not show std in the output figure')
flags.DEFINE_string('goal', '', 'goal of the learners')
flags.DEFINE_string('fig', '', 'file for saving generated figure')
flags.DEFINE_string('f', '', 'kernel')

BANDIT_PKG = 'banditpylib.bandits'
LEARNER_PKG = 'banditpylib.learners'
PROTOCOL_PKG = 'banditpylib.protocols'


class _constants():
  """class for storing modular constants"""
  config = None
  data = None
  new_policies = None
  ana_df = None


def _draw_regretmin(data):
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

  if FLAGS.fig:
    logging.info('output figure to %s' % FLAGS.fig)
    plt.savefig(FLAGS.fig)
    plt.close()

  return analysis


def _draw_fixbudgetbai(data):
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

  if FLAGS.fig:
    logging.info('output figure to %s' % FLAGS.fig)
    plt.savefig(FLAGS.fig)
    plt.close()

  return analysis


def _draw_fixconfbai(data):
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

  if FLAGS.fig:
    logging.info('output figure to %s' % FLAGS.fig)
    plt.savefig(FLAGS.fig)
    plt.close()

  return analysis


def _plot(argv):
  del argv
  if not _constants.data:
    raise Exception('Data is empty!')
  if not FLAGS.goal:
    raise Exception('Please specify the goal of learners!')
  elif FLAGS.goal == 'regretmin':
    _constants.ana_df = _draw_regretmin(_constants.data)
  elif FLAGS.goal == 'bestarmid.fixbudget':
    _constants.ana_df = _draw_fixbudgetbai(_constants.data)
  elif FLAGS.goal == 'bestarmid.fixconf':
    _constants.ana_df = _draw_fixconfbai(_constants.data)
  else:
    raise Exception('No specified goal!')


def plot(data, goal='', novar=False, save_fig=''):
  _constants.data = data
  if goal:
    FLAGS.goal = goal
  FLAGS.novar = novar
  FLAGS.fig = save_fig
  try:
    app.run(_plot)
  except SystemExit:
    return _constants.ana_df


def _parse(config, new_policies):
  setups = []
  bandit_name = config['environment']['bandit']
  Bandit = getattr(import_module(BANDIT_PKG), bandit_name)
  bandit_pars = config['environment']['params']
  learner_goal = config['learners']['goal']
  FLAGS.goal = learner_goal
  # initialize learners and their corresponding protocols
  for learner_config in config['learners']['policies']:
    learner_package = '%s.%s.%s' % \
        (LEARNER_PKG, learner_goal, learner_config['params']['type'])
    learner_name = learner_config['policy']
    Learner = getattr(import_module(learner_package), learner_name)

    Protocol = getattr(import_module(PROTOCOL_PKG), Learner.protocol)
    protocol = Protocol(learner_config['params'])

    if 'SinglePlayer' in Learner.protocol:
      # single player
      bandit = Bandit(bandit_pars)
      learner = Learner(learner_config['params'])
    else:
      # multiple players
      if 'num_players' not in learner_config['params']:
        raise Exception(
            '%s: please specify the number of players!' % learner_name)
      num_players = learner_config['params']['num_players']
      bandit = [Bandit(bandit_pars) for _ in range(num_players)]
      learner = [Learner(learner_config['params']) for _ in range(num_players)]
    setups.append((bandit, protocol, learner))

  if new_policies:
    for learner in new_policies:
      if isinstance(learner, list):
        # multi-players
        num_players = len(learner)
        bandit = [Bandit(bandit_pars) for _ in range(num_players)]
        Protocol = getattr(import_module(PROTOCOL_PKG), learner[0].protocol)
        protocol = Protocol({'num_players': num_players})
      else:
        # single player
        bandit = Bandit(bandit_pars)
        protocol = getattr(import_module(PROTOCOL_PKG), learner.protocol)()
      setups.append((bandit, protocol, learner))

  goals = [setup[2][0].goal if isinstance(learner, list) else setup[2].goal
           for setup in setups]
  if any(goal != goals[0] for goal in goals):
    raise Exception('Learners must have the same goal!')
  logging.info('run with goal %s' % goals[0])
  return setups, config['running']


def _run(argv):
  del argv
  if FLAGS.debug:
    # DEBUG, INFO, WARN, ERROR, FATAL
    logging.set_verbosity(logging.DEBUG)
  else:
    logging.set_verbosity(logging.INFO)

  setups, running_pars = _parse(_constants.config, _constants.new_policies)
  try:
    temp_dir = tempfile.mkdtemp()
    data_file = os.path.join(temp_dir, FLAGS.out)
    for (bandit, protocol, learner) in setups:
      protocol.play(bandit, learner, running_pars, data_file)
    data = []
    with open(data_file, 'r') as file:
      for line in file:
        data.append(json.loads(line))
    os.remove(data_file)
  finally:
    os.rmdir(temp_dir)
  _constants.data = data


def run(config, new_policies=None, debug=False):
  _constants.config = config
  _constants.new_policies = new_policies
  FLAGS.debug = debug
  try:
    app.run(_run)
  except SystemExit:
    return _constants.data
