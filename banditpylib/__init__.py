"""
A lightweight python library for bandit algorithms
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

__all__ = [
    'plot',
    'run'
]


class Constants():
  """Class for storing modular constants"""
  config = None
  data = None
  new_policies = None
  ana_df = None


def draw_regretmin(data):
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


def draw_fixbudgetbai(data):
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


def draw_fixconfbai(data):
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
  if not Constants.data:
    raise Exception('Data is empty!')
  if not FLAGS.goal:
    raise Exception('Please specify the goal of learners!')
  elif FLAGS.goal == 'regretmin':
    Constants.ana_df = draw_regretmin(Constants.data)
  elif FLAGS.goal == 'bestarmid.fixbudget':
    Constants.ana_df = draw_fixbudgetbai(Constants.data)
  elif FLAGS.goal == 'bestarmid.fixconf':
    Constants.ana_df = draw_fixconfbai(Constants.data)
  else:
    raise Exception('No specified goal!')


def plot(data, goal='', novar=False, save_fig=''):
  """Method for plotting the figure

  Args:
    data (dict): generated data
    goal (str): goal of the learners
    novar (bool): set it to ``True`` if you also want the std plotted
    save_fig (str): file where the generated figure is stored
  """
  Constants.data = data
  if goal:
    FLAGS.goal = goal
  FLAGS.novar = novar
  FLAGS.fig = save_fig
  try:
    app.run(_plot)
  except SystemExit:
    return Constants.ana_df


def parse_setup(Bandit, bandit_pars, Learner, learner_pars):
  Protocol = getattr(import_module(PROTOCOL_PKG), Learner.protocol)
  protocol = Protocol(learner_pars)
  if 'SinglePlayer' in Learner.protocol:
    # single player
    bandit = Bandit(bandit_pars)
    learner = Learner(learner_pars)
  else:
    # multiple players
    if 'num_players' not in learner_pars:
      raise Exception(
          '%s: please specify the number of players!' %
          Learner.__class__.__name__)
    num_players = learner_pars['num_players']
    bandit = [Bandit(bandit_pars) for _ in range(num_players)]
    learner = [Learner(learner_pars) for _ in range(num_players)]
  return (bandit, protocol, learner)


def parse(config, new_policies):
  setups = []

  bandit_classname = config['environment']['bandit']
  Bandit = getattr(import_module(BANDIT_PKG), bandit_classname)
  bandit_pars = config['environment']['params']
  learner_goal = config['learners']['goal']
  FLAGS.goal = learner_goal
  # initialize learners and their corresponding protocols
  if not isinstance(config['learners']['policies'], list):
    raise Exception('Please specify policies in a list!')
  for learner_config in config['learners']['policies']:
    learner_classname = learner_config['policy']
    learner_pars = learner_config['params']
    learner_package = '%s.%s.%s' % \
        (LEARNER_PKG, learner_goal, learner_pars['type'])
    Learner = getattr(import_module(learner_package), learner_classname)
    setups.append(parse_setup(Bandit, bandit_pars, Learner, learner_pars))

  if new_policies:
    for learner_config in new_policies:
      if not isinstance(learner_config, tuple):
        raise Exception('New policy should be given in a two-tuple!')
      if len(learner_config) != 2:
        raise Exception('Two-tuple for the new policy please!')
      setups.append(parse_setup(
          Bandit, bandit_pars, learner_config[0], learner_config[1]))

  goals = [setup[2][0].goal if isinstance(setup[2], list) else setup[2].goal
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

  setups, running_pars = parse(Constants.config, Constants.new_policies)
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
  Constants.data = data


def run(config, new_policies=None, debug=False):
  """Method for playing the game

  Args:
    config (dict): config should comply with the following form.

      .. code-block:: yaml

        {
          "environment": {
            # Class name of the environment. Check `bandits` package for
            # supported environments.
            "bandit": "",
            "params": {
              # "name: value" pairs. Parameters for the environment. Check
              # environment *__init__* method for parameters.
            }
          },
          "learners": {
            # Goal of the learners. Now we only support "regretmin",
            # "bestarmid.fixbudget", and "bestarmid.fixbudget".
            "goal": "",
            # A list of policy configurations
            "policies": [
              {
                # Class name for the policy
                "policy": "",
                "params": {
                  # Type of the policy which should be the closest package it
                  # belongs. For example, the type of banditpylib.learners.
                  # regretmin.ordinarylearner.EpsGreedy
                  # should be "ordinarylearner".
                  "type": "",
                  # "name: value" pairs. Other parameters for the policy. Check
                  # learner class *__init__* method for parameters.
                }
              },
            ]
          },
          "running": {
            # "name: value" pairs. Parameters for running the setup. These
            # parameters will be passed to prorotol for running the setup. Check
            # protocol *__init__* method for parameters.
          }
        }
    new_policies ([(class, dict),]): for each two-tuple (A, B),
      A is the class defined by yourself and B denotes the
      parameters of your policy.
    debug (bool): set it to ``True`` to enter the debug mode

  .. warning::
    The running time of debug mode may be increased heavily compared to the
    normal mode. This mode is used to debug errors generated by subprocesses.

  Return:
    dict: generated data
  """
  Constants.config = config
  Constants.new_policies = new_policies
  FLAGS.debug = debug
  try:
    app.run(_run)
  except SystemExit:
    return Constants.data
