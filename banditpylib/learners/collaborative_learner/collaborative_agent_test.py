"""
import numpy as np
import pandas as pd
import tempfile
import json
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")

from banditpylib.arms import GaussianArm
from banditpylib.bandits import OrdinaryBandit
from banditpylib.protocols import CollaborativeLearningProtocol, trial_data_messages_to_dict
from banditpylib.learners.collaborative_learner import CollaborativeMaster

confidence = 0.95
means = [0.3, 0.5, 0.7]
rounds, horizon, num_agents = 10, 100000, 5

arms = [GaussianArm(mu=mean, std=1) for mean in means]
bandit = OrdinaryBandit(arms=arms)
master = CollaborativeMaster(arm_num=len(arms),
    num_rounds=rounds, time_horizon=horizon, num_agents=num_agents)

trials = 20
game = CollaborativeLearningProtocol(bandit=bandit, master=master,
    rounds=rounds, horizon=horizon)
game.play(trials=trials, output_filename="trial_outputs.txt")

data_df = trial_data_messages_to_dict("trial_outputs.txt")
data_df['confidence'] = confidence

fig = plt.figure()
ax = plt.subplot(111)
sns.barplot(x='confidence', y='total_actions', hue='learner', data=data_df)
plt.ylabel('pulls')
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()

fig = plt.figure()
ax = plt.subplot(111)
sns.barplot(x='confidence', y='regret', hue='learner', data=data_df)
plt.ylabel('regret')
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()

"""