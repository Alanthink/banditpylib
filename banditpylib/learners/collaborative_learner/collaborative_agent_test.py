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
from banditpylib.learners.collaborative_learner import CollaborativeAgent

confidence = 0.95
means = [0.3, 0.5, 0.7]
arms = [GaussianArm(mu=mean, std=1) for mean in means]
bandit = OrdinaryBandit(arms=arms)
agent = CollaborativeAgent(arm_num=len(
    arms), name='Collaborative Agent', num_rounds=10, time_horizon=1000)
trials = 20

game = CollaborativeLearningProtocol(bandit=bandit, agent=agent,
    num_agents=10, rounds=10, horizon=1000)
game.play(trials=trials, output_filename="trial_outputs.txt")

print(agent.broadcast())

data_df = trial_data_messages_to_dict("trial_outputs.txt")
data_df['confidence'] = confidence

fig = plt.figure()
ax = plt.subplot(111)
sns.barplot(x='confidence', y='total_actions', hue='learner', data=data_df)
plt.ylabel('pulls')
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()
"""