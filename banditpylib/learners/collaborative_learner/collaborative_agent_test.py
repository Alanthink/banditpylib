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
from banditpylib.protocols import CollaborativeLearnerProtocol, trial_data_messages_to_dict
from banditpylib.learners.ordinary_fcbai_learner import CollaborativeAgent, LilUCBHeuristic

confidence = 0.95
means = [0.3, 0.5, 0.7]
arms = [GaussianArm(mu=mean, var=1) for mean in means]
bandit = OrdinaryBandit(arms=arms)
agent = CollaborativeAgent(arm_num=len(
    arms), name='Collaborative Agent', num_rounds=10, time_horizon=1000)
agent.assign_arms([2])
agent.reset()
trials = 20
learners = [LilUCBHeuristic(arm_num=len(
    arms), confidence=confidence, name='Heuristic lilUCB'), agent]

game = CollaborativeLearnerProtocol(bandit=bandit, learners=learners)
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