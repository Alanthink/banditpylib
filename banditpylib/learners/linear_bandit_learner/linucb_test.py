import matplotlib.pyplot as plt
import numpy as np
import tempfile
import seaborn as sns
sns.set(style="darkgrid")

from banditpylib.bandits import LinearBandit
from banditpylib.protocols import SinglePlayerProtocol, trial_data_messages_to_dict
from banditpylib.learners.linear_bandit_learner import LinUCB

horizon = 2000
num_arms = 100
len_feature = 5
features = []

for _ in range(num_arms):
    features.append(np.random.normal(0, size=len_feature))
    features[-1] = features[-1] / np.linalg.norm(features[-1])

theta = np.random.normal(0, size=len_feature)

bandit = LinearBandit(features, theta)
learners = [LinUCB(num_arms, features, 1 / horizon, 1e-3)]

intermediate_regrets = list(range(0, horizon + 1, 50))
temp_file = tempfile.NamedTemporaryFile()

game = SinglePlayerProtocol(
    bandit, learners,
    intermediate_regrets=intermediate_regrets, horizon=horizon)

# game.play(trials=1, output_filename=temp_file.name)

data_df = trial_data_messages_to_dict(temp_file.name)
sns.lineplot(x='total_actions', y='regret', hue='learner', data=data_df)
plt.show()
