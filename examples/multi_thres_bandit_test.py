#%%
# test in vscode by run cell
import numpy as np
import pandas as pd
import tempfile
import json
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")
import datetime
import os 

import warnings
warnings.simplefilter('ignore')
import logging
logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

from banditpylib.arms import CategoricalArm
from banditpylib.bandits import MultiThresholdingBandit
from banditpylib.protocols import SinglePlayerProtocol
from banditpylib.learners.multi_thres_bandit_learner import M_OPT_KG, M_LSA, M_Uniform




def play_game(ps, budget, trials=20):
  arms = [CategoricalArm(p) for p in ps]
  categ_num = len(ps[0])
  bandit = MultiThresholdingBandit(arms=arms)
  learners = [M_OPT_KG(arm_num=len(arms), budget=budget, categ_num=categ_num, opt=True),
              M_LSA(arm_num=len(arms), budget=budget, categ_num=categ_num),
              M_Uniform(arm_num=len(arms), budget=budget, categ_num=categ_num)]
  gap = 2
  # record intermediate regrets for each trial of a game
  intermediate_regrets = list(range(0, budget+1, gap))[1:]
  temp_file = tempfile.NamedTemporaryFile()
  
  # simulator
  game = SinglePlayerProtocol(bandit=bandit, learners=learners, intermediate_regrets=intermediate_regrets)
  # start playing the game
  # add `debug=True` for debugging purpose
  game.play(trials=trials, output_filename=temp_file.name)
  
  with open(temp_file.name, 'r') as f:
    data = []
    lines = f.readlines()
    for line in lines:
      data.append(json.loads(line))
    data_df = pd.DataFrame.from_dict(data)
  
  data_df = data_df.groupby(['learner', 'total_actions'])['regret'].mean().reset_index()
  data_df['regret'] = np.log(data_df['regret'])
  data_df = data_df.replace(float('nan'), 0)
  data_df.head()

  
  sns.lineplot(x='total_actions', y='regret', hue='learner', data=data_df)



if __name__ == '__main__':
  ps=[[0.1,0.1,0.8], [0.05,0.1,0.85],[0.1,0.3,0.6],[0.32,0.32,0.36],[0.2,0.3,0.5],[0.3,0.3,0.4]]
  play_game(ps, budget=300, trials=5)

  

# %%
