import gym
import random
import numpy as np
import collections
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


LR = 1e-3
GAMMA   = 0.99
EPSILON = 1.0
EPS_MIN = 0.05
EPS_DEC = 5e-4
ReplayBuffer = collections.deque(maxlen=100000)

from oroloEnv import MidEnv
from TicTacToe import TicTacToe
import random
#DEBUG = False
envs = MidEnv.make([TicTacToe() for _ in range(4)])
seed = 7

dummy_env = envs.compute()
obs_space = dummy_env.single_obs_space.shape[0]
act_space = dummy_env.single_act_space.n
device = 'cpu'


class Model(nn.Module):
  def __init__(self, obs_space, act_space, lr=LR):
    super(Model, self).__init__();
    self.fc1 = nn.Linear(obs_space, 24)
    self.fc2 = nn.Linear(24, 24)
    self.fc3 = nn.Linear(24, act_space)
    self.optimizer = optim.Adam(self.parameters(), lr=lr)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x)) 
    return self.fc3(x)


  def get_action(self, obs):
    if np.random.random() > EPSILON:
      obs = torch.tensor(obs, dtype=torch.float).to(device)
      action = self.forward(obs).argmax(1).numpy()
    else: action = np.random.randint(low=0,high=9, size=4)
    return action

def train(batch_size=32):
  loss = nn.MSELoss()
  if len(ReplayBuffer) >= batch_size:
    #print("Training")
    batch = random.sample(ReplayBuffer, batch_size)

    states, actions, rewards, nstates, dones = [], [], [], [], []
    for x in batch:
      states.append(x[0]); actions.append(x[1]); rewards.append(x[2])
      nstates.append(x[3]); dones.append(x[4])
    states = torch.tensor(np.concatenate(states),dtype=torch.float).to(device)
    actions = torch.tensor(np.concatenate(actions),dtype=torch.int64).to(device)
    nstates = torch.tensor(np.concatenate(nstates),dtype=torch.float).to(device)
    rewards = torch.tensor(np.concatenate(rewards),dtype=torch.float).to(device)
    dones = torch.tensor(np.concatenate(dones),dtype=bool).to(device)
    actions = actions[:,None]
    rewards = rewards[:,None]

    #print(states.shape, actions.shape)
    q_pred = policy(states).gather(1, actions)
    q_targ = target(nstates).max(1)[0].unsqueeze(1)
    #@print(q_targ.shape, rewards.shape)
    q_targ[dones] = 0.0  # set all terminal states' value to zero
    #print(q_targ.shape, rewards.shape)
    q_targ = rewards + GAMMA * q_targ
    #print(q_targ.shape, q_pred.shape)

    #
    loss = F.smooth_l1_loss(q_pred, q_targ).to(device)
    policy.optimizer.zero_grad()
    loss.backward()
    policy.optimizer.step()
    return loss.item()

  return 0

policy = Model( obs_space, act_space, LR).to(device)
target = Model( obs_space, act_space, LR).to(device) 
target.load_state_dict( policy.state_dict() ) 

scores, losses, avg_scores = [], [], []
for epi in range(500):
  obs = envs.reset(seed=random.randint(0,999)).compute()
  score = 0
  while True:
    actions = policy.get_action(obs)
    n_obs , rews, dones = envs.step(actions).compute()

    ReplayBuffer.append((obs,actions,rews,n_obs,dones))
    loss = train()
    losses.append(loss)
    score+=rews.mean()
    obs = n_obs 

    if True in dones:
      scores.append(score)
      avg_score = np.mean(scores[-100:]) # moving average of last 100 episodes
      avg_scores.append(avg_score)
      print(f"Episode {epi}, Return: {scores[-1]}, Avg return: {avg_score}")
      break

  # update target
  target.load_state_dict( policy.state_dict() ) 

  # update EPSILON
  EPSILON = max(EPS_MIN, EPSILON*EPS_DEC)


# *************************** Metrics ***************************  
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1,2)
fig.suptitle('DQN performance')
ax[0].plot(np.arange(len(losses)), losses)
ax[0].set( xlabel='Iteration', ylabel='losses',title='Losses')
ax[1].plot(np.arange(len(scores)), scores)
ax[1].plot(np.arange(len(avg_scores)), avg_scores)
ax[1].set( xlabel='Iteration', ylabel='Episodic return',title='Episodic return')
fig.savefig("test.png")
plt.show()  
