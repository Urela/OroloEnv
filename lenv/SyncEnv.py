from __future__ import annotations
import multiprocessing as mp

from copy import deepcopy
import numpy as np
from enum import Enum
from typing import Union, Tuple, Optional, NamedTuple
from typing import Sequence, List, Any

class observation_space: 
  dtype = 'i' # interger type # https://www.bookstack.cn/read/python3.9-library-zh/499befbdb032db77.md
  shape = (11,)

class action_space: 
  dtype = 'i' # interger type # https://www.bookstack.cn/read/python3.9-library-zh/499befbdb032db77.md
  shape = (1,)
  n = 9

Space = Union[observation_space, action_space]

#----------------------------------------------------------------------------------------------------

def create_EmptyArray(space: Space, n: int = 1, fn=np.zeros):
  shape = space.shape if (n is None) else (n,) + space.shape
  #return fn(shape, dtype=space.dtype)
  return fn(shape, dtype='i')

# we only mutate environment
class SyncEnv:
  def __init__(
    self,
    env_fxns:  Sequence[Env], # Functions that create the environments.
    obs_space: Optional[Space] = None,    # Observation space of a single environment
    act_space: Optional[Space] = None,    # Action space of a single environment
    copy: bool = True,

  ):

    self.env_fxns = env_fxns
    self.envs = env_fxns
    self.copy = copy

    self.num_envs=len(self.env_fxns)
    if (obs_space is None) or (act_space is None):
      obs_space = obs_space or self.envs[0].observation_space
      act_space = act_space or self.envs[0].action_space 

    self.single_obs_space = obs_space
    self.single_act_space = act_space
    self.obs = create_EmptyArray(self.single_obs_space , n=self.num_envs, fn=np.zeros )

    self._rewards = np.zeros((self.num_envs), dtype=np.float64)
    self._dones = np.zeros((self.num_envs), dtype=np.bool_)
    self._actions = None


  @classmethod
  def make(cls, env): return cls(env)
  def compute(self): return self.envs
  def step(self, actions: Optional[Union[int, List[int]]]):
    ''' if this was an async envirnonment we would collect processes here'''
    self._actions=actions
    observations = []
    for i, (env, action) in enumerate(zip(self.envs, self._actions)):
      (   observation,
          self._rewards[i],
          self._dones[i],
      ) = env.step(action)


      if self._dones[i]: observation = env.reset()
      observations.append(observation)
    self.obs = np.stack(observations, axis=0, out=self.obs)
    return (deepcopy(self.obs) if self.copy else self.obs,
            np.copy(self._rewards),
            np.copy(self._dones),
           )

  def reset(
    self,
    seed: Optional[Union[int, List[int]]] = None,
    args: Optional[dict] = None,
  ):
    ''' if this was an async envirnonment we would collect processes here'''
    if seed is None: seed = [None for _ in range(self.num_envs)]
    if isinstance(seed, int): seed = [seed + i for i in range(self.num_envs)]
    assert len(seed) == self.num_envs

    self._dones[:] = False
    observations = []
    for i, (env, single_seed) in enumerate(zip(self.envs, seed)):
      kwargs = {}
      if single_seed is not None:
        kwargs["seed"] = single_seed
      if args is not None:
        kwargs["args"] = args
      observations.append(env.reset())
    self.obs = np.stack(observations, axis=0, out=self.obs)
    return (deepcopy(self.obs) if self.copy else self.obs)

  def render(self,args): 
    for e in self.envs: print(e.render() )
    return self
