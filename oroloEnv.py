from __future__ import annotations 
from enum import Enum
from typing  import Union, Tuple, Type, NamedTuple
import functools
import operator
import time

"""
TODO :
 - [o] Monitor performance
   - [x] FPS :
 - [o] Backends
   - [x] async and sync
   - [ ] jax : jit compiler
   - [ ] triden : GPU
 - [ ] more environements
"""
class BaseEnv:
  def __init__(self,env: Env): self.env = env
  @classmethod
  def make(cls, env): return cls(env)
  def compute(self): return self.env
  def step(self, *action, **kwargs): return (env.step(action))
  def reset(self, *args, **kwargs): return (env.reset())
  #def render(self, *args, **kwargs) return (env.render())
  #def close(self, *args, **kwargs) return (env.close())

EnvStates = Enum('EnvStates', ['MAKE','RESET','STEP','CLOSE','RENDER'])
LAZY = True
DEBUG = False

from lenv.AsyncEnv import AsyncEnv
from lenv.SyncEnv import SyncEnv
Devices = {'async':AsyncEnv, 'sync':SyncEnv, 'base':BaseEnv }
import sys
sys.setrecursionlimit(10000)

class EnvState(NamedTuple):
  op : Op
  src: List[Type[oroloEnv], Type[EnvOp]]
  arg: Any=None

# ***************** Generates tree nodes *****************
# These functions are unware of the envirnoment they should be runninng

def wait_make(benv:BaseEnv):
  return Devices[benv.device].make(benv.op.arg), [], EnvStates.MAKE

def wait_reset(benv:BaseEnv):
  src = [ x.evaluate() for x in benv.op.src if x is not None ]
  return src[0].reset(*benv.op.arg), src, EnvStates.RESET

def wait_step(benv:BaseEnv):
  src = [ x.evaluate() for x in benv.op.src if x is not None ]
  return src[0].step(benv.op.arg), src, EnvStates.STEP

def wait_render(benv:BaseEnv):
  src = [ x.evaluate() for x in benv.op.src if x is not None ]
  return src[0].step(benv.op.arg), src, EnvStates.RENDER

def find_EnvState(estate:EnvStates) -> List[oroloEnv]:
  arr = [ find_EnvState(e) for e in estate.src if isinstance(e, EnvStates) ]
  return functools.reduce(lambda a,b:a+b,arr,[estate])

evaluator = {EnvStates.RESET: wait_reset, EnvStates.STEP: wait_step, EnvStates.MAKE:wait_make, EnvStates.RENDER: wait_render}

# ***************** Base Environment *****************
class oroloEnv:
  def __init__(self, device, op:EnvStates):
    self.device = device
    self.buffer: Optional[LowEnv] = None
    self.op = op
    if not LAZY : self.evaluate()
    pass

  def __repr__(self): return  f"<oroloEnv env : {self.buffer} >"

  def evaluate(self) -> LowEnv:
    if self.buffer is None:
      st = time.monotonic() 
      self.buffer, srcs, optype = evaluator[self.op.op](self)
      et = time.monotonic() 
      if DEBUG: print(f"{[x.op for x in find_EnvState(self.op)]} : -> FPS:{1/(et-st):.0f}")
      del self.op # prune our AST  to save space
    return self.buffer

  @staticmethod
  def make(x:oroloEnv, device='async'):
    return oroloEnv(device, EnvState(EnvStates.MAKE, tuple(), x)) 

  def compute(a:oroloEnv): return a.evaluate()

  #################################################################
  # if this was an async envirnonment we would deploy processes here
  def reset(x:oroloEnv, seed=None, args=None):
    return oroloEnv(x.device, EnvState(EnvStates.RESET, (x,), (seed,args)))
  def step(x:oroloEnv, action=None):
    return oroloEnv(x.device, EnvState(EnvStates.STEP, (x,), action))

  def render(x:oroloEnv, arg=None): 
    return oroloEnv(x.device, EnvState(EnvStates.RENDER, (x,), arg))
  def close(x:oroloEnv): pass


if __name__=='__main__':

  from games.TicTacToe import TicTacToe
  import random
  DEBUG = True
  envs = oroloEnv.make([TicTacToe() for _ in range(4)], device='async')
  seed = 7

  for epi in range(3):
    obs = envs.reset(seed=7).compute()
    for _ in range(3):
      action = random.randint(0,9)
      #obs,rew,done = envs.step(action=action).compute()
      obss,rews, dones = envs.step(action=[random.randint(0,9) for _ in range(4)]).compute()

