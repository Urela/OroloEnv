from copy import deepcopy
import numpy as np
import multiprocessing as mp
from enum import Enum
from typing import Union, Tuple, Optional, NamedTuple
from collections.abc import Callable
from typing import Sequence, List, Any
import time

class AsyncState(Enum):
  DEFAULT = "default"
  WAITING_RESET = "reset"
  WAITING_STEP = "step"
  WAITING_CALL = "call"
class observation_space: 
  dtype = 'i' # interger type # https://www.bookstack.cn/read/python3.9-library-zh/499befbdb032db77.md
  shape = (11,)

class action_space: 
  dtype = 'i' # interger type # https://www.bookstack.cn/read/python3.9-library-zh/499befbdb032db77.md
  shape = (1,)
  n = 9

Space = Union[observation_space, action_space]

#----------------------------------------------------------------------------------------------------
def create_SharedMemory(space: Space, n: int = 1, ctx=mp) -> mp.Array:
  #return ctx.Array(space.dtype, n * int(np.prod(space.shape)))
  return ctx.Array('i', n * int(np.prod(space.shape)))

def read_SharedMemory(space: Space, shared_memory, n: int = 1):
  #return np.frombuffer(shared_memory.get_obj(), dtype=space.dtype).reshape((n,) + space.shape)
  return np.frombuffer(shared_memory.get_obj(), dtype='i').reshape((n,) + space.shape)

def create_EmptyArray(space: Space, n: int = 1, fn=np.zeros):
  shape = space.shape if (n is None) else (n,) + space.shape
  #return fn(shape, dtype=space.dtype)
  return fn(shape, dtype='i')

#def WriteToMemory(space: Space, index: int, value: np.ndarray, shared_memory: mp.Array):
def WriteToMemory(space: Space, index: int, value: np.ndarray, shared_memory):
  size = int(np.prod(space.shape))
  #destination = np.frombuffer(shared_memory.get_obj(), dtype=space.dtype)
  destination = np.frombuffer(shared_memory.get_obj(), dtype='i')
  np.copyto(
    destination[index * size : (index + 1) * size],
    np.asarray(value, dtype=space.dtype).flatten(),
  )

def _worker_shared_memory(index, env, pipe, parent_pipe, shared_memory, error_queue):
  assert shared_memory is not None
  observation_space = env.observation_space
  parent_pipe.close()
  try:
    while True:
      command, data = pipe.recv()
      #print(command, data)
      if command == "reset":
        observation = env.reset()
        WriteToMemory(observation_space, index, observation, shared_memory)
        pipe.send(((None,), True))
      elif command == "step":
        (observation,reward,done) = env.step(data)

        if done: observation = env.reset()
        WriteToMemory(observation_space, index, observation, shared_memory)
        pipe.send(((None, reward, done), True))
      elif command == "close":
        pipe.send(((None,), True))
        break
      else: raise RuntimeError( f"Received unknown command `{command}`. Must be one of <`reset`, `seed`, `close`>.")
  except (KeyboardInterrupt, Exception):
    error_queue.put((index,) + sys.exc_info()[:2])
    pipe.send((None, False))
    print("SUCCESS = False")
  finally:
    #env.close()
    pass


import contextlib
import os
@contextlib.contextmanager
def clear_mpi_env_vars():
    """Clears the MPI of environment variables."""
    removed_environment = {}
    for k, v in list(os.environ.items()):
        for prefix in ["OMPI_", "PMI_"]:
            if k.startswith(prefix):
                removed_environment[k] = v
                del os.environ[k]
    try:
        yield
    finally:
        os.environ.update(removed_environment)

#----------------------------------------------------------------------------------------------------

# we only mutate environment
class AsyncEnv:
  def __init__(
    self,
    env_fxns: Sequence['Env'], # Functions that create the environments.
    obs_space: Optional[Space] = None,    # Observation space of a single environment
    act_space: Optional[Space] = None,    # Action space of a single environment
    copy: bool = True,
    #---------- new stuff ----------
    shared_memory: bool = True,
    context: Optional[str] = None,
    daemon: bool = True,
    worker: Optional[Callable] = None,
    #-------------------------------
  ):

    self.envs = env_fxns
    self.copy = copy

    self.num_envs=len(self.envs)
    if (obs_space is None) or (act_space is None):
      obs_space = obs_space or self.envs[0].observation_space
      act_space = act_space or self.envs[0].action_space 

    self.single_obs_space = obs_space
    self.single_act_space = act_space

    self._rewards = np.zeros((self.num_envs), dtype=np.float64)
    self._dones = np.zeros((self.num_envs), dtype=np.bool_)
    self._actions = None

    #---------- new stuff ----------
    ctx = mp.get_context(context)
    self.shared_memory = shared_memory

    if self.shared_memory:
      try:
        _obs_buffer = create_SharedMemory(self.single_obs_space, n=self.num_envs, ctx=ctx)
        self.obs = read_SharedMemory(self.single_obs_space, _obs_buffer, n=self.num_envs)
      except Exception as err: print(f"Unexpected {err=}, {type(err)=}")
    else:
      _obs_buffer = None 
      self.obs = create_EmptyArray(self.single_obs_space, n=self.num_envs, fn=np.zeros)

    self.parent_pipes, self.processes = [], []
    self.error_queue = ctx.Queue()

    target = worker or _worker_shared_memory 

    with clear_mpi_env_vars():
      for idx, env in enumerate(self.envs):
        parent_pipe, child_pipe = ctx.Pipe()
        process = ctx.Process(
          target=target,
          name=f"Worker<{type(self).__name__}>-{idx}",
          args=( idx, env, child_pipe, parent_pipe, _obs_buffer, self.error_queue, ),
        )

        self.parent_pipes.append(parent_pipe)
        self.processes.append(process)

        process.daemon = daemon
        process.start()
        child_pipe.close()
    self._state = AsyncState.DEFAULT
    pass
 


  @classmethod
  def make(cls, env): return cls(env)
  def compute(self): return self.envs
  def step_async(self, actions):
    """Sets :attr:`_actions` for use by the :meth:`step_wait` by converting the ``actions`` to an iterable version."""
    #self._actions = iterate(self.action_space, actions)
    #self._actions = actions #iterate(self.action_space, actions)


    if self._state != AsyncState.DEFAULT:
      raise NotImplementedError(
        f"Calling `step_async` while waiting for a pending call to `{self._state.value}` to complete.",
        self._state.value,
      )

    for pipe, action in zip(self.parent_pipes, actions):
      pipe.send(("step", action))
    self._state = AsyncState.WAITING_STEP

    #print(self._state, "step_async", ">>>>>>>>>>>>>>>>>>>>>")


  def step_wait( self, timeout: Optional[Union[int, float]] = None
      ) :# -> Tuple[Any, NDArray[Any], NDArray[Any], NDArray[Any], dict]:
    #self._assert_is_running()
    if self._state != AsyncState.WAITING_STEP:
      raise NotImplementedError(
          "Calling `step_wait` without any prior call " "to `step_async`.",
          AsyncState.WAITING_STEP.value,
      )

    if not self._poll(timeout):
      self._state = AsyncState.DEFAULT
      raise mp.TimeoutError(
          f"The call to `step_wait` has timed out after {timeout} second(s)."
      )

    observations_list, rewards, dones = [], [], [] 
    successes = []
    #print(self.parent_pipes)
    for i, pipe in enumerate(self.parent_pipes):
      result, success = pipe.recv()
      #print("pipe success", success)
      successes.append(success)
      if success:
        obs, rew, done = result
        #print(obs, rew, done )
        observations_list.append(obs)
        rewards.append(rew)
        dones.append(done)

    self._raise_if_errors(successes)
    self._state = AsyncState.DEFAULT

    if not self.shared_memory:
      self.obs = concatenate(
      self.single_observation_space,
      observations_list,
      self.obs,
    )

    #print(self._state, "step_wait", ">>>>>>>>>>>>>>>>>>>>>")
    return (
      deepcopy(self.obs) if self.copy else self.obs,
      np.array(rewards),
      np.array(dones, dtype=np.bool_)
    )
        
  def step(self, actions): 
    self.step_async(actions)
    #return self.step_wait(timeout=1)
    return self.step_wait(timeout=2)

  def reset_wait(
    self,
    timeout: Optional[Union[int, float]] = None,
    seed: Optional[Union[int, List[int]]] = None,
    options: Optional[dict] = None,
  ):
    #self._assert_is_running()
    if self._state != AsyncState.WAITING_RESET:
      raise NotImplementedError("Calling `reset_wait` without any prior " "call to `reset_async`.",
        AsyncState.WAITING_RESET.value,
      )

    if not self._poll(timeout):
      self._state = AsyncState.DEFAULT
      raise mp.TimeoutError( f"The call to `reset_wait` has timed out after {timeout} second(s).")

    results, successes = zip(*[pipe.recv() for pipe in self.parent_pipes])

    self._raise_if_errors(successes)
    self._state = AsyncState.DEFAULT

    #results = zip(*results)
    if not self.shared_memory:
      self.obs = concatenate(
          self.single_obs_space, results, self.obs
      )

    #print(self._state, "reset_wait", ">>>>>>>>>>>>>>>>>>>>>")
    return (deepcopy(self.obs) if self.copy else self.obs)

  def reset_async(
    self,
    seed: Optional[Union[int, List[int]]] = None,
    options: Optional[dict] = None,
  ):
    """Send calls to the :obj:`reset` methods of the sub-environments."""
    if self._state != AsyncState.DEFAULT:
      raise NotImplementedError( f"Calling `reset_async` while waiting for a pending call to `{self._state.value}` to complete",self._state.value,)

    if seed is None: seed = [None for _ in range(self.num_envs)]
    if isinstance(seed, int): seed = [seed + i for i in range(self.num_envs)]
    assert len(seed) == self.num_envs

    for pipe, single_seed in zip(self.parent_pipes, seed):
      single_kwargs = {}
      if single_seed is not None: single_kwargs["seed"] = single_seed
      if options is not None: single_kwargs["options"] = options

      pipe.send(("reset", single_kwargs))
    self._state = AsyncState.WAITING_RESET

    #print(self._state, "reset_async", ">>>>>>>>>>>>>>>>>>>>>")
    pass

  #def reset( self,*, seed: Optional[Union[int, List[int]]] = None, options: Optional[dict] = None,):
  def reset( self, seed: Optional[Union[int, List[int]]] = None, options: Optional[dict] = None,):

    self.reset_async(seed=seed, options=options)
    return self.reset_wait(seed=seed, options=options)

  def render(self,args): 
    for e in self.envs: print(e.render() )
    return self
  def _poll(self, timeout=None):
    #self._assert_is_running()
    if timeout is None:
        return True
    end_time = time.perf_counter() + timeout
    delta = None
    for pipe in self.parent_pipes:
        delta = max(end_time - time.perf_counter(), 0)
        if pipe is None:
            return False
        if pipe.closed or (not pipe.poll(delta)):
            return False
    return True


  def _raise_if_errors(self, successes):
    if all(successes):
        return

    num_errors = self.num_envs - sum(successes)
    assert num_errors > 0
    for i in range(num_errors):
      index, exctype, value = self.error_queue.get()
      print(f"Received the following error from Worker-{index}: {exctype.__name__}: {value}")
      #logger.error(
      #    f"Received the following error from Worker-{index}: {exctype.__name__}: {value}"
      #)
      #logger.error(f"Shutting down Worker-{index}.")
      self.parent_pipes[index].close()
      self.parent_pipes[index] = None

      if i == num_errors - 1:
        print("Raising the last exception back to the main process.")
        #logger.error("Raising the last exception back to the main process.")
        raise exctype(value)

  def __del__(self):
      """On deleting the object, checks that the vector environment is closed."""
      if not getattr(self, "closed", True) and hasattr(self, "_state"):
          self.close(terminate=True)

