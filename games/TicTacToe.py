import numpy as np

''' 
Credits to: https://github.com/geohot/ai-notebooks  
'''

class TicTacToe():
  def __init__(self, state=None):
    self.reset()
    if state is not None:
      self.state = state

  class observation_space: 
    #dtype = 'i' # interger type # https://www.bookstack.cn/read/python3.9-library-zh/499befbdb032db77.md
    dtype = np.int
    shape = (11,)

  class action_space: 
    #dtype = 'i' # interger type # https://www.bookstack.cn/read/python3.9-library-zh/499befbdb032db77.md
    dtype = np.int
    shape = (1,)
    n = 9
      
  def reset(self):
    # [0->9:board, 10:win or loose, 11:current player turn]
    self.state = [0]*11
    # randomly pick starting player
    self.state[-1] = 1 #np.random.choice([-1,1])
    self.done = False
    return self.state

  def step(self, action):
    # Don't move in empty space or finished game
    reward=0
    if self.state[-2] == 1 or self.state[action]!= 0:
      reward = -10 # negative reward for every postion
    else:
      self.state[action] = self.state[-1]
      for (x,y,z) in [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]:
        if (self.state[-1] * (self.state[x] + self.state[y] + self.state[z])) == 3:
          reward += self.state[-1]  
          self.done = True
      reward = reward*self.state[-1]

    # if game finished, reward = 0, else store reward 
    if self.state[-2] != 0:
      reward = 0
    else: self.state[-2] = reward
    self.state[-1] *= -1 # change players

    # no more empty spots
    #self.done = np.all(np.array(self.state[0:9]) != 0)  
    if np.all(np.array(self.state[0:9]) != 0):  
      self.done = True
      
    return self.state, reward, self.done

  def render(self):
    print(f'Turn {self.state[-1]}, Done:{self.done}')
    print(np.array(self.state[0:9]).reshape(3,3))
 
    
  

if __name__ == "__main__":
  env = TicTacToe()
  state = env.reset()
  print(env.reset())
  print(env.step(4))
  print(env.step(0))
  print(env.step(3))
  print(env.step(1))
  print(env.step(6))
  print(env.step(2))




  #print(env.step(0))
  #print(env.observation_space.shape)
  #print(env.action_space.n)
