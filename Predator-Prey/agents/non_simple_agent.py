import numpy as np

class RandomAgent(object):
    def __init__(self, action_dim):
        self._action_dim = action_dim

    def act(self, obs):

        if np.random.rand() < 3./8. :
            return 2
        else:
            return np.random.randint(self._action_dim)

        # return 2

    def train(self, minibatch, step):
        return

class StaticAgent(object):
    def __init__(self, action):
        self._action = action

    def act(self, obs):
        return self._action

    def train(self, minibatch, step):
        return
