import numpy as np
import config
FLAGS = config.flags.FLAGS

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
class ActiveAgent(object):
    def __init__(self, action_dim):
        self._action_dim = action_dim
        self._n_predator = FLAGS.n_predator
        self._n_prey = FLAGS.n_prey
        self._state_dim = 2 * (self._n_predator + self._n_prey)


    def act(self, state, num):
        state_i = self.state_to_index(state)
        # s = np.reshape(state_i, [self._state_dim/2, 2])
        self.map_size = FLAGS.map_size
        threshold = self.map_size * 2.0
        i = self._n_predator + num 
        action_i = 2
        if np.random.rand() < 1.0 :
            return np.random.randint(self._action_dim)
        pos_i = np.argwhere(np.array(state)==i+1)[0]
        for j in range(FLAGS.n_predator):
            pos_j = np.argwhere(np.array(state)==j+1)[0]
            if abs(pos_i[0] - pos_j[0]) + abs(pos_i[1] - pos_j[1]) < threshold:
                p = np.zeros(5)
                threshold = abs(pos_i[0] - pos_j[0]) + abs(pos_i[1] - pos_j[1])
                if (pos_i[0] - pos_j[0]) >= abs((pos_i[1] - pos_j[1])):
                    p[0] = 1
                elif (pos_i[1] - pos_j[1]) >= abs((pos_i[0] - pos_j[0])):
                    p[1] = 1
                elif (pos_i[1] - pos_j[1]) <= -abs((pos_i[0] - pos_j[0])):
                    p[3] = 1
                elif (pos_i[0] - pos_j[0]) <= -abs((pos_i[1] - pos_j[1])):
                    p[4] = 1
                action_i = np.random.choice(self._action_dim, p=p/np.sum(p))
        if threshold == 1:
            return 2
        return action_i

    def state_to_index(self, state):
            """
            For the single agent case, the state is only related to the position of agent 1
            :param state:
            :return:
            """
            # p1, p2 = self.get_predator_pos(state)
            ret = np.zeros(self._state_dim)
            for i in range(FLAGS.n_predator + FLAGS.n_prey):
                p = np.argwhere(np.array(state)==i+1)[0]
                #p = self.get_pos_by_id(state, i+1)
                ret[2 * i] = p[0]
                ret[2 * i + 1] = p[1]

            return ret
