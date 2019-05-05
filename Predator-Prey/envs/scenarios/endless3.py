import numpy as np
from collections import deque
from envs.grid_core import World
from envs.grid_core import CoreAgent as Agent
from envs.scenario import BaseScenario
import config

FLAGS = config.flags.FLAGS

n_predator = FLAGS.n_predator
n_prey = FLAGS.n_prey
n_prey1 = FLAGS.n_prey1
n_prey2 = FLAGS.n_prey2
map_size = FLAGS.map_size
penalty = FLAGS.penalty

class Prey(Agent):
    def __init__(self):
        super(Prey, self).__init__("prey", "green")
        self._movement_mask = np.array(
            [[0,1,0],
             [1,0,1],
             [0,1,0]], dtype=np.int8)

    def cannot_move(self):
        minimap = (self._obs[:,:,0] != 0)
        return np.sum(minimap*self._movement_mask)==4

    def can_observe_predator(self):
        shape = np.shape(self._obs[:,:,0])
        obs_size = shape[0]*shape[1]
        obs = np.reshape(self._obs[:,:,0] *self._movement_mask, obs_size)
        ret = np.shape(np.where(obs == 3))[1] > 0
        return ret

    def can_observe_two_predator(self):
        shape = np.shape(self._obs[:,:,0])
        obs_size = shape[0]*shape[1]
        obs = np.reshape(self._obs[:,:,0] *self._movement_mask, obs_size)
        ret = np.shape(np.where(obs == 3))[1] > 1
        return ret

    def can_observe_three_predator(self):
        shape = np.shape(self._obs[:,:,0])
        obs_size = shape[0]*shape[1]
        obs = np.reshape(self._obs[:,:,0] *self._movement_mask, obs_size)
        ret = np.shape(np.where(obs == 3))[1] > 2
        return ret

class Prey2(Agent):
    def __init__(self):
        super(Prey2, self).__init__("prey2", "red")
        self._movement_mask = np.array(
            [[0,1,0],
             [1,0,1],
             [0,1,0]], dtype=np.int8)

    def cannot_move(self):
        minimap = (self._obs[:,:,0] != 0)
        return np.sum(minimap*self._movement_mask)==4

    def can_observe_predator(self):
        shape = np.shape(self._obs[:,:,0])
        obs_size = shape[0]*shape[1]
        obs = np.reshape(self._obs[:,:,0] *self._movement_mask, obs_size)
        ret = np.shape(np.where(obs == 3))[1] > 0
        return ret

    def can_observe_two_predator(self):
        shape = np.shape(self._obs[:,:,0])
        obs_size = shape[0]*shape[1]
        obs = np.reshape(self._obs[:,:,0] *self._movement_mask, obs_size)
        ret = np.shape(np.where(obs == 3))[1] > 1
        return ret

    def can_observe_three_predator(self):
        shape = np.shape(self._obs[:,:,0])
        obs_size = shape[0]*shape[1]
        obs = np.reshape(self._obs[:,:,0] *self._movement_mask, obs_size)
        ret = np.shape(np.where(obs == 3))[1] > 2
        return ret

class Predator(Agent):
    def __init__(self):
        super(Predator, self).__init__("predator", "blue")
        self._obs = deque(maxlen=FLAGS.history_len)
        self.obs_range = 1

    def can_observe_prey(self):
        shape = np.shape(self._obs)
        obs_size = shape[1]*shape[2]
        obs = np.reshape(self._obs, obs_size)
        ret = np.shape(np.where(obs > 3))[1] > 0
        return ret

    def update_obs(self, obs):
        self._obs.append(obs[:,:,0]) # use only the first channel

    def fill_obs(self):
        # fill the whole history with the current observation
        for i in range(FLAGS.history_len-1):
            self._obs.append(self._obs[-1])

class Scenario(BaseScenario):
    def __init__(self):
        self.prey_captured = False

    def make_world(self):
        world = World(width=map_size, height=map_size)

        agents = []
        self.atype_to_idx = {
            "predator": [],
            "prey": [],
            "prey2": []
        }

        # add predators
        for i in xrange(n_predator):
            agents.append(Predator())
            self.atype_to_idx["predator"].append(i)

        # add preys
        for i in xrange(n_prey1):
            agents.append(Prey())
            self.atype_to_idx["prey"].append(n_predator + i)

        for i in xrange(n_prey2):
            agents.append(Prey2())
            self.atype_to_idx["prey2"].append(n_predator + n_prey1 + i)

        world.agents = agents
        for i, agent in enumerate(world.agents):
            agent.id = i + 1
            agent.silent = True 

        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        world.empty_grid()

        # randomly place agent
        for agent in world.agents:
            world.placeObj(agent)

        world.set_observations()

        # fill the history with current observation
        for i in self.atype_to_idx["predator"]:
            world.agents[i].fill_obs()

        self.prey_captured = False

    def reward(self, agent, world):
        if agent.itype == "predator":
            reward = 0.
            count = 0
            for i in self.atype_to_idx["prey"]:
                # reward += -0.01
                prey = world.agents[i]
                # if prey.can_observe_three_predator():
                #     reward += 10.0
                if prey.can_observe_predator():
                    reward += +1.0
                    # print "WIN"
                    # print "CATCH"
                # elif prey.can_observe_predator():
                #     # print "LOSE"
                #     # reward += 0.
                #     reward += +penalty/10.
                    # if penalty > 10:
                    #     reward += (penalty-10)/10.
                    # # else:
                    # reward += +1.
                # if prey.can_observe_predator():
                #     count += 1
            for i in self.atype_to_idx["prey2"]:
                # reward += -0.01
                prey = world.agents[i]
                # if prey.can_observe_three_predator():
                #     reward += 10.0
                if prey.can_observe_two_predator():
                    reward += 1.0
                    # print "WIN"
                    # print "CATCH"
                elif prey.can_observe_predator():
                    # print "LOSE"
                    # reward += 0.
                    reward += -penalty/10.
            # if reward > 1:
            #     print "CATCH"
            # if count > 1:
            #     reward += 1.0
            # elif count == 1:
            #     reward += -penalty/10.
            # else:
            #     reward += 0.


            return reward/(n_predator)

        else: # if prey
            if agent.cannot_move():
                return 0

        return 0

    def observation(self, agent, world):
        # print agent.get_obs.shape
        obs = np.array(agent.get_obs()).flatten()
        return obs

    def done(self, agent, world):
        if agent.itype == "prey":
            if agent.can_observe_predator():
                # world.resetObj(agent)
                return True
        if agent.itype == "prey2":
            if agent.can_observe_two_predator():
                # world.resetObj(agent)
                return True
        # if agent.itype == "predator":
        #     if agent.can_observe_prey():
        #         # world.resetObj(agent)
        #         return True
        return False
        #return self.prey_captured
