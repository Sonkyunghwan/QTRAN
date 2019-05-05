#!/usr/bin/env python
# coding=utf8

"""
===========================================
 :mod:`qlearn` Q-Learning
===========================================


설명
=====

Choose action based on q-learning algorithm

"""

import numpy as np
from agents.pos_cac_fo.agent import Agent
from agents.simple_agent import RandomAgent as NonLearningAgent
from agents.evaluation import Evaluation
import logging
import config

FLAGS = config.flags.FLAGS
logger = logging.getLogger("Agent")
result = logging.getLogger('Result')

training_step = FLAGS.training_step
testing_step = FLAGS.testing_step

if FLAGS.epsilon == "No":
    print "No epsilon decreasing"
    epsilon_dec = 0.0/training_step
elif FLAGS.epsilon == "Yes":
    print "Epsilon decreasing"
    epsilon_dec = 2.0/training_step
epsilon_min = 0.1


class Trainer(object):

    def __init__(self, env):
        logger.info("Centralized DQN Trainer is created")

        self._env = env
        self._eval = Evaluation()
        self._n_predator = FLAGS.n_predator
        self._n_prey = FLAGS.n_prey
        self.action_dim = self._env.call_action_dim()
        self.state_dim = self._env.call_state_dim()

        self._agent = Agent(self.action_dim, self.state_dim)

        self.epsilon = 1.0

    def learn(self):

        step = 0
        episode = 0
        print_flag = False
        array = np.zeros([FLAGS.training_step/FLAGS.eval_step,4])
        while step < training_step:
            episode += 1
            ep_step = 0
            obs = self._env.reset()
            state = obs
            total_reward = 0

            while True:
                step += 1
                ep_step += 1
                action = self.get_action(obs, step, state)
                obs_n, reward, done, info = self._env.step(action)
                state_n = obs_n
                
                done_single = sum(done) > 0
                if ep_step >= FLAGS.max_step :
                    done_single = True
                self.train_agents(state, action, reward, state_n, done_single)

                obs = obs_n
                state = state_n
                total_reward += np.sum(reward) * (FLAGS.df ** (ep_step-1))
                # if step % 100 ==0:
                #     print step, self._agent.q()
                if is_episode_done(done, step) or ep_step >= FLAGS.max_step :
                    if print_flag:
                        print "[train_ep %d]" % (episode),"\tstep:", step, "\tep_step:", ep_step, "\treward", total_reward
                    break
                

            if episode % FLAGS.eval_step == 0:

                self.test(episode)


        self._eval.summarize()


    def get_action(self, obs, step, state, train=True):
        act_n = []
        self.epsilon = max(self.epsilon - epsilon_dec, epsilon_min)

        # Action of predator
        action_list = self._agent.act(state)

        for i in range(self._n_predator):
            if train and (step < FLAGS.m_size * FLAGS.pre_train_step or np.random.rand() < self.epsilon):  # with prob. epsilon
                action = np.random.choice(self.action_dim)
                act_n.append(action)
            else:
                act_n.append(action_list[i])
                


        return np.array(act_n, dtype=np.int32)

    def train_agents(self, state, action, reward, state_n, done):
        self._agent.train(state, action, reward, state_n, done)

    def test(self, curr_ep=None):

        step = 0
        episode = 0

        test_flag = FLAGS.kt
        sum_reward = 0
        while step < testing_step:
            episode += 1
            obs = self._env.reset()
            state = obs
            if test_flag:
                print "\nInit\n", state
            total_reward = 0

            ep_step = 0

            while True:

                step += 1
                ep_step += 1

                action = self.get_action(obs, step, state, False)

                obs_n, reward, done, info = self._env.step(action)
                state_n = obs_n

                if test_flag:
                    aa = raw_input('>')
                    if aa == 'c':
                        test_flag = False
                    print action
                    print state_n
                    print reward

                obs = obs_n
                state = state_n
                total_reward += np.sum(reward) * (FLAGS.df ** (ep_step-1))

                if is_episode_done(done, step, "test") or ep_step >= FLAGS.max_step:
                    break
            sum_reward += total_reward

        print "Algorithm ", FLAGS.algorithm, ",Average reward: ", curr_ep, sum_reward /episode
        self._eval.update_value("test_result", sum_reward /episode, curr_ep)



def is_episode_done(done, step, e_type="train"):

    if e_type == "test":
        if sum(done) > 0 or step >= FLAGS.testing_step:
            return True
        else:
            return False

    else:
        if sum(done) > 0 or step >= FLAGS.training_step:
            return True
        else:
            return False


