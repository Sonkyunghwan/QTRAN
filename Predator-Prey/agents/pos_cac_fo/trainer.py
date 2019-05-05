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
import tensorflow as tf
from agents.pos_cac_fo.agent import Agent
from agents.simple_agent import RandomAgent as NonLearningAgent
from agents.evaluation import Evaluation
from agents.simple_agent import StaticAgent as StAgent
from agents.simple_agent import ActiveAgent as AcAgent
import logging
import config
from envs.gui import canvas

FLAGS = config.flags.FLAGS
logger = logging.getLogger("Agent")
result = logging.getLogger('Result')

training_step = FLAGS.training_step
testing_step = FLAGS.testing_step

epsilon_dec = 2.0/training_step
epsilon_min = 0.1


class Trainer(object):

    def __init__(self, env):
        logger.info("Centralized DQN Trainer is created")

        self._env = env 
        self._eval = Evaluation()
        self._n_predator = FLAGS.n_predator
        self._n_prey = FLAGS.n_prey
        self._agent_profile = self._env.get_agent_profile()
        self._agent_precedence = self._env.agent_precedence

        self._agent = Agent(self._agent_profile["predator"]["act_dim"], self._agent_profile["predator"]["obs_dim"][0])
        self._prey_agent = AcAgent(5)

        self.epsilon = 1.0
        if FLAGS.load_nn:
            self.epsilon = epsilon_min

        if FLAGS.gui:
            self.canvas = canvas.Canvas(self._n_predator, self._n_prey, FLAGS.map_size)
            self.canvas.setup()
    def learn(self):

        step = 0
        episode = 0
        print_flag = False
        count = 1

        while step < training_step:
            episode += 1
            ep_step = 0
            obs = self._env.reset()
            state = self._env.get_full_encoding()[:, :, 2]
            total_reward = 0
            total_reward_pos = 0
            total_reward_neg = 0
            self.random_action_generator()
            while True:
                step += 1
                ep_step += 1
                action = self.get_action(obs, step, state)
                obs_n, reward, done, info = self._env.step(action)
                state_n = self._env.get_full_encoding()[:, :, 2]
                done_single = sum(done) > 0

                self.train_agents(state, action, reward, state_n, done_single)
                obs = obs_n
                state = state_n
                total_reward += np.sum(reward)
                if np.sum(reward) >= 0:
                    total_reward_pos += np.sum(reward)
                else:
                    total_reward_neg += np.sum(reward)

                if is_episode_done(done, step) or ep_step >= FLAGS.max_step :
                    # print step, ep_step, total_reward
                    if print_flag and episode % FLAGS.eval_step == 1:
                        print "[train_ep %d]" % (episode), "\treward", total_reward_pos, total_reward_neg
                    break

            if episode % FLAGS.eval_step == 0:
                self.test(episode)

        self._eval.summarize()
    
    def random_action_generator(self):
        rand_unit = np.random.uniform(size = (FLAGS.n_predator, 5))
        self.rand = rand_unit / np.sum(rand_unit, axis=1, keepdims=True)
        

    def get_action(self, obs, step, state, train=True):
        act_n = []
        if train == True:
            self.epsilon = max(self.epsilon - epsilon_dec, epsilon_min)

        # Action of predator

        action_list = self._agent.act(state)
        for i in range(self._n_predator):
            if train and (step < FLAGS.m_size * FLAGS.pre_train_step or np.random.rand() < self.epsilon):
                action = np.random.choice(5)
                act_n.append(action)
            else:              
                act_n.append(action_list[i])



        # Action of prey
        for i in range(FLAGS.n_prey):
            act_n.append(self._prey_agent.act(state, i))
        # act_n[1] = 2

        return np.array(act_n, dtype=np.int32)

    def train_agents(self, state, action, reward, state_n, done):
        self._agent.train(state, action, reward, state_n, done)

    def test(self, curr_ep=None):

        step = 0
        episode = 0

        test_flag = FLAGS.kt
        sum_reward = 0
        sum_reward_pos = 0
        sum_reward_neg = 0
        while step < testing_step:
            episode += 1
            obs = self._env.reset()
            state = self._env.get_full_encoding()[:, :, 2]
            if test_flag:
                print "\nInit\n", state
            total_reward = 0
            total_reward_pos = 0
            total_reward_neg = 0

            ep_step = 0

            while True:

                step += 1
                ep_step += 1

                action = self.get_action(obs, step, state, False)
                obs_n, reward, done, info = self._env.step(action)
                state_n = self._env.get_full_encoding()[:, :, 2]
                state_next = state_to_index(state_n)
                if FLAGS.gui:
                    self.canvas.draw(state_next, done, "Score:" + str(total_reward) + ", Step:" + str(ep_step))

                if test_flag:
                    aa = raw_input('>')
                    if aa == 'c':
                        test_flag = False
                    print action
                    print state_n
                    print reward

                obs = obs_n
                state = state_n
                r = np.sum(reward)
                # if r == 0.1:
                #     r = r * (-1.) * FLAGS.penalty
                total_reward += r # * (FLAGS.df ** (ep_step-1))
                if r > 0:
                    total_reward_pos += r
                else:
                    total_reward_neg -= r


                if is_episode_done(done, step, "test") or ep_step >= FLAGS.max_step:

                    if FLAGS.gui:
                        self.canvas.draw(state_next, done, "Hello", "Score:" + str(total_reward) + ", Step:" + str(ep_step))

                    break
            sum_reward += total_reward
            sum_reward_pos += total_reward_pos
            sum_reward_neg += total_reward_neg
        if FLAGS.scenario =="pursuit":
            print "Test result: Average steps to capture: ", curr_ep, float(step)/episode
            self._eval.update_value("training result: ", float(step)/episode, curr_ep)
        elif FLAGS.scenario =="endless" or FLAGS.scenario =="endless2" or FLAGS.scenario =="endless3":
            print "Average reward:", FLAGS.penalty, curr_ep, sum_reward /episode, sum_reward_pos/episode, sum_reward_neg/episode
            self._eval.update_value("training result: ", sum_reward/episode, curr_ep)
            self._agent.logging(sum_reward/episode, curr_ep * 100)


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

def state_to_index(state):
    """
    For the single agent case, the state is only related to the position of agent 1
    :param state:
    :return:
    """

    ret = np.zeros(2 * (FLAGS.n_predator + FLAGS.n_prey))
    for i in range(FLAGS.n_predator + FLAGS.n_prey):
        p = np.argwhere(np.array(state)==i+1)[0]
        #p = self.get_pos_by_id(state, i+1)
        ret[2 * i] = p[0]
        ret[2 * i + 1] = p[1]


    return ret

    


