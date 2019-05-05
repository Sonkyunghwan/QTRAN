#!/usr/bin/env python
# coding=utf8

"""
===========================================
 :mod:`qlearn` Q-Learning
===========================================

=====

Choose action based on q-learning algorithm
"""

import numpy as np
import tensorflow as tf
import math
from agents.pos_cac_fo.dq_network import *
from agents.pos_cac_fo.replay_buffer import *
from agents.evaluation import Evaluation

import logging
import config

FLAGS = config.flags.FLAGS
logger = logging.getLogger("Agent")
result = logging.getLogger('Result')


class Agent(object):

    def __init__(self, action_dim, obs_dim, name=""):
        logger.info("Centralized DQN Agent")

        self._n_predator = FLAGS.n_predator
        self._n_prey = FLAGS.n_prey
        self.map_size = FLAGS.map_size

        self._obs_dim = obs_dim

        self._action_dim = action_dim * self._n_predator
        self._action_dim_single = action_dim
        self._n_object = (self._n_predator + self._n_prey)
        self._state_dim = 2 * (self._n_predator + self._n_prey)
        self._state_dim_single = (self.map_size**2)

        self._name = name
        self.update_cnt = 0
        self.target_update_period = 10000

        self.df = FLAGS.df
        self.lr = FLAGS.lr

        # Make Q-network
        tf.reset_default_graph()
        my_graph = tf.Graph()

        with my_graph.as_default():
            self.sess = tf.Session(graph=my_graph, config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
            self.q_network = DQNetwork(self.sess, self._state_dim, self._action_dim_single, self._n_predator) 
            self.sess.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver()
            if FLAGS.load_nn:
                print "LOAD!"
                self.saver.restore(self.sess, "./results/nn3/n-"+str(FLAGS.n_predator)+"-s-endless3-map-"+str(FLAGS.map_size)+"-penalty-"+str(FLAGS.penalty)+"-a-"+str(FLAGS.algorithm)+"-lr-0.0005-ms-32-seed-"+str(FLAGS.seed)+"-"+str(FLAGS.comment))
            self.train_writer = tf.summary.FileWriter(config.tb_filename, self.sess.graph)

        self.replay_buffer = ReplayBuffer()

        self._eval = Evaluation()
        self.q_prev = None
        self.s_array = np.random.randint(self.map_size, size = (2 * (FLAGS.n_prey + FLAGS.n_predator), 100))

    def act(self, state):

        predator_rand = np.random.permutation(FLAGS.n_predator)
        prey_rand = np.random.permutation(FLAGS.n_prey)      

        s = self.state_to_index(state)
    
        action = self.q_network.get_action(s[None])[0]
    
        return action



    def train(self, state, action, reward, state_n, done):

        
        predator_rand = np.random.permutation(FLAGS.n_predator)
        prey_rand = np.random.permutation(FLAGS.n_prey)
        
        a = self.action_to_onehot(action)
        s = self.state_to_index(state)
        s_n = self.state_to_index(state_n)
        r = np.sum(reward)

        self.store_sample(s, a, r, s_n, done)

        self.update_network()
        return 0

    def store_sample(self, s, a, r, s_n, done):
        self.replay_buffer.add_to_memory((s, a, r, s_n, done))
        return 0

    def update_network(self):
        self.update_cnt += 1
        if len(self.replay_buffer.replay_memory) < FLAGS.pre_train_step*minibatch_size:
            return 0

        minibatch = self.replay_buffer.sample_from_memory()
        self.q_network.training_qnet(minibatch)


        if self.update_cnt % self.target_update_period == 0:
            self.q_network.training_target_qnet()

        if self.update_cnt % 10000 == 0:
            self.saver.save(self.sess, config.nn_filename, self.update_cnt)

        return 0

    def state_to_index(self, state):
        """
        For the single agent case, the state is only related to the position of agent 1
        :param state:
        :return:
        """

        ret = np.zeros(self._state_dim)
        for i in range(FLAGS.n_predator + FLAGS.n_prey):
            p = np.argwhere(np.array(state)==i+1)[0]

            ret[2 * i] = (p[0] - FLAGS.map_size /2.) / FLAGS.map_size
            ret[2 * i + 1] = (p[1] - FLAGS.map_size /2.) / FLAGS.map_size


        return ret

    def get_predator_pos(self, state):
        """
        return position of agent 1 and 2
        :param state: input is state
        :return:
        """
        state_list = list(np.array(state).ravel())
        return state_list.index(1), state_list.index(2)

    def get_pos_by_id(self, state, id):
        state_list = list(np.array(state).ravel())
        return state_list.index(id)

    def onehot(self, index, size):
        n_hot = np.zeros(size)
        n_hot[index] = 1.0
        return n_hot

    def index_to_action(self, index):
        action_list = []
        for i in range(FLAGS.n_predator-1):
            action_list.append(index%5)
            index = index/5
        action_list.append(index)
        return action_list

    def action_to_index(self, action):
        index = 0
        for i in range(FLAGS.n_predator):
            index += action[i] * 5 ** i
        return index

    def action_to_onehot(self, action):
        onehot = np.zeros([self._n_predator, self._action_dim_single])
        for i in range(self._n_predator):
            onehot[i, action[i]] = 1
        return onehot

    def onehot_to_action(self, onehot):
        action = np.zeros([self._n_predator])
        for i in range(self._n_predator):
            action[i] = int(np.argmax(onehot[i]))
        return action

    def q_diff(self):

        # if self.q_prev == None:
        #     self.q_prev = self.q()
        #     return

        # q_next = self.q()

        # d = 0.0
        # a = 0.0
        # for i in range(100):
        #     d += math.fabs(self.q_prev[i] - q_next[i])
        #     a += q_next[i]
        # avg = a/100

        # self._eval.update_value("q_avg", avg, self.update_cnt)
        # self._eval.update_value("q_diff", d, self.update_cnt)

        # self.q_prev = q_next

        # print self.update_cnt, d, avg

        print self.update_cnt

    def q(self):
        q_value = []
        # for i in range(100):
        #     s = self.s_array[:,i]
        #     s = (s - FLAGS.map_size /2.) / FLAGS.map_size
        #     q = self.q_network.get_target_q_values(s[None])[0]
        #     q_max = np.max(q)
        #     q_value.append(q_max)
        return q_value

    def logging(self, reward, step):

        summary = self.q_network.summary(reward, step) 
 
        self.train_writer.add_summary(summary, step)