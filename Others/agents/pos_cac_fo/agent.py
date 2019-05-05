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
import math
from agents.pos_cac_fo.dq_network import *
from agents.pos_cac_fo.replay_buffer import *
from agents.evaluation import Evaluation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import logging
import config

FLAGS = config.flags.FLAGS
logger = logging.getLogger("Agent")
result = logging.getLogger('Result')


class Agent(object):

    def __init__(self, action_dim, obs_dim, name=""):
        logger.info("Centralized DQN Agent")


        self._obs_dim = obs_dim
        self._n_player = FLAGS.n_predator
        self._action_dim = action_dim * self._n_player
        self._action_dim_single = action_dim
        self._state_dim = obs_dim

        self._name = name
        self.update_cnt = 0
        self.target_update_period = 3000

        self.df = FLAGS.df
        self.lr = FLAGS.lr

        # Make Q-network
        tf.reset_default_graph()
        my_graph = tf.Graph()

        with my_graph.as_default():
            self.sess = tf.Session(graph=my_graph, config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
            self.q_network = DQNetwork(self.sess, self._state_dim, self._action_dim_single, self._n_player)
            self.sess.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver()
            if FLAGS.load_nn:
                print "LOAD!"
                self.saver.restore(self.sess, FLAGS.nn_file)

        self.replay_buffer = ReplayBuffer()

        self._eval = Evaluation()
        self.q_prev = None

        self.ims = []
        plt.clf()
        self.fig = plt.figure()
        self.axes = plt.gca()
        plt.xticks(list(range(0,25,5)))
        plt.yticks(list(range(0,25,5)))
        self.axes.tick_params(axis='both',labelsize = 15)



    def act(self, state):
        state_i = state
        s = np.reshape(state_i, self._state_dim)
        
        action = self.q_network.get_action(s[None])[0]

        return action

    def train(self, state, action, reward, state_n, done):

        a = self.action_to_onehot(action)
        s = state
        s_n = state_n
        r = np.sum(reward)

        self.store_sample(s, a, r, s_n, done)
        self.update_network()
        return 0

    def store_sample(self, s, a, r, s_n, done):
        self.replay_buffer.add_to_memory((s, a, r, s_n, done))
        return 0

    def update_network(self):
        self.update_cnt += 1
        if len(self.replay_buffer.replay_memory) < FLAGS.m_size * FLAGS.pre_train_step:
            return 0

        minibatch = self.replay_buffer.sample_from_memory()
        self.q_network.training_qnet(minibatch)

        if self.update_cnt % self.target_update_period == 0:
            self.q_network.training_target_qnet()
            if FLAGS.qtrace:
                if self.update_cnt % 10000 == 0:
                    self.q_diff()

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
        onehot = np.zeros([self._n_player, self._action_dim_single])
        for i in range(self._n_player):
            onehot[i, action[i]] = 1
        return onehot

    def q_diff(self):


        print self.update_cnt

    def q(self, mode, step =0):

        if mode == 0:
            q_value = np.zeros([2,2])
            for i in range(2):
                for j in range(2):
                    s = np.array([10])
                    action = [i,j]
                    a = self.action_to_onehot(action)
                    q_value[i,j] = self.q_network.get_q_values(s[None],a[None])[0]


            return q_value#, qi_value, p_value

        if mode == 1:
            q_value = np.zeros([3,3])
            for i in range(3):
                for j in range(3):
                    s = np.array([1])
                    action = [i,j]
                    a = self.action_to_onehot(action)
                    q_value[i,j] = self.q_network.get_q_values(s[None],a[None])[0]

            qi_value = self.q_network.get_qp_values(s[None])[0] 

            q_value2 = np.zeros([3,3])
            for i in range(3):
                for j in range(3):
                    s = np.array([2])
                    action = [i,j]
                    a = self.action_to_onehot(action)
                    q_value2[i,j] = self.q_network.get_q_values(s[None],a[None])[0]

            qi_value2 = self.q_network.get_qp_values(s[None])[0]



            return q_value, qi_value, q_value2, qi_value2

        if mode == 2:
            q_value = np.zeros([11])
            for i in range(11):
                s = np.ones(10) * 0.5
                a = np.zeros([10,2])
                a[:10-i,0] = 1.
                a[10-i:,1] = 1.
                q_value[i] = self.q_network.get_q_values(s[None],a[None])[0]
            
            return q_value


        elif mode == 3:
            q_value_1 = np.zeros([2,2])
            q_value_2 = np.zeros([2,2])
            q_value_3 = np.zeros([2,2])
            for i in range(2):
                for j in range(2):
                    s = np.array([0])
                    action = [i,j]
                    a = self.action_to_onehot(action)
                    q_value_1[i,j] = self.q_network.get_q_values(s[None],a[None])[0]

            q_value_2 = np.zeros([2,2])
            for i in range(2):
                for j in range(2):
                    s = np.array([1])
                    action = [i,j]
                    a = self.action_to_onehot(action)
                    q_value_2[i,j] = self.q_network.get_q_values(s[None],a[None])[0]
            
            q_value_3 = np.zeros([2,2])
            for i in range(2):
                for j in range(2):
                    s = np.array([2])
                    action = [i,j]
                    a = self.action_to_onehot(action)
                    q_value_3[i,j] = self.q_network.get_q_values(s[None],a[None])[0]

            return q_value_1, q_value_2, q_value_3

        
        elif mode == 4:
            samples = 1000
            x = np.zeros(samples)
            y = np.zeros(samples)
            z = np.zeros(samples)
            for i in range (samples):
                act_n = []
                for j in range(self._n_player):
                    action = np.random.choice(self._action_dim_single)
                    act_n.append(action)
                # act_n = np.array(list(bin(int(i))[2:].zfill(8)),dtype='int')
                a = self.action_to_onehot(act_n)
                s = np.ones(self._n_player) * 0.1
                x[i] = np.sum(np.array(act_n) * s)
                y[i] = self.q_network.get_q_values(s[None],a[None])[0]
                z[i] = self.q_network.get_qp_values(s[None],a[None])[0]
            
            order = np.argsort(x)
            xs = np.array(x)[order]
            ys = np.array(y)[order]
            zs = np.array(z)[order]

            np.save(config.file_name + "1", xs)
            np.save(config.file_name + "2", ys)
            np.save(config.file_name + "3", zs)

            plt.scatter(xs, ys)
            plt.xlim(0, 2.5)
            plt.ylim(0, 18)
            plt.xlabel('State-Action Fair')
            plt.ylabel('Q-value')
            plt.savefig(config.file_name + '-A1.png')

            plt.clf()

            plt.plot(xs, ys)
            plt.xlim(0, 2.5)
            plt.ylim(0, 18)
            plt.xlabel('State-Action Fair')
            plt.ylabel('Q-value')
            plt.savefig(config.file_name + '-A2.png')

            plt.clf()

            plt.scatter(xs, zs)
            plt.xlim(0, 2.5)
            plt.ylim(0, 18)
            plt.xlabel('State-Action Fair')
            plt.ylabel('Q-value')
            plt.savefig(config.file_name + '-B1.png')

            plt.clf()

            plt.plot(xs, zs)
            plt.xlim(0, 2.5)
            plt.ylim(0, 18)
            plt.xlabel('State-Action Fair')
            plt.ylabel('Q-value')
            plt.savefig(config.file_name + '-B2.png')

            return "FINISH"

        elif mode == 5:
            s = np.array([1])
            Q_matrix = np.zeros([21,21])
            Q_matrix2 = np.zeros([21,21])
            for i in range(21):
                for j in range(21):
                    act_n = np.array([i,j])
                    a = self.action_to_onehot(act_n)
                    Q_matrix[i,j] = np.mean(self.q_network.get_q_values(s[None],a[None]))
                    Q_matrix2[i,j] = np.mean(self.q_network.get_qp_values(s[None],a[None]))
            optimal_action = self.q_network.get_action(s[None])[0]

            ind = np.unravel_index(np.argmax(Q_matrix, axis=None), Q_matrix.shape)
            print 'optimal_action', optimal_action, np.mean(self.q_network.get_q_values(s[None],self.action_to_onehot(optimal_action)[None]))
            print 'ind', ind, self.q_network.get_q_values(s[None],self.action_to_onehot(ind)[None])[0]

            # plt.clf()
            # self.fig = plt.figure(figsize=(4,4))
            # self.ims = []
            title = self.axes.text(0.5,1.05,"Step {}".format(step), 
                    size=plt.rcParams["axes.titlesize"],
                    ha="center", transform=self.axes.transAxes, )
            print "ADD!"
            self.ims.append([plt.pcolor(Q_matrix2,vmin=-10, vmax=10), title])



            return Q_matrix, Q_matrix2

        elif mode == 6:
            im_ani = animation.ArtistAnimation(self.fig, self.ims, interval=200, #repeat_delay=3000,
                                   blit=False)
            im_ani.save(str(FLAGS.algorithm)+'.gif', dpi=80, writer='imagemagick')
            return True


        elif mode == 5:
            
            s = np.array([1] * FLAGS.n_predator)
            optimal_action = self.q_network.get_action(s[None])[0]
            r = np.sum(np.array(optimal_action))/10.

            x = np.linspace(0, 10, 1000)
            y = np.array([r * np.exp( -np.square(r-5)/1) + r * np.exp(-np.square(r-8)/0.25) for r in x])
            # a = np.load("1-" + "pqmix5" + "-" + str((i+1)*1000)+".npy")
            x2 = np.sum(np.array(optimal_action))/10.
            y2 = x2 * np.exp( -np.square(x2-5)/1) + x2 * np.exp(-np.square(x2-8)/0.25)
            im = plt.plot(x,y,'black')
            im2 = plt.plot([x2],[y2],'ro')
            title = self.axes.text(0.5,1.05,"Step {}".format(step), 
                    size=plt.rcParams["axes.titlesize"],
                    ha="center", transform=self.axes.transAxes, )
            self.ims.append(im2)
            


            s = np.array([1])
            Q_matrix = np.zeros([21,21])
            Q_matrix2 = np.zeros([21,21])
            for i in range(21):
                for j in range(21):
                    act_n = np.array([i,j])
                    a = self.action_to_onehot(act_n)
                    Q_matrix[i,j] = np.mean(self.q_network.get_q_values(s[None],a[None]))
                    Q_matrix2[i,j] = np.mean(self.q_network.get_qp_values(s[None],a[None]))
            optimal_action = self.q_network.get_action(s[None])[0]

            ind = np.unravel_index(np.argmax(Q_matrix, axis=None), Q_matrix.shape)
            print 'optimal_action', optimal_action, np.mean(self.q_network.get_q_values(s[None],self.action_to_onehot(optimal_action)[None]))
            print 'ind', ind, self.q_network.get_q_values(s[None],self.action_to_onehot(ind)[None])[0]

            # plt.clf()
            # self.fig = plt.figure(figsize=(4,4))
            # self.ims = []
            title = self.axes.text(0.5,1.05,"Step {}".format(step), 
                    size=plt.rcParams["axes.titlesize"],
                    ha="center", transform=self.axes.transAxes, )
            print "ADD!"
            self.ims.append([plt.pcolor(Q_matrix2,vmin=-10, vmax=10), title])





