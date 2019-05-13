import numpy as np
import tensorflow as tf
from collections import deque
import random
import config
import os
from agents.evaluation import Evaluation

FLAGS = config.flags.FLAGS

gamma = FLAGS.df  # reward discount factor
learning_rate = FLAGS.lr
h1 = 64
h2 = 64
h3 = 64
h4 = 64
h5 = 64
tau = 0.999

option_V = True # Value Network
option_W = False # Weight
# if FLAGS.algorithm == "vdn" or FLAGS.algorithm == "qmix":
#     option_O = True
# else:
option_O = False
# option_O = True # Double Q-learning
option_D = False

option_F1 = True

option_reg = False
option_SG = True

r_bonus = 0.0

replay_memory_capacity = 50000  # capacity of experience replay memory
minibatch_size = FLAGS.m_size  # size of minibatch from experience replay memory for updates

class DQNetwork(object):
    def __init__(self, sess, state_dim, action_dim_single, n_predator):
        self.sess = sess
        self.state_dim = state_dim
        self.action_dim_single = action_dim_single
        self.n_predator = n_predator
        self.action_dim = action_dim_single * n_predator
        # placeholders
        self.s_in = tf.placeholder(dtype=tf.float32, shape=[None, self.state_dim])
        self.a_in = tf.placeholder(dtype=tf.float32, shape=[None, self.n_predator, self.action_dim_single])
        self.y_in = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        # self.y_in = tf.placeholder(dtype=tf.float32, shape=[None, self.n_predator])
        self.p_in = tf.placeholder(dtype=tf.float32, shape=[None, self.n_predator, self.action_dim_single])
        self.beta_in = tf.placeholder(dtype=tf.float32, shape=None)
        self.r_in = tf.placeholder(dtype = tf.float32, shape=None)
        self.meanq_in = tf.placeholder(dtype = tf.float32, shape=None)
        self.alpha = -2.0
        self.beta = 1.0
        # indicators (go into target computation)
        self.is_training_ph = tf.placeholder(dtype=tf.bool, shape=())  # for dropout

        with tf.variable_scope('q_network'):
            if FLAGS.algorithm == "vdn":
                self.q_network, self.actor_network, self.v, self.qp  = self.generate_VDN(self.s_in, self.a_in, True)
                with tf.device('/cpu:0'): 
                    self.cpu_q_network, self.cpu_actor_network, self.cpu_v, self.cpu_qp  = self.generate_VDN(self.s_in, self.a_in, True)
            elif FLAGS.algorithm == "qmix":
                self.q_network, self.actor_network, self.v, self.qp  = self.generate_QMIX(self.s_in, self.a_in, True)
                with tf.device('/cpu:0'): 
                    self.cpu_q_network, self.cpu_actor_network, self.cpu_v, self.cpu_qp  = self.generate_QMIX(self.s_in, self.a_in, True)
            elif FLAGS.algorithm == "pqmix":
                self.q_network, self.actor_network, self.v, self.p, self.qp  = self.generate_PQMIX(self.s_in, self.a_in, self.p_in, True)
                with tf.device('/cpu:0'): 
                    self.cpu_q_network, self.cpu_actor_network, self.cpu_v, self.cpu_p, self.cpu_qp  =  self.generate_PQMIX(self.s_in, self.a_in, self.p_in, True)
            elif FLAGS.algorithm == "pqmix2":
                self.q_network, self.actor_network, self.v, self.p, self.qp  = self.generate_PQMIX2(self.s_in, self.a_in, True)
                with tf.device('/cpu:0'): 
                    self.cpu_q_network, self.cpu_actor_network, self.cpu_v, self.cpu_p, self.cpu_qp  = self.generate_PQMIX2(self.s_in, self.a_in, True)
            elif FLAGS.algorithm == "pqmix3":
                self.q_network, self.actor_network, self.v, self.p, self.qp  = self.generate_PQMIX3(self.s_in, self.a_in, True)
                with tf.device('/cpu:0'): 
                    self.cpu_q_network, self.cpu_actor_network, self.cpu_v, self.cpu_p, self.cpu_qp  = self.generate_PQMIX3(self.s_in, self.a_in, True)
            elif FLAGS.algorithm == "pqmix4":
                self.q_network, self.actor_network, self.v, self.p, self.qp  = self.generate_PQMIX4(self.s_in, self.a_in, True)
                with tf.device('/cpu:0'): 
                    self.cpu_q_network, self.cpu_actor_network, self.cpu_v, self.cpu_p, self.cpu_qp  = self.generate_PQMIX4(self.s_in, self.a_in, True)
            # elif FLAGS.algorithm == "pqmix5":
            #     self.q_network, self.actor_network, self.v, self.p, self.qp  = self.generate_PQMIX5(self.s_in, self.a_in, True)
            #     with tf.device('/cpu:0'): 
            #         self.cpu_q_network, self.cpu_actor_network, self.cpu_v, self.cpu_p, self.cpu_qp  = self.generate_PQMIX5(self.s_in, self.a_in, True)
            elif FLAGS.algorithm == "pqmix5": #QTRAN-alt
                self.q_network, self.actor_network, self.v, self.p, self.qp, self.d1, self.d2, self.d3  = self.generate_PQMIX5(self.s_in, self.a_in, True)
                with tf.device('/cpu:0'): 
                    self.cpu_q_network, self.cpu_actor_network, self.cpu_v, self.cpu_p, self.cpu_qp, self.cd1, self.cd2, self.cd3  = self.generate_PQMIX5(self.s_in, self.a_in, True)
            elif FLAGS.algorithm == "pqmix6":
                self.q_network, self.actor_network, self.v, self.qp  = self.generate_PQMIX6(self.s_in, self.a_in, True)
                with tf.device('/cpu:0'): 
                    self.cpu_q_network, self.cpu_actor_network, self.cpu_v, self.cpu_qp  = self.generate_PQMIX6(self.s_in, self.a_in, True)
            elif FLAGS.algorithm == "pqmix7": #QTRAN-base
                self.q_network, self.actor_network, self.v, self.p, self.qp  = self.generate_PQMIX7(self.s_in, self.a_in, True)
                with tf.device('/cpu:0'): 
                    self.cpu_q_network, self.cpu_actor_network, self.cpu_v, self.cpu_p, self.cpu_qp  = self.generate_PQMIX7(self.s_in, self.a_in, True)

        with tf.variable_scope('target_q_network'):
            if FLAGS.algorithm == "vdn":
                self.target_q_network, self.target_actor_network, self.tv, self.tqp = self.generate_VDN(self.s_in, self.a_in, False)
                with tf.device('/cpu:0'): 
                    self.cpu_target_q_network, self.cpu_target_actor_network, self.cpu_tv, self.cpu_tqp = self.generate_VDN(self.s_in, self.a_in, False)
            elif FLAGS.algorithm == "qmix":
                self.target_q_network, self.target_actor_network, self.tv, self.tqp = self.generate_QMIX(self.s_in, self.a_in, False)
                with tf.device('/cpu:0'): 
                    self.cpu_target_q_network, self.cpu_target_actor_network, self.cpu_tv, self.cpu_tqp = self.generate_QMIX(self.s_in, self.a_in, False)
            elif FLAGS.algorithm == "pqmix":
                self.target_q_network, self.target_actor_network, self.tv, self.tp, self.tqp = self.generate_PQMIX(self.s_in, self.a_in, self.p_in, False)
                with tf.device('/cpu:0'): 
                    self.cpu_target_q_network, self.cpu_target_actor_network, self.cpu_tv, self.cpu_tp, self.cpu_tqp = self.generate_PQMIX(self.s_in, self.a_in, False)
            elif FLAGS.algorithm == "pqmix2":
                self.target_q_network, self.target_actor_network, self.tv, self.tp, self.tqp = self.generate_PQMIX2(self.s_in, self.a_in, False)
                with tf.device('/cpu:0'): 
                    self.cpu_target_q_network, self.cpu_target_actor_network, self.cpu_tv, self.cpu_tp, self.cpu_tqp = self.generate_PQMIX2(self.s_in, self.a_in, False)
            elif FLAGS.algorithm == "pqmix3":
                self.target_q_network, self.target_actor_network, self.tv, self.tp, self.tqp = self.generate_PQMIX3(self.s_in, self.a_in, False)
                with tf.device('/cpu:0'): 
                    self.cpu_target_q_network, self.cpu_target_actor_network, self.cpu_tv, self.cpu_tp, self.cpu_tqp = self.generate_PQMIX3(self.s_in, self.a_in, False)
            elif FLAGS.algorithm == "pqmix4":
                self.target_q_network, self.target_actor_network, self.tv, self.tp, self.tqp = self.generate_PQMIX4(self.s_in, self.a_in, False)
                with tf.device('/cpu:0'): 
                    self.cpu_target_q_network, self.cpu_target_actor_network, self.cpu_tv, self.cpu_tp, self.cpu_tqp = self.generate_PQMIX4(self.s_in, self.a_in, False)
            # elif FLAGS.algorithm == "pqmix5":
            #     self.target_q_network, self.target_actor_network, self.tv, self.tp, self.tqp = self.generate_PQMIX5(self.s_in, self.a_in, False)
            #     with tf.device('/cpu:0'): 
            #         self.cpu_target_q_network, self.cpu_target_actor_network, self.cpu_tv, self.cpu_tp, self.cpu_tqp = self.generate_PQMIX5(self.s_in, self.a_in, False)
            elif FLAGS.algorithm == "pqmix5":
                self.target_q_network, self.target_actor_network, self.tv, self.tp, self.tqp, self.td1, self.td2, self.td3 = self.generate_PQMIX5(self.s_in, self.a_in, False)
                with tf.device('/cpu:0'): 
                    self.cpu_target_q_network, self.cpu_target_actor_network, self.cpu_tv, self.cpu_tp, self.cpu_tqp, self.ctd1, self.ctd2, self.ctd3 = self.generate_PQMIX5(self.s_in, self.a_in, False)
            elif FLAGS.algorithm == "pqmix6":
                self.target_q_network, self.target_actor_network, self.tv, self.tqp = self.generate_PQMIX6(self.s_in, self.a_in, False)
                with tf.device('/cpu:0'): 
                    self.cpu_target_q_network, self.cpu_target_actor_network, self.cpu_tv, self.cpu_tqp = self.generate_PQMIX6(self.s_in, self.a_in, False)
            elif FLAGS.algorithm == "pqmix7":
                self.target_q_network, self.target_actor_network, self.tv, self.tp, self.tqp = self.generate_PQMIX7(self.s_in, self.a_in, False)
                with tf.device('/cpu:0'): 
                    self.cpu_target_q_network, self.cpu_target_actor_network, self.cpu_tv, self.cpu_tp, self.cpu_tqp = self.generate_PQMIX7(self.s_in, self.a_in, False)
        
        self.action_onehot = tf.one_hot(self.actor_network, self.action_dim_single, on_value=1.0, off_value=0.0, axis=-1)
        self.target_action_onehot = tf.one_hot(self.target_actor_network, self.action_dim_single, on_value=1.0, off_value=0.0, axis=-1)
        with tf.variable_scope('optimization'):
            
            self.delta = self.y_in - self.q_network

            self.clipped_error = tf.square(self.delta)

            
            self.cost = tf.reduce_mean(self.clipped_error) 
            self.meanq = tf.reduce_mean(self.q_network)
            self.meanq_history = tf.summary.scalar('mean_Q-' + str(self.n_predator) + '-' + str(FLAGS.penalty), self.meanq_in)
            self.reward_history = tf.summary.scalar('mean_reward-' + str(self.n_predator) + '-' + str(FLAGS.penalty), self.r_in)
            self.merged = tf.summary.merge_all()

            if FLAGS.algorithm == "pqmix3" or FLAGS.algorithm == "pqmix4" or FLAGS.algorithm == "pqmix5" or FLAGS.algorithm == "pqmix7":
                
                self.clipped_error = tf.square(self.delta)

                self.cost1 = tf.reduce_mean(self.clipped_error)      
                self.clipped_p = tf.abs(self.p)
                self.cost2 = tf.reduce_mean(self.clipped_p)
                self.cost = 1 * self.cost1 + (self.beta_in) * self.cost2 # pqmix, 1, 0.01 0.001


            elif FLAGS.algorithm == "pqmix6":

                flag = 1 + tf.reduce_min(tf.layers.flatten(self.a_in) - tf.layers.flatten(self.action_onehot), reduction_indices=1, keep_dims=True)

                self.delta = tf.where(self.delta > 0.0,
                                    1.0 * self.delta,
                                    1.0 * self.delta, name='delta2')
                self.clipped_error2 = tf.square(self.delta)
                self.cost1 = tf.reduce_mean((flag + 0.0) *self.clipped_error)
                self.cost2 = tf.reduce_mean((1.0-flag) *self.clipped_error2)
                self.cost =  self.cost1 +  self.cost2


            self.train_network = tf.train.AdamOptimizer(learning_rate, epsilon = 1.5e-4).minimize(self.cost)

        o_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q_network')
        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_q_network')
        self.update_target_network = [tf.assign(t, o) for o, t in zip(o_params, t_params)]

        self.cnt = 0
        self.TDerror = 0
        self.TDerror2 = 0
        self.TDerror3 = 0
        self.meanq_value = 0
        self.meanq_counter = 0
        self._eval = Evaluation()

    def generate_single_q_network(self, obs_single, action_single, trainable=True):

        hidden_1 = tf.layers.dense(obs_single, h1, activation=tf.nn.relu,
                                   kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                                   bias_initializer=tf.constant_initializer(0.00),  # biases
                                   use_bias=True,
                                   trainable=trainable, reuse=tf.AUTO_REUSE, name='dense_h1')
        hidden_2 = tf.layers.dense(hidden_1, h2, activation=tf.nn.relu,
                                   kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                                   bias_initializer=tf.constant_initializer(0.00),  # biases
                                   use_bias=True,
                                   trainable=trainable, reuse=tf.AUTO_REUSE, name='dense_h2')
        hidden_3 = tf.layers.dense(hidden_2, h3, activation=tf.nn.relu,
                                   kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                                   bias_initializer=tf.constant_initializer(0.00),  # biases
                                   use_bias=True,
                                   trainable=trainable, reuse=tf.AUTO_REUSE, name='dense_h3')
        q_values = tf.layers.dense(hidden_3, self.action_dim_single,
                                    kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                                   bias_initializer=tf.constant_initializer(0.00),  # biases
                                   use_bias=True,
                                   trainable=trainable, reuse=tf.AUTO_REUSE, name='dense_h4')#, reuse=tf.AUTO_REUSE, name='dense_h4')

        optimal_action = tf.expand_dims(tf.argmax(q_values, 1),-1)
        q = tf.reduce_sum(q_values * action_single, axis=1, keep_dims=True)
        qmax = tf.reduce_max(q_values, axis=1, keep_dims=True)


        return q, optimal_action, qmax, q_values

    def generate_single_q_network_prime(self, obs_single, action_single, i, trainable=True):

        hidden_1 = tf.layers.dense(obs_single, h1, activation=tf.nn.relu,
                                   kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                                   bias_initializer=tf.constant_initializer(0.00),  # biases
                                   use_bias=True,
                                   trainable=trainable, reuse=tf.AUTO_REUSE, name='dense_h1')
        hidden_2 = tf.layers.dense(hidden_1, h2, activation=tf.nn.relu,
                                   kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                                   bias_initializer=tf.constant_initializer(0.00),  # biases
                                   use_bias=True,
                                   trainable=trainable, reuse=tf.AUTO_REUSE, name='dense_h2')
        hidden_3 = tf.layers.dense(hidden_2, h3, activation=tf.nn.relu,
                                   kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                                   bias_initializer=tf.constant_initializer(0.00),  # biases
                                   use_bias=True,
                                   trainable=trainable, reuse=tf.AUTO_REUSE, name='dense_h3')
        q_values = tf.layers.dense(hidden_3, self.action_dim_single,
                                    kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                                   bias_initializer=tf.constant_initializer(0.00),  # biases
                                   use_bias=True,
                                   trainable=trainable, reuse=tf.AUTO_REUSE, name='dense_h4')#, reuse=tf.AUTO_REUSE, name='dense_h4')

        optimal_action = tf.expand_dims(tf.argmax(q_values, 1),-1)
        q = tf.reduce_sum(q_values * action_single, axis=1, keep_dims=True)
        qmax = tf.reduce_max(q_values, axis=1, keep_dims=True)  
        
        optimal_action_onehot = tf.one_hot(tf.argmax(q_values, 1), self.action_dim_single, on_value=1.0, off_value=0.0, axis=-1)

        if option_D == True:
            q_values_softmax = tf.nn.softmax(q_values, dim=-1)
            #print q_values_softmax
            q_softmax = tf.reduce_sum(q_values_softmax * action_single, axis=1, keep_dims=True)
            # print concat_sa
            concat_sa = tf.concat([hidden_2, action_single], 1)
            concat_sa2 = tf.concat([hidden_2, q_values_softmax], 1)

        else:

            concat_sa = tf.concat([hidden_2, action_single], 1)
            concat_sa2 = tf.concat([hidden_2,optimal_action_onehot], 1)


        hidden_key1 = tf.layers.dense(concat_sa, h4, activation=tf.nn.relu,
                                   kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                                   bias_initializer=tf.constant_initializer(0.00),  # biases
                                   use_bias=True,
                                   trainable=trainable, reuse=tf.AUTO_REUSE, name='dense_k1')
        key1 = tf.layers.dense(hidden_key1 , h5, 
                                   kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                                   bias_initializer=tf.constant_initializer(0.00),  # biases
                                   use_bias=True,
                                   trainable=trainable, reuse=tf.AUTO_REUSE, name='dense_k2')
        hidden_key1_t = tf.layers.dense(concat_sa2, h4, activation=tf.nn.relu,
                                   kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                                   bias_initializer=tf.constant_initializer(0.00),  # biases
                                   use_bias=True,
                                   trainable=trainable, reuse=tf.AUTO_REUSE, name='dense_k1')
        key1_t = tf.layers.dense(hidden_key1_t , h5, 
                                   kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                                   bias_initializer=tf.constant_initializer(0.00),  # biases
                                   use_bias=True,
                                   trainable=trainable, reuse=tf.AUTO_REUSE, name='dense_k2')
        
        hidden_key2 = tf.layers.dense(hidden_2, h4, activation=tf.nn.relu,
                                   kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                                   bias_initializer=tf.constant_initializer(0.00),  # biases
                                   use_bias=True,
                                   trainable=trainable, reuse=tf.AUTO_REUSE, name='dense_k3')
        key2 = tf.layers.dense(hidden_key2, h5, 
                                   kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                                   bias_initializer=tf.constant_initializer(0.00),  # biases
                                   use_bias=True,
                                   trainable=trainable, reuse=tf.AUTO_REUSE, name='dense_k4')

            
        return q, optimal_action, qmax, q_values, key1, key1_t, key2#, hidden_key1, hidden_key1_t
        
    def generate_attention_network(self, obs_single, action_single, q_values, trainable=True):

        concat_sa = tf.concat([obs_single, action_single] , 1)
        concat_sq = obs_single
        query = tf.layers.dense(concat_sq, h5, #activation=tf.nn.elu,
                                   kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                                   bias_initializer=tf.constant_initializer(0.00),  # biases
                                   use_bias=True,
                                   trainable=trainable, reuse=tf.AUTO_REUSE, name='dense_aq1')
        key = tf.layers.dense(concat_sa, h5, #activation=tf.nn.elu,
                                   kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                                   bias_initializer=tf.constant_initializer(0.00),  # biases
                                   use_bias=True,
                                   trainable=trainable, reuse=tf.AUTO_REUSE, name='dense_ak1')
        val = tf.layers.dense(concat_sa, h5, #activation=tf.nn.elu,
                                   kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                                   bias_initializer=tf.constant_initializer(0.00),  # biases
                                   use_bias=True,
                                   trainable=trainable, reuse=tf.AUTO_REUSE, name='dense_av1')

        val2 = tf.layers.dense(concat_sq, h5, #activation=tf.nn.elu,
                                   kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                                   bias_initializer=tf.constant_initializer(0.00),  # biases
                                   use_bias=True,
                                   trainable=trainable, reuse=tf.AUTO_REUSE, name='dense_av2')


        return query, key, val, val2

    def generate_self_attention_network(self, Q, K, V, V2, action, trainable=True):
        
        attention = tf.matmul(Q, K, transpose_b=True)

        d_k = tf.cast(tf.shape(K)[-1], dtype=tf.float32)
        attention = tf.divide(attention, tf.sqrt(d_k)) # [batch_size, sequence_length, sequence_length]

        attention = 5 * tf.nn.tanh(attention)

        # print attention
        ones = tf.ones([FLAGS.m_size, FLAGS.n_predator])
        attention = tf.matrix_set_diag(attention, -10 * ones, name=None)

        attention = tf.nn.softmax(attention, dim=-1)


        output = V2 + tf.matmul(attention, V)


        F = tf.layers.dense(output, self.action_dim_single, #activation=tf.nn.softplus,
                                   kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                                   bias_initializer=tf.constant_initializer(0.00),  # biases
                                   use_bias=True,
                                   trainable=trainable, reuse=tf.AUTO_REUSE, name='dense_af')

        if option_F1 == True:
            F = F - tf.reduce_min(F, reduction_indices = 2, keep_dims = True)
        else:
            F = tf.abs(F)
        selected_F = tf.reduce_sum(F * action, reduction_indices=2, keep_dims = False)

        return F, selected_F

    def generate_regularization_network(self, key1, key2, action, trainable = True):
        
        test = tf.reduce_mean(key1, reduction_indices=1, keep_dims = True) - key1 / self.n_predator
        vector = key2 + tf.reduce_mean(key1, reduction_indices=1, keep_dims = True) - key1 / self.n_predator

        reg_hidden_1 = tf.layers.dense(vector, h1, activation=tf.nn.relu,
                                   kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                                   bias_initializer=tf.constant_initializer(0.00),  # biases
                                   use_bias=True,
                                   trainable=trainable, reuse=tf.AUTO_REUSE, name='dense_reg1')
        reg_hidden_2 = tf.layers.dense(reg_hidden_1, h2, activation=tf.nn.relu,
                                   kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                                   bias_initializer=tf.constant_initializer(0.00),  # biases
                                   use_bias=True,
                                   trainable=trainable, reuse=tf.AUTO_REUSE, name='dense_reg2')
        F = tf.layers.dense(reg_hidden_2, self.action_dim_single, #activation=tf.nn.softplus,
                                   kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                                   bias_initializer=tf.constant_initializer(0.00),  # biases
                                   use_bias=True,
                                   trainable=trainable, reuse=tf.AUTO_REUSE, name='dense_reg3')
        
        F = F - tf.reduce_min(F, reduction_indices = 2, keep_dims = True)

        selected_F = tf.reduce_sum(F * action, reduction_indices=2, keep_dims = False)

        return F, selected_F, vector, test

    def generate_regularization_network2(self, key1, key2, action, trainable = True):
        
        test = tf.reduce_mean(key1, reduction_indices=1, keep_dims = True) - key1 / self.n_predator
        vector = key2 + tf.reduce_mean(key1, reduction_indices=1, keep_dims = True) - key1 / self.n_predator

        reg_hidden_1 = tf.layers.dense(vector, h1, activation=tf.nn.relu,
                                   kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                                   bias_initializer=tf.constant_initializer(0.00),  # biases
                                   use_bias=True,
                                   trainable=trainable, reuse=tf.AUTO_REUSE, name='dense_reg1')
        reg_hidden_2 = tf.layers.dense(reg_hidden_1, h2, activation=tf.nn.relu,
                                   kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                                   bias_initializer=tf.constant_initializer(0.00),  # biases
                                   use_bias=True,
                                   trainable=trainable, reuse=tf.AUTO_REUSE, name='dense_reg2')
        Q = tf.layers.dense(reg_hidden_2, self.action_dim_single, #activation=tf.nn.softplus,
                                   kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                                   bias_initializer=tf.constant_initializer(0.00),  # biases
                                   use_bias=True,
                                   trainable=trainable, reuse=tf.AUTO_REUSE, name='dense_reg3')
        selected_Q = tf.reduce_sum(Q * action, reduction_indices=2, keep_dims = False)


        return Q, selected_Q, vector, test
    
    
    def generate_VDN(self, s, action, trainable=True):
        q_list = list()
        q_optimal_list = list()
        q_values_list = list()
        action_list = list()


        for i in range(self.n_predator):
            obs = tf.reshape(s, shape=[-1, self.state_dim/2, 2])
            loc = tf.reshape(obs[:,i,:], shape = [-1, 1, 2])
            id = tf.reshape(obs[:,0,0], shape = [-1, 1])*0 + float(i)/self.n_predator
            obs = obs - loc
            obs = tf.reshape(obs, shape=[-1, self.state_dim])
            # obs = tf.where(tf.abs(obs) < 2.5 / FLAGS.map_size,
            #                         obs,
            #                         obs*0, name='clipped_error')
            obs_n = tf.concat([obs, id, tf.reshape(loc, shape = [-1, 2])],1)
            # obs_n = tf.concat([obs, tf.reshape(loc, shape = [-1, 2])],1)
            action_single = action[:,i,:]
            q_single, optimal_action, qmax, q_values_single = self.generate_single_q_network(obs_n, action_single, trainable)

            q_list.append(q_single)
            action_list.append(optimal_action)
            q_optimal_list.append(qmax)
            q_values_list.append(q_values_single)


        q_values = tf.concat(q_list, axis=1)
        qmax_values = tf.concat(q_optimal_list, axis=1)
        optimal_action = tf.concat(action_list, axis=1)
        optimal_action_onehot = tf.one_hot(optimal_action, self.action_dim_single, on_value=1.0, off_value=0.0, axis=-1)

        q_value = tf.reduce_sum(q_values, reduction_indices=1, keep_dims = True)
        value = tf.reduce_sum(qmax_values, reduction_indices=1, keep_dims = True)

        if option_O == True:
            value = q_value
        
        return q_value, optimal_action, value, q_value

    def generate_QMIX(self, s, action, trainable=True):
        q_list = list()
        q_optimal_list = list()
        q_values_list = list()
        action_list = list()


        for i in range(self.n_predator):
            obs = tf.reshape(s, shape=[-1, self.state_dim/2, 2])
            loc = tf.reshape(obs[:,i,:], shape = [-1, 1, 2])
            id = tf.reshape(obs[:,0,0], shape = [-1, 1])*0 + float(i)/self.n_predator
            obs = obs - loc
            obs = tf.reshape(obs, shape=[-1, self.state_dim])
            # obs = tf.where(tf.abs(obs) < 2.5 / FLAGS.map_size,
            #                         obs,
            #                         obs*0, name='clipped_error')
            obs_n = tf.concat([obs, id, tf.reshape(loc, shape = [-1, 2])],1)
            # obs_n = tf.concat([obs, tf.reshape(loc, shape = [-1, 2])],1)
            action_single = action[:,i,:]
            q_single, optimal_action, qmax, q_values_single = self.generate_single_q_network(obs_n, action_single, trainable)

            q_list.append(q_single)
            action_list.append(optimal_action)
            q_optimal_list.append(qmax)
            q_values_list.append(q_values_single)


        q_values = tf.concat(q_list, axis=1)
        qmax_values = tf.concat(q_optimal_list, axis=1)
        optimal_action = tf.concat(action_list, axis=1)
        optimal_action_onehot = tf.one_hot(optimal_action, self.action_dim_single, on_value=1.0, off_value=0.0, axis=-1)

        s2 = s #* 0 + 1

        w1 = tf.layers.dense(s2, self.n_predator * h3,
                                   kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                                   bias_initializer=tf.constant_initializer(0.00),  # biases
                                   use_bias=True,
                                   trainable=trainable, reuse=tf.AUTO_REUSE, name='dense_w1')

        w2 = tf.layers.dense(s2, h3 * 1, 
                                   kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                                   bias_initializer=tf.constant_initializer(0.00),  # biases
                                   use_bias=True,
                                   trainable=trainable, reuse=tf.AUTO_REUSE, name='dense_w2')

        b1 = tf.layers.dense(s2, h3,
                                   kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                                   bias_initializer=tf.constant_initializer(0.00),  # biases
                                   use_bias=True,
                                   trainable=trainable, reuse=tf.AUTO_REUSE, name='dense_b1')
        b2_h = tf.layers.dense(s2, h3,  activation=tf.nn.relu,
                                   kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                                   bias_initializer=tf.constant_initializer(0.00),  # biases
                                   use_bias=True,
                                   trainable=trainable, reuse=tf.AUTO_REUSE, name='dense_b2_h')
        b2 = tf.layers.dense(b2_h, 1, 
                                   kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                                   bias_initializer=tf.constant_initializer(0.00),  # biases
                                   use_bias=True,
                                   trainable=trainable, reuse=tf.AUTO_REUSE, name='dense_b2')
        w1 = tf.abs(w1)
        w1 = tf.reshape(w1, [-1, self.n_predator, h3])
        w2 = tf.abs(w2)
        w2 = tf.reshape(w2, [-1, h3, 1])
        q_values = tf.reshape(q_values, [-1,1,self.n_predator])
        q_hidden = tf.nn.elu(tf.reshape(tf.matmul(q_values, w1),[-1,h3]) ) +b1

        q_hidden = tf.reshape(q_hidden, [-1,1,h3])
        q_value = tf.reshape(tf.matmul(q_hidden, w2),[-1,1]) + b2

        qmax_values = tf.reshape(qmax_values, [-1,1,self.n_predator])
        qmax_hidden = tf.nn.elu(tf.reshape(tf.matmul(qmax_values, w1),[-1,h3]) ) +b1

        qmax_hidden = tf.reshape(qmax_hidden, [-1,1,h3])
        value = tf.reshape(tf.matmul(qmax_hidden, w2),[-1,1]) + b2

        if option_O == True:
            value = q_value

        
        return q_value, optimal_action, value, q_value

    def generate_PQMIX(self, s, action, p, trainable=True):
        q_list = list()
        q_optimal_list = list()
        q_values_list = list()
        action_list = list()


        for i in range(self.n_predator):
            obs = tf.reshape(s, shape=[-1, self.state_dim/2, 2])
            loc = tf.reshape(obs[:,i,:], shape = [-1, 1, 2])
            id = tf.reshape(obs[:,0,0], shape = [-1, 1])*0 + float(i)/self.n_predator
            obs = obs - loc
            obs = tf.reshape(obs, shape=[-1, self.state_dim])
            # obs = tf.where(tf.abs(obs) < 2.5 / FLAGS.map_size,
            #                         obs,
            #                         obs*0, name='clipped_error')
            obs_n = tf.concat([obs, id, tf.reshape(loc, shape = [-1, 2])],1)
            # obs_n = tf.concat([obs, tf.reshape(loc, shape = [-1, 2])],1)
            action_single = action[:,i,:]
            q_single, optimal_action, qmax, q_values_single = self.generate_single_q_network(obs_n, action_single, trainable)

            q_list.append(q_single)
            action_list.append(optimal_action)
            q_optimal_list.append(qmax)
            q_values_list.append(q_values_single)


        q_values = tf.concat(q_list, axis=1)
        qmax_values = tf.concat(q_optimal_list, axis=1)
        optimal_action = tf.concat(action_list, axis=1)
        optimal_action_onehot = tf.one_hot(optimal_action, self.action_dim_single, on_value=1.0, off_value=0.0, axis=-1)
        s_penalty = tf.concat([tf.layers.flatten(action), s],1)
        sm_penalty = tf.concat([tf.layers.flatten(p), s],1)


        p_hidden_1 = tf.layers.dense(s_penalty , h1, activation=tf.nn.elu,
                                   kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                                   bias_initializer=tf.constant_initializer(0.00),  # biases
                                   use_bias=True,
                                   trainable=trainable, reuse=tf.AUTO_REUSE, name='dense_ph1')
        p_hidden_2 = tf.layers.dense(p_hidden_1, h2, activation=tf.nn.elu,
                                   kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                                   bias_initializer=tf.constant_initializer(0.00),  # biases
                                   use_bias=True,
                                   trainable=trainable, reuse=tf.AUTO_REUSE, name='dense_ph2')

        p_network = tf.layers.dense(p_hidden_2, 1, #activation = tf.nn.softplus,
                                   kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                                   bias_initializer=tf.constant_initializer(0.00),  # biases
                                   use_bias=True,
                                   trainable=trainable, reuse=tf.AUTO_REUSE, name='dense_ph4')

        mp_hidden_1 = tf.layers.dense(sm_penalty , h1, activation=tf.nn.elu,
                                   kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                                   bias_initializer=tf.constant_initializer(0.00),  # biases
                                   use_bias=True, reuse=tf.AUTO_REUSE,
                                   trainable=trainable, name='dense_ph1')
        mp_hidden_2 = tf.layers.dense(mp_hidden_1, h2, activation=tf.nn.elu,
                                   kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                                   bias_initializer=tf.constant_initializer(0.00),  # biases
                                   use_bias=True, reuse=tf.AUTO_REUSE,
                                   trainable=trainable, name='dense_ph2')
        mp_network = tf.layers.dense(mp_hidden_2 , 1, #activation = tf.nn.softplus,
                                   kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                                   bias_initializer=tf.constant_initializer(0.00),  # biases
                                   use_bias=True, reuse=tf.AUTO_REUSE,
                                   trainable=trainable, name='dense_ph4')

        penalty_value = tf.abs(p_network - mp_network)
        penalty_value = tf.reduce_mean(penalty_value, reduction_indices=1, keep_dims = True)
        q_value = tf.reduce_sum(q_values, reduction_indices=1, keep_dims = True) - penalty_value
        value = tf.reduce_sum(qmax_values, reduction_indices=1, keep_dims = True)

        return q_value, optimal_action, value, optimal_action_onehot, tf.reduce_sum(q_values, reduction_indices=1, keep_dims = True)

    def generate_PQMIX2(self, s, action, trainable=True):
        q_list = list()
        q_optimal_list = list()
        q_values_list = list()
        action_list = list()
        key1_list = list()
        key1_t_list = list()
        weight_list = list()

        

        for i in range(self.n_predator):
            obs = tf.reshape(s, shape=[-1, self.state_dim/2, 2])
            loc = tf.reshape(obs[:,i,:], shape = [-1, 1, 2])
            id = tf.reshape(obs[:,0,0], shape = [-1, 1])*0 + float(i)/self.n_predator
            obs = obs - loc
            obs = tf.reshape(obs, shape=[-1, self.state_dim])
            # obs = tf.where(tf.abs(obs) < 2.5 / FLAGS.map_size,
            #                         obs,
            #                         obs*0, name='clipped_error')
            obs_n = tf.concat([obs, id, tf.reshape(loc, shape = [-1, 2])],1)
            # obs_n = tf.concat([obs, tf.reshape(loc, shape = [-1, 2])],1)
            action_single = action[:,i,:]
            q_single, optimal_action, qmax, q_values_single, key1, key1_t, key2 = self.generate_single_q_network_prime(obs_n, action_single, i, trainable)

            q_list.append(q_single)
            action_list.append(optimal_action)
            q_optimal_list.append(qmax)
            q_values_list.append(q_values_single)
            key1_list.append(key1)
            key1_t_list.append(key1_t)



        q_values = tf.concat(q_list, axis=1)
        qmax_values = tf.concat(q_optimal_list, axis=1)
        optimal_action = tf.concat(action_list, axis=1)
        optimal_action_onehot = tf.one_hot(optimal_action, self.action_dim_single, on_value=1.0, off_value=0.0, axis=-1)



        key_vector_1 = tf.concat(key1_list, axis=1)
        key_vector_1_t = tf.concat(key1_t_list, axis=1)


        key_vector_1 = tf.reshape(key_vector_1, shape=[-1, self.n_predator, h5])
        key_vector_1_t = tf.reshape(key_vector_1_t, shape=[-1, self.n_predator, h5])

        key_vector_1 = tf.reduce_mean(key_vector_1, reduction_indices=1)
        key_vector_1_t = tf.reduce_mean(key_vector_1_t, reduction_indices=1)

        s_penalty = key_vector_1
        sm_penalty = key_vector_1_t


        p_hidden_1 = tf.layers.dense(s_penalty , h1, activation=tf.nn.elu,
                                   kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                                   bias_initializer=tf.constant_initializer(0.00),  # biases
                                   use_bias=True, reuse=tf.AUTO_REUSE,
                                   trainable=trainable, name='dense_ph1')
        p_hidden_2 = tf.layers.dense(p_hidden_1, h2, activation=tf.nn.elu,
                                   kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                                   bias_initializer=tf.constant_initializer(0.00),  # biases
                                   use_bias=True, reuse=tf.AUTO_REUSE,
                                   trainable=trainable, name='dense_ph2')

        p_network = tf.layers.dense(p_hidden_2, 1, #activation = tf.nn.softplus,
                                   kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                                   bias_initializer=tf.constant_initializer(0.00),  # biases
                                   use_bias=True, reuse=tf.AUTO_REUSE,
                                   trainable=trainable, name='dense_ph4')

        mp_hidden_1 = tf.layers.dense(sm_penalty , h1, activation=tf.nn.elu,
                                   kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                                   bias_initializer=tf.constant_initializer(0.00),  # biases
                                   use_bias=True, reuse=tf.AUTO_REUSE,
                                   trainable=trainable, name='dense_ph1')
        mp_hidden_2 = tf.layers.dense(mp_hidden_1, h2, activation=tf.nn.elu,
                                   kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                                   bias_initializer=tf.constant_initializer(0.00),  # biases
                                   use_bias=True, reuse=tf.AUTO_REUSE,
                                   trainable=trainable, name='dense_ph2')

        mp_network = tf.layers.dense(mp_hidden_2 , 1, #activation = tf.nn.softplus,
                                   kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                                   bias_initializer=tf.constant_initializer(0.00),  # biases
                                   use_bias=True, reuse=tf.AUTO_REUSE,
                                   trainable=trainable, name='dense_ph4')



        penalty_value = tf.abs(p_network - mp_network) 
        # penalty_value = tf.square(p_network - mp_network)
        penalty_value = tf.reduce_mean(penalty_value, reduction_indices=1, keep_dims = True)




        q_value = tf.reduce_sum(q_values, reduction_indices=1, keep_dims = True) - penalty_value
        # q_value2 = tf.stop_gradient(tf.reduce_sum(q_values, reduction_indices=1, keep_dims = True)) - penalty_value
        value = tf.reduce_sum(qmax_values, reduction_indices=1, keep_dims = True)

        if option_V == True:

            v_hidden_1 = tf.layers.dense(s , h1, activation=tf.nn.relu,
                                    kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                                    bias_initializer=tf.constant_initializer(0.00),  # biases
                                    use_bias=True, reuse=tf.AUTO_REUSE,
                                    trainable=trainable, name='dense_v1')
            v_hidden_2 = tf.layers.dense(v_hidden_1, h2, activation=tf.nn.relu,
                                    kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                                    bias_initializer=tf.constant_initializer(0.00),  # biases
                                    use_bias=True, reuse=tf.AUTO_REUSE,
                                    trainable=trainable, name='dense_v2')
            v_network = tf.layers.dense(v_hidden_2, 1, #activation = tf.nn.softplus,
                                   kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                                   bias_initializer=tf.constant_initializer(0.00),  # biases
                                   use_bias=True, reuse=tf.AUTO_REUSE,
                                   trainable=trainable, name='dense_v3')

            q_value = q_value + v_network
            value = value + v_network

        if option_O == True:
            value = q_value


        return q_value, optimal_action, value, penalty_value, q_value

    def generate_PQMIX3(self, s, action, trainable=True):
        q_list = list()
        q_optimal_list = list()
        q_values_list = list()
        action_list = list()
        hidden_list = list()
        hidden_list_2 = list()


        for i in range(self.n_predator):
            obs = tf.reshape(s, shape=[-1, self.state_dim/2, 2])
            loc = tf.reshape(obs[:,i,:], shape = [-1, 1, 2])
            id = tf.reshape(obs[:,0,0], shape = [-1, 1])*0 + float(i)/self.n_predator
            obs = obs - loc
            obs = tf.reshape(obs, shape=[-1, self.state_dim])
            # obs = tf.where(tf.abs(obs) < 2.5 / FLAGS.map_size,
            #                         obs,
            #                         obs*0, name='clipped_error')
            obs_n = tf.concat([obs, id, tf.reshape(loc, shape = [-1, 2])],1)
            # obs_n = tf.concat([obs, tf.reshape(loc, shape = [-1, 2])],1)
            action_single = action[:,i,:]
            q_single, optimal_action, qmax, q_values_single = self.generate_single_q_network(obs_n, action_single, trainable)

            q_list.append(q_single)
            action_list.append(optimal_action)
            q_optimal_list.append(qmax)
            q_values_list.append(q_values_single)
            # hidden_list.append(hidden)
            # hidden_list_2.append(hidden2)


        q_values = tf.concat(q_list, axis=1)
        qmax_values = tf.concat(q_optimal_list, axis=1)
        optimal_action = tf.concat(action_list, axis=1)
        optimal_action_onehot = tf.one_hot(optimal_action, self.action_dim_single, on_value=1.0, off_value=0.0, axis=-1)
        s_penalty = tf.concat([tf.layers.flatten(action), s],1)
        sm_penalty = tf.concat([tf.layers.flatten(optimal_action_onehot), s],1)


        
        p_hidden_1 = tf.layers.dense(s_penalty , h1, activation=tf.nn.elu,
                                   kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                                   bias_initializer=tf.constant_initializer(0.00),  # biases
                                   use_bias=True, reuse=tf.AUTO_REUSE, 
                                   trainable=trainable, name='dense_ph1')
        p_hidden_2 = tf.layers.dense(p_hidden_1, h2, activation=tf.nn.elu,
                                   kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                                   bias_initializer=tf.constant_initializer(0.00),  # biases
                                   use_bias=True, reuse=tf.AUTO_REUSE,
                                   trainable=trainable, name='dense_ph2')
        p_network = tf.layers.dense(p_hidden_2, 1, #activation = tf.nn.softplus,
                                   kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                                   bias_initializer=tf.constant_initializer(0.00),  # biases
                                   use_bias=True, reuse=tf.AUTO_REUSE,
                                   trainable=trainable, name='dense_ph4')

        mp_hidden_1 = tf.layers.dense(sm_penalty , h1, activation=tf.nn.elu,
                                   kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                                   bias_initializer=tf.constant_initializer(0.00),  # biases
                                   use_bias=True, reuse=tf.AUTO_REUSE,
                                   trainable=trainable, name='dense_ph1')
        mp_hidden_2 = tf.layers.dense(mp_hidden_1, h2, activation=tf.nn.elu,
                                   kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                                   bias_initializer=tf.constant_initializer(0.00),  # biases
                                   use_bias=True, reuse=tf.AUTO_REUSE,
                                   trainable=trainable, name='dense_ph2')
        mp_network = tf.layers.dense(mp_hidden_2 , 1, #activation = tf.nn.softplus,
                                   kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                                   bias_initializer=tf.constant_initializer(0.00),  # biases
                                   use_bias=True, reuse=tf.AUTO_REUSE,
                                   trainable=trainable, name='dense_ph4')


        p_network = tf.abs(p_network)
        mp_network = tf.abs(mp_network)
        penalty_value = tf.reduce_mean(p_network, reduction_indices=1, keep_dims = True)
        m_penalty_value = tf.reduce_mean(mp_network, reduction_indices=1, keep_dims = True)


        q_value = tf.reduce_sum(q_values, reduction_indices=1, keep_dims = True) - penalty_value #* (1-tf.to_float(flag2))
        value = tf.reduce_sum(qmax_values, reduction_indices=1, keep_dims = True) - m_penalty_value

        new_penalty = tf.square(m_penalty_value)
        flag = 1 + tf.reduce_min(tf.layers.flatten(action) - tf.layers.flatten(optimal_action_onehot), reduction_indices=1, keep_dims=True)
        new_penalty = new_penalty * flag


        return q_value, optimal_action, value, new_penalty, tf.reduce_sum(q_values, reduction_indices=1, keep_dims = True)


    def generate_PQMIX4(self, s, action, trainable=True):
        q_list = list()
        q_optimal_list = list()
        q_values_list = list()
        action_list = list()
        key1_list = list()
        key1_t_list = list()
        weight_list = list()
        

        for i in range(self.n_predator):
            obs = tf.reshape(s, shape=[-1, self.state_dim/2, 2])
            loc = tf.reshape(obs[:,i,:], shape = [-1, 1, 2])
            id = tf.reshape(obs[:,0,0], shape = [-1, 1])*0 + float(i)/self.n_predator
            obs = obs - loc
            obs = tf.reshape(obs, shape=[-1, self.state_dim])
            # obs = tf.where(tf.abs(obs) < 2.5 / FLAGS.map_size,
            #                         obs,
            #                         obs*0, name='clipped_error')
            obs_n = tf.concat([obs, id, tf.reshape(loc, shape = [-1, 2])],1)
            # obs_n = tf.concat([obs, tf.reshape(loc, shape = [-1, 2])],1)
            action_single = action[:,i,:]
            q_single, optimal_action, qmax, q_values_single, key1, key1_t, key2 = self.generate_single_q_network_prime(obs_n, action_single, i, trainable)
            

            q_list.append(q_single)
            action_list.append(optimal_action)
            q_optimal_list.append(qmax)
            q_values_list.append(q_values_single)
            key1_list.append(key1)
            key1_t_list.append(key1_t)


        q_values = tf.concat(q_list, axis=1)
        qmax_values = tf.concat(q_optimal_list, axis=1)
        optimal_action = tf.concat(action_list, axis=1)
        optimal_action_onehot = tf.one_hot(optimal_action, self.action_dim_single, on_value=1.0, off_value=0.0, axis=-1)



        key_vector_1 = tf.concat(key1_list, axis=1)
        key_vector_1_t = tf.concat(key1_t_list, axis=1)


        key_vector_1 = tf.reshape(key_vector_1, shape=[-1, self.n_predator, h5])
        key_vector_1_t = tf.reshape(key_vector_1_t, shape=[-1, self.n_predator, h5])

        key_vector_1 = tf.reduce_mean(key_vector_1, reduction_indices=1)
        key_vector_1_t = tf.reduce_mean(key_vector_1_t, reduction_indices=1)

        s_penalty = key_vector_1
        sm_penalty = key_vector_1_t
            

        p_hidden_1 = tf.layers.dense(s_penalty , h1, activation=tf.nn.elu,
                                   kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                                   bias_initializer=tf.constant_initializer(0.00),  # biases
                                   use_bias=True, reuse=tf.AUTO_REUSE,
                                   trainable=trainable, name='dense_ph1')
        p_hidden_2 = tf.layers.dense(p_hidden_1, h2, activation=tf.nn.elu,
                                   kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                                   bias_initializer=tf.constant_initializer(0.00),  # biases
                                   use_bias=True, reuse=tf.AUTO_REUSE,
                                   trainable=trainable, name='dense_ph2')

        


        mp_hidden_1 = tf.layers.dense(sm_penalty , h1, activation=tf.nn.elu,
                                   kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                                   bias_initializer=tf.constant_initializer(0.00),  # biases
                                   use_bias=True, reuse=tf.AUTO_REUSE,
                                   trainable=trainable, name='dense_ph1')
        mp_hidden_2 = tf.layers.dense(mp_hidden_1, h2, activation=tf.nn.elu,
                                   kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                                   bias_initializer=tf.constant_initializer(0.00),  # biases
                                   use_bias=True, reuse=tf.AUTO_REUSE,
                                   trainable=trainable, name='dense_ph2')

        
            
        p_network = tf.layers.dense(p_hidden_2, 1, #activation = tf.nn.softplus,
                                kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                                bias_initializer=tf.constant_initializer(0.00),  # biases
                                use_bias=True, reuse=tf.AUTO_REUSE,
                                trainable=trainable, name='dense_ph4')

        mp_network = tf.layers.dense(mp_hidden_2 , 1, #activation = tf.nn.softplus,
                                kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                                bias_initializer=tf.constant_initializer(0.00),  # biases
                                use_bias=True, reuse=tf.AUTO_REUSE,
                                trainable=trainable, name='dense_ph4')
        

        p_network = tf.abs(p_network-0.0)
        mp_network = tf.abs(mp_network-0.0)

          
        penalty_value = tf.reduce_mean(p_network, reduction_indices=1, keep_dims = True)
        m_penalty_value = tf.reduce_mean(mp_network, reduction_indices=1, keep_dims = True)
         

        q_value = tf.reduce_sum(q_values, reduction_indices=1, keep_dims = True) - penalty_value
        value = tf.reduce_sum(qmax_values, reduction_indices=1, keep_dims = True) #- m_penalty_value


        new_penalty = tf.square(m_penalty_value)
        final_penalty = new_penalty

        if option_V == True:

            v_hidden_1 = tf.layers.dense(s , h1, activation=tf.nn.relu,
                                    kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                                    bias_initializer=tf.constant_initializer(0.00),  # biases
                                    use_bias=True, reuse=tf.AUTO_REUSE,
                                    trainable=trainable, name='dense_v1')
            v_hidden_2 = tf.layers.dense(v_hidden_1, h2, activation=tf.nn.relu,
                                    kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                                    bias_initializer=tf.constant_initializer(0.00),  # biases
                                    use_bias=True, reuse=tf.AUTO_REUSE,
                                    trainable=trainable, name='dense_v2')
            v_network = tf.layers.dense(v_hidden_2, 1, #activation = tf.nn.softplus,
                                   kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                                   bias_initializer=tf.constant_initializer(0.00),  # biases
                                   use_bias=True, reuse=tf.AUTO_REUSE,
                                   trainable=trainable, name='dense_v3')

            q_value = q_value + v_network
            value = value + v_network
        
        if option_O == True:
            value = q_value

        return q_value, optimal_action, value, final_penalty, q_value

    def generate_PQMIX5(self, s, action, trainable=True):
        q_list = list()
        q_optimal_list = list()
        q_values_list = list()
        action_list = list()
        key1_list = list()
        key1_t_list = list()
        key2_list = list()

        for i in range(self.n_predator):
            obs = tf.reshape(s, shape=[-1, self.state_dim/2, 2])
            loc = tf.reshape(obs[:,i,:], shape = [-1, 1, 2])
            id = tf.reshape(obs[:,0,0], shape = [-1, 1])*0 + float(i)/self.n_predator
            obs = obs - loc
            obs = tf.reshape(obs, shape=[-1, self.state_dim])
            # obs = tf.where(tf.abs(obs) < 2.5 / FLAGS.map_size,
            #                         obs,
            #                         obs*0, name='clipped_error')
            obs_n = tf.concat([obs, id, tf.reshape(loc, shape = [-1, 2])],1)
            # obs_n = tf.concat([obs, tf.reshape(loc, shape = [-1, 2])],1)
            action_single = action[:,i,:]
            q_single, optimal_action, qmax, q_values_single, key1, key1_t, key2 = self.generate_single_q_network_prime(obs_n, action_single, i, trainable)
            

            q_list.append(q_single)
            action_list.append(optimal_action)
            q_optimal_list.append(qmax)
            q_values_list.append(q_values_single)
            key1_list.append(key1)
            key1_t_list.append(key1_t)
            key2_list.append(key2)


        q_values = tf.concat(q_list, axis=1)
        qmax_values = tf.concat(q_optimal_list, axis=1)
        optimal_action = tf.concat(action_list, axis=1)
        optimal_action_onehot = tf.one_hot(optimal_action, self.action_dim_single, on_value=1.0, off_value=0.0, axis=-1)
        q_values_all = tf.reshape(tf.concat(q_values_list, axis=1), shape = [-1, self.n_predator, self.action_dim_single])
        

        key_vector_1 = tf.concat(key1_list, axis=1)
        key_vector_1_t = tf.concat(key1_t_list, axis=1)
        key_vector_2 = tf.concat(key2_list, axis=1)

        key_vector_1 = tf.reshape(key_vector_1, shape=[-1, self.n_predator, h5])
        key_vector_1_t = tf.reshape(key_vector_1_t, shape=[-1, self.n_predator, h5])
        key_vector_2 = tf.reshape(key_vector_2, shape=[-1, self.n_predator, h5])


        if option_reg == True:
            F, F_selected, vec, test = self.generate_regularization_network(key_vector_1, key_vector_2, action)
            target_F, target_F_selected, vec_tar, test_tar = self.generate_regularization_network(key_vector_1_t, key_vector_2, optimal_action_onehot)

            q_values_counterfactual = q_values_all - tf.reshape(q_values,shape=[-1,self.n_predator,1]) \
                                    + tf.reshape(tf.reduce_sum(q_values, reduction_indices=1, keep_dims = True),shape=[-1,1,1])
            qmax_values_counterfactual = q_values_all - tf.reshape(qmax_values,shape=[-1,self.n_predator,1]) \
                                    + tf.reshape(tf.reduce_sum(qmax_values, reduction_indices=1, keep_dims = True), shape=[-1,1,1])

            r_q_values_counterfactual = tf.stop_gradient(q_values_counterfactual - F)
            r_qmax_values_counterfactual = tf.stop_gradient(qmax_values_counterfactual - target_F)
            # r_q_values_counterfactual = q_values_counterfactual - F
            # r_qmax_values_counterfactual = qmax_values_counterfactual - target_F
            r_qmax_values_selected = tf.reduce_sum(qmax_values, reduction_indices=1, keep_dims = True) - target_F_selected

            q_value = tf.reduce_sum(q_values, reduction_indices=1, keep_dims = True) - F_selected

            value = tf.reduce_max(tf.reduce_max(r_qmax_values_counterfactual, reduction_indices=2), reduction_indices=1, keep_dims = True)
            
            # Q_diff =  tf.reduce_max(r_qmax_values_counterfactual, reduction_indices=2) -tf.reduce_max(qmax_values_counterfactual, reduction_indices=2)
            # error1 = tf.square(Q_diff)

            Q_diff = tf.where(r_qmax_values_counterfactual - tf.reduce_max(r_qmax_values_counterfactual, reduction_indices=2, keep_dims = True) == 0, \
                    r_qmax_values_counterfactual - qmax_values_counterfactual, 0 * r_qmax_values_counterfactual)
            Q_diff = tf.reduce_sum(Q_diff, reduction_indices=2)
            error1 = tf.square(Q_diff)

            # error2 = tf.reduce_min(q_values_counterfactual - r_q_values_counterfactual, reduction_indices=2)
            # error2 = tf.square(error2)

            penalty = tf.reduce_mean(error1) #+ tf.reduce_mean(error2) 

        else:
            
            Q, Q_selected, vec, test = self.generate_regularization_network2(key_vector_1, key_vector_2, action)
            target_Q, target_Q_selected, vec_tar, test_tar = self.generate_regularization_network2(key_vector_1_t, key_vector_2, optimal_action_onehot)
            
            q_values_counterfactual = q_values_all - tf.reshape(q_values,shape=[-1,self.n_predator,1]) \
                                    + tf.reshape(tf.reduce_sum(q_values, reduction_indices=1, keep_dims = True),shape=[-1,1,1])
            qmax_values_counterfactual = q_values_all - tf.reshape(qmax_values,shape=[-1,self.n_predator,1]) \
                                    + tf.reshape(tf.reduce_sum(qmax_values, reduction_indices=1, keep_dims = True), shape=[-1,1,1])

            r_q_values_counterfactual = tf.stop_gradient(Q)
            r_qmax_values_counterfactual = tf.stop_gradient(target_Q)
            r_qmax_values_selected = tf.stop_gradient(target_Q_selected)
            target_F_selected = tf.reduce_sum(qmax_values, reduction_indices=1, keep_dims = True) - tf.stop_gradient(target_Q_selected)

 
            q_value = Q_selected

            value = tf.reduce_max(tf.reduce_max(r_qmax_values_counterfactual, reduction_indices=2), reduction_indices=1, keep_dims = True)
            # value = tf.reduce_sum(qmax_values, reduction_indices=1, keep_dims = True)
            # value = tf.reduce_mean(target_Q_selected, reduction_indices=1, keep_dims = True)
            # value = tf.reduce_max(r_qmax_values_counterfactual, reduction_indices=2)

            sg_q_values_counterfactual = q_values_all - tf.stop_gradient(tf.reshape(q_values,shape=[-1,self.n_predator,1])) \
                                    + tf.stop_gradient(tf.reshape(tf.reduce_sum(q_values, reduction_indices=1, keep_dims = True),shape=[-1,1,1]))
            sg_qmax_values_counterfactual = q_values_all - tf.stop_gradient(tf.reshape(qmax_values,shape=[-1,self.n_predator,1])) \
                                    + tf.stop_gradient(tf.reshape(tf.reduce_sum(qmax_values, reduction_indices=1, keep_dims = True), shape=[-1,1,1]))


            # Q_diff = tf.reduce_max(r_qmax_values_counterfactual, reduction_indices=2) - tf.reduce_max(qmax_values_counterfactual, reduction_indices=2)
            # error1 = tf.square(Q_diff)

            Q_diff = r_qmax_values_selected - tf.reduce_sum(qmax_values, reduction_indices=1, keep_dims = True)
            error1 = tf.square(Q_diff)

            # Q_diff = tf.reduce_min(qmax_values_counterfactual - r_qmax_values_counterfactual, reduction_indices=2)
            # error1 = tf.square(Q_diff)

            # Q_diff = tf.where(r_qmax_values_counterfactual - tf.reduce_max(r_qmax_values_counterfactual, reduction_indices=2, keep_dims = True) == 0, \
            #         tf.reduce_max(qmax_values_counterfactual, reduction_indices=2, keep_dims = True) - qmax_values_counterfactual, 0 * r_qmax_values_counterfactual)
            # Q_diff = tf.reduce_sum(Q_diff, reduction_indices=2)
            # error1 = tf.abs(Q_diff)

            # Q_diff = tf.where(qmax_values_counterfactual - tf.reduce_max(qmax_values_counterfactual, reduction_indices=2, keep_dims = True) == 0, \
            #         tf.reduce_max(qmax_values_counterfactual, reduction_indices=2, keep_dims = True) - r_qmax_values_counterfactual, 0 * r_qmax_values_counterfactual)
            # Q_diff = tf.reduce_sum(Q_diff, reduction_indices=2)
            # error1 = tf.abs(Q_diff)

            # Q_diff = tf.where(r_qmax_values_counterfactual - tf.reduce_max(r_qmax_values_counterfactual, reduction_indices=2, keep_dims = True) == 0, \
            #         r_qmax_values_counterfactual - qmax_values_counterfactual, 0 * r_qmax_values_counterfactual)
            # Q_diff = tf.reduce_sum(Q_diff, reduction_indices=2)
            # error1 = tf.square(Q_diff)

            # penalty2 = tf.where(penalty2 > 0.0, 1.0 * (penalty2), 0.0 * (penalty2))
            # error2 = q_values_counterfactual - r_q_values_counterfactual
            # error2 = tf.where(error2 < 0.0, 1.0 * (error2), 0.0 * (error2))
            error2 = tf.reduce_min(q_values_counterfactual - r_q_values_counterfactual, reduction_indices=2)
            error2 = tf.square(error2)
        

            penalty = tf.reduce_mean(error1) + tf.reduce_mean(error2) 

            # print error1
            # print error2
            # print penalty

        return q_value, optimal_action, value, penalty, r_qmax_values_counterfactual, q_values_counterfactual,r_qmax_values_counterfactual- tf.reduce_max(r_qmax_values_counterfactual, reduction_indices=2, keep_dims = True), qmax_values_counterfactual- tf.reduce_max(qmax_values_counterfactual, reduction_indices=2, keep_dims = True)

    def generate_PQMIX6(self, s, action, trainable=True):
        q_list = list()
        q_optimal_list = list()
        q_values_list = list()
        action_list = list()


        for i in range(self.n_predator):
            obs = tf.reshape(s, shape=[-1, self.state_dim/2, 2])
            loc = tf.reshape(obs[:,i,:], shape = [-1, 1, 2])
            id = tf.reshape(obs[:,0,0], shape = [-1, 1])*0 + float(i)/self.n_predator
            obs = obs - loc
            obs = tf.reshape(obs, shape=[-1, self.state_dim])
            # obs = tf.where(tf.abs(obs) < 2.5 / FLAGS.map_size,
            #                         obs,
            #                         obs*0, name='clipped_error')
            obs_n = tf.concat([obs, id, tf.reshape(loc, shape = [-1, 2])],1)
            # obs_n = tf.concat([obs, tf.reshape(loc, shape = [-1, 2])],1)
            action_single = action[:,i,:]
            q_single, optimal_action, qmax, q_values_single = self.generate_single_q_network(obs_n, action_single, trainable)

            q_list.append(q_single)
            action_list.append(optimal_action)
            q_optimal_list.append(qmax)
            q_values_list.append(q_values_single)


        q_values = tf.concat(q_list, axis=1)
        qmax_values = tf.concat(q_optimal_list, axis=1)
        optimal_action = tf.concat(action_list, axis=1)
        optimal_action_onehot = tf.one_hot(optimal_action, self.action_dim_single, on_value=1.0, off_value=0.0, axis=-1)

        q_value = tf.reduce_sum(q_values, reduction_indices=1, keep_dims = True)
        value = tf.reduce_sum(qmax_values, reduction_indices=1, keep_dims = True)
        
        return q_value, optimal_action, value, value
        
    def generate_PQMIX7(self, s, action, trainable=True):
        q_list = list()
        q_optimal_list = list()
        q_values_list = list()
        action_list = list()
        key1_list = list()
        key1_t_list = list()
        weight_list = list()
        

        for i in range(self.n_predator):
            obs = tf.reshape(s, shape=[-1, self.state_dim/2, 2])
            loc = tf.reshape(obs[:,i,:], shape = [-1, 1, 2])
            id = tf.reshape(obs[:,0,0], shape = [-1, 1])*0 + float(i)/self.n_predator
            obs = obs - loc
            obs = tf.reshape(obs, shape=[-1, self.state_dim])
            # obs = tf.where(tf.abs(obs) < 2.5 / FLAGS.map_size,
            #                         obs,
            #                         obs*0, name='clipped_error')
            obs_n = tf.concat([obs, id, tf.reshape(loc, shape = [-1, 2])],1)
            # obs_n = tf.concat([obs, tf.reshape(loc, shape = [-1, 2])],1)
            action_single = action[:,i,:]
            q_single, optimal_action, qmax, q_values_single, key1, key1_t, key2 = self.generate_single_q_network_prime(obs_n, action_single, i, trainable)
            

            q_list.append(q_single)
            action_list.append(optimal_action)
            q_optimal_list.append(qmax)
            q_values_list.append(q_values_single)
            key1_list.append(key1)
            key1_t_list.append(key1_t)


        q_values = tf.concat(q_list, axis=1)
        qmax_values = tf.concat(q_optimal_list, axis=1)
        optimal_action = tf.concat(action_list, axis=1)
        optimal_action_onehot = tf.one_hot(optimal_action, self.action_dim_single, on_value=1.0, off_value=0.0, axis=-1)



        key_vector_1 = tf.concat(key1_list, axis=1)
        key_vector_1_t = tf.concat(key1_t_list, axis=1)


        key_vector_1 = tf.reshape(key_vector_1, shape=[-1, self.n_predator, h5])
        key_vector_1_t = tf.reshape(key_vector_1_t, shape=[-1, self.n_predator, h5])

        key_vector_1 = tf.reduce_mean(key_vector_1, reduction_indices=1)
        key_vector_1_t = tf.reduce_mean(key_vector_1_t, reduction_indices=1)

        s_penalty = key_vector_1
        sm_penalty = key_vector_1_t
            

        p_hidden_1 = tf.layers.dense(s_penalty , h1, activation=tf.nn.elu,
                                   kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                                   bias_initializer=tf.constant_initializer(0.00),  # biases
                                   use_bias=True, reuse=tf.AUTO_REUSE,
                                   trainable=trainable, name='dense_ph1')
        p_hidden_2 = tf.layers.dense(p_hidden_1, h2, activation=tf.nn.elu,
                                   kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                                   bias_initializer=tf.constant_initializer(0.00),  # biases
                                   use_bias=True, reuse=tf.AUTO_REUSE,
                                   trainable=trainable, name='dense_ph2')

        


        mp_hidden_1 = tf.layers.dense(sm_penalty , h1, activation=tf.nn.elu,
                                   kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                                   bias_initializer=tf.constant_initializer(0.00),  # biases
                                   use_bias=True, reuse=tf.AUTO_REUSE,
                                   trainable=trainable, name='dense_ph1')
        mp_hidden_2 = tf.layers.dense(mp_hidden_1, h2, activation=tf.nn.elu,
                                   kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                                   bias_initializer=tf.constant_initializer(0.00),  # biases
                                   use_bias=True, reuse=tf.AUTO_REUSE,
                                   trainable=trainable, name='dense_ph2')

        
            
        p_network = tf.layers.dense(p_hidden_2, 1, #activation = tf.nn.softplus,
                                kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                                bias_initializer=tf.constant_initializer(0.00),  # biases
                                use_bias=True, reuse=tf.AUTO_REUSE,
                                trainable=trainable, name='dense_ph4')

        mp_network = tf.layers.dense(mp_hidden_2 , 1, #activation = tf.nn.softplus,
                                kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                                bias_initializer=tf.constant_initializer(0.00),  # biases
                                use_bias=True, reuse=tf.AUTO_REUSE,
                                trainable=trainable, name='dense_ph4')
         

        q_value = p_network
        value = mp_network

        penalty1 = tf.stop_gradient(value) - tf.reduce_sum(qmax_values, reduction_indices=1, keep_dims = True)
        penalty2 = tf.stop_gradient(q_value) - tf.reduce_sum(q_values, reduction_indices=1, keep_dims = True)
            


        penalty1 = tf.square(penalty1)
        penalty2 = tf.where(penalty2 > 0.0, 1.0 * (penalty2), 0.0 * (penalty2))
        penalty2 = tf.square(penalty2)



        error = tf.reduce_mean(penalty1) + tf.reduce_mean(penalty2) #+ option_VE * tf.square(penalty3)
        
        value = tf.reduce_sum(qmax_values, reduction_indices=1, keep_dims = True)

        

        return q_value, optimal_action, value, error, error
    
    def get_action(self, state_ph):
        return self.sess.run(self.cpu_actor_network, feed_dict={self.s_in: state_ph})

    def get_q_values(self, state_ph, action_ph):
        if FLAGS.algorithm == "pqmix":
            target_penalty = self.sess.run(self.cpu_tp, feed_dict={self.s_in: state_ph})
            return self.sess.run(self.cpu_q_network, feed_dict={self.s_in: state_ph,
                                                        self.a_in: action_ph, self.p_in: target_penalty})

        else:
            return self.sess.run(self.cpu_q_network, feed_dict={self.s_in: state_ph,
                                                        self.a_in: action_ph})

    def get_qp_values(self, state_ph, action_ph):
        if FLAGS.algorithm == "pqmix":
            target_penalty = self.sess.run(self.cpu_tp, feed_dict={self.s_in: state_ph})
            return self.sess.run(self.cpu_qp, feed_dict={self.s_in: state_ph,
                                                        self.a_in: action_ph, self.p_in: target_penalty})

        else:
            return self.sess.run(self.cpu_qp, feed_dict={self.s_in: state_ph,
                                                        self.a_in: action_ph})

    def get_target_q_values(self, state_ph, action_ph):
        return self.sess.run(self.target_q_network, feed_dict={self.s_in: state_ph,
                                                               self.a_in: action_ph})

    def training_qnet(self, minibatch):
        y = []
        self.cnt += 1


        # Get target value from target network
        if FLAGS.algorithm == "pqmix":
            # target_action, stp = self.sess.run([self.target_action_onehot,self.tp], feed_dict={self.s_in: [data[3] for data in minibatch]})
            stp = self.sess.run(self.tp, feed_dict={self.s_in: [data[3] for data in minibatch]})
            # target_q_values = self.sess.run(self.target_q_network, feed_dict={self.s_in: [data[3] for data in minibatch], self.a_in: target_action, self.p_in : stp})
            target_q_values = self.sess.run(self.tv, feed_dict={self.s_in: [data[3] for data in minibatch], self.p_in : stp})
            target_penalty = self.sess.run(self.tp, feed_dict={self.s_in: [data[0] for data in minibatch]})
        else:
            if option_O == True:
                target_action = self.sess.run(self.action_onehot, feed_dict={self.s_in: [data[3] for data in minibatch]})
                target_q_values = self.sess.run(self.tv, feed_dict={self.s_in: [data[3] for data in minibatch], self.a_in: target_action})
            else:     
                target_q_values = self.sess.run(self.tv, feed_dict={self.s_in: [data[3] for data in minibatch]})
        # target_penalty = self.sess.run(self.tp, feed_dict={self.s_in: [data[0] for data in minibatch]})

        # target_q_values = self.sess.run(self.target_v, feed_dict={self.s_in: [data[3] for data in minibatch]})
        y = np.zeros([minibatch_size])
        r = np.array([[data[2]] for data in minibatch])
        done = np.array([[data[4]] for data in minibatch])
        y = r + gamma * (1-done) * target_q_values




        if FLAGS.algorithm == "pqmix":
            c,c2, _= self.sess.run([self.cost, self.meanq, self.train_network], feed_dict={
                    self.y_in: y,
                    self.a_in: [data[1] for data in minibatch],
                    self.s_in: [data[0] for data in minibatch],
                    self.p_in: target_penalty
                })
            self.TDerror += c
            self.TDerror2 += c2
            self.meanq_value += c2
            self.meanq_counter += 1
            # print [data[1] for data in minibatch][0]
            if self.cnt % 10000 == 0 and self.cnt >= 10000:
                
                print "Penalty: ", FLAGS.penalty, "TD-error: ", self.cnt, self.TDerror/10000
                self.TDerror = 0

            if self.cnt % 10000 == 0 and self.cnt >= 10000:
                print "Penalty: ", FLAGS.penalty, "meanQ: ", self.cnt, self.TDerror2/10000, '%.3f' % self.beta
                self.TDerror2 = 0
        elif FLAGS.algorithm == "pqmix3" or FLAGS.algorithm == "pqmix4" or FLAGS.algorithm == "pqmix5" or FLAGS.algorithm == "pqmix7":
            # self.beta = 0.001 #0.001 + 0.999 * self.cnt / FLAGS.training_step
            c1, c2, _ = self.sess.run([self.cost1, self.meanq, self.train_network], feed_dict={
                    self.y_in: y,
                    self.a_in: [data[1] for data in minibatch],
                    self.s_in: [data[0] for data in minibatch],
                    self.beta_in: self.beta
                })
            self.TDerror += c1
            self.TDerror2 += c2
            self.meanq_value += c2
            self.meanq_counter += 1


            if self.cnt % 10000 == 0 and self.cnt >= 10000:
                print "Penalty: ", FLAGS.penalty, "TD-error1: ", self.cnt, self.TDerror/10000, '%.3f' % self.beta
                self.TDerror = 0

            if self.cnt % 10000 == 0 and self.cnt >= 10000:
                print "Penalty: ", FLAGS.penalty, "meanQ: ", self.cnt, self.TDerror2/10000, '%.3f' % self.beta
                self.TDerror2 = 0


        else:
            c,c2,_ = self.sess.run([self.cost, self.meanq,self.train_network], feed_dict={
                    self.y_in: y,
                    self.a_in: [data[1] for data in minibatch],
                    self.s_in: [data[0] for data in minibatch]#,
                    #self.p_in: target_penalty
                })
            self.TDerror += c
            self.TDerror2 += c2
            self.meanq_value += c2
            self.meanq_counter += 1

            if self.cnt % 10000 == 0 and self.cnt >= 10000:
                print "Penalty: ", FLAGS.penalty, "TD-error: ", self.cnt, self.TDerror/10000
                self.TDerror = 0

            if self.cnt % 10000 == 0 and self.cnt >= 10000:
                print "Penalty: ", FLAGS.penalty, "meanQ: ", self.cnt, self.TDerror2/10000, '%.3f' % self.beta
                self.TDerror2 = 0
        

            
    def training_target_qnet(self):
        """
        copy weights from q_network to target q_network
        :return:
        """
        self.sess.run(self.update_target_network)

    def summary(self, reward, step):


        summary = self.sess.run(self.merged, feed_dict = {self.meanq_in: self.meanq_value/self.meanq_counter, self.r_in: reward})

        self.meanq_value = 0
        self.meanq_counter = 0

        return summary



