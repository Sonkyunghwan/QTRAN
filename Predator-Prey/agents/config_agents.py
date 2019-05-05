#!/usr/bin/env python
# coding=utf8
# import agents


def config_agent(_flags):
    flags = _flags

    flags.DEFINE_string("agent", "cac_fo", "Agent")

    flags.DEFINE_integer("training_step", 500000, "Training time step")
    flags.DEFINE_integer("testing_step", 1000, "Testing time step")
    flags.DEFINE_integer("max_step", 200, "Maximum time step per episode")
    flags.DEFINE_integer("eval_step", 1000, "Number of steps before training")
    # flags.DEFINE_integer("training_step", 5000, "Training time step")
    # flags.DEFINE_integer("testing_step", 1000, "Testing time step")
    # flags.DEFINE_integer("max_step", 200, "Maximum time step per episode")
    # flags.DEFINE_integer("eval_step", 1000, "Number of steps before training")

    flags.DEFINE_integer("b_size", 10000, "Size of the replay memory")
    flags.DEFINE_integer("m_size", 32, "Minibatch size")
    flags.DEFINE_integer("pre_train_step", 300, "during [m_size * pre_step] take random action")
    flags.DEFINE_float("lr", 0.00025, "Learning rate")
    # flags.DEFINE_float("lr", 0.01, "Learning rate") # it is for single
    flags.DEFINE_float("df", 0.99, "Discount factor")

    flags.DEFINE_boolean("load_nn", False, "Load nn from file or not")
    flags.DEFINE_string("nn_file", "results/nn/n-2-s-endless3-map-5-penalty-10-a-pqmix5-lr-0.0005-ms-32-seed-28-0103231136-215-3000000", "The name of file for loading")
    
    flags.DEFINE_boolean("train", True, "Training or testing")
    flags.DEFINE_boolean("qtrace", False, "Use q trace")
    flags.DEFINE_boolean("kt", False, "Keyboard input test")
    flags.DEFINE_boolean("use_action_in_critic", False, "Use guided samples")
    flags.DEFINE_string("algorithm", "ddd",
                    "Which agent to run, as a python path to an Agent class.")
    




def get_filename():
    import config
    FLAGS = config.flags.FLAGS

    return "a-"+FLAGS.algorithm+"-lr-"+str(FLAGS.lr)+"-ms-"+str(FLAGS.m_size)
