#!/usr/bin/env python
# coding=utf8



def config_env(_flags):
    flags = _flags
    

    # Scenario
    flags.DEFINE_string("scenario", "pursuit", "Scenario")
    flags.DEFINE_integer("n_predator", 2, "Number of predators")
    flags.DEFINE_integer("n_prey1", 1, "Number of preys 1")
    flags.DEFINE_integer("n_prey2", 1, "Number of preys 2")
    flags.DEFINE_integer("n_prey", 2, "Number of preys")
    # Observation
    flags.DEFINE_integer("history_len", 1, "How many previous steps we look back")

    # core
    flags.DEFINE_integer("map_size", 3, "Size of the map")
    flags.DEFINE_float("render_every", 1000, "Render the nth episode")

    # Penalty
    flags.DEFINE_integer("penalty", 1, "reward penalty")

def get_filename():
    import config
    FLAGS = config.flags.FLAGS

    return "s-"+FLAGS.scenario+"-map-"+str(FLAGS.map_size)+"-penalty-"+str(FLAGS.penalty)