#!/usr/bin/env python3
import numpy as np
import random
import helpers

##################################################
##################################################
##################################################
##################################################
##################################################
##################################################
##################################################


# @Daniel: your code goes here
def compute_vel_cmd(dico):
    #getting inputs
    crowd = helpers.get_crowd(dico) #[[id1 x1 y1][id2 x2 y2]...] in meters
    clock = helpers.get_clock(dico) # 1 value float in seconds
    odom = helpers.get_odom(dico) # [x, y, theta, dxdt, dydt, dthetadt]
    report = dico["report"]

    print(dico)

    # Do cool stuff
    #
    #
    #
    #
    #

    # setting output
    vel_cmd = (1, 0)

    return vel_cmd

##################################################
##################################################
##################################################
##################################################
##################################################
##################################################
##################################################


def compute_straight_vel_cmd():
    vel_cmd = (1, 0)
    return vel_cmd

def compute_rand_vel_cmd(dico):
    vel_cmd = (random.random() * 0.2, 60.0*random.random())
    return vel_cmd
