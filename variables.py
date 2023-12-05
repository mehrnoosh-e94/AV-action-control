import numpy as np
import random
from scipy.stats import norm
import matplotlib.pyplot as plt
import statistics
import math
import pickle
from tqdm import tqdm
import collections
from operator import add

font = {'family' : 'normal',
        'size'   : 30}

plt.rc('font', **font)


T_Y = 5
T_G = 10
T_R = 10
total_T = T_Y + T_G + T_R

d_lower = 90
d_upper = 120
v_lower = 10
v_upper = 15

delta_t = 1

phi_list = ['Y','G','R']
phi_T_dict = {'Y': T_Y, 'G': T_G, 'R': T_R}
phi_next_phi_dict = {'G':'Y', 'Y':'R', 'R':'G'}
phi_prev_phi_dict = {'G':'R', 'Y':'G', 'R':'Y'}

# std_d=1
# std_v=1
# std_t_phi=0.1

std_d_m=1
std_v_m=1
phi_obs_prob = 0.9
max_phi_prob = 0.95
min_phi_prob = 0.75

trials = 20000000
episodes = 100
N = 10000

learning_rate = 0.001
discount_factor = 0.99999 
epsilon = 0.1

actions = np.asarray([-3, -2, -1, 0, 1, 2, 3]) #TODO , check for the physics
discrete_d = np.arange(-8, 121, 8)
v_max=15
discrete_v = np.arange(0, v_max+1, 1)

discrete_phi_t = np.asarray([('Y', 1), ('Y', 2), ('Y', 3), ('Y', 4), ('Y', 5)\
                            , ('R', 1), ('R', 2), ('R', 3), ('R', 4), ('R', 5), ('R', 6), ('R', 7), ('R', 8), ('R', 9), ('R', 10)\
                            , ('G', 1), ('G', 2), ('G', 3), ('G', 4), ('G', 5), ('G', 6), ('G', 7), ('G', 8), ('G', 9), ('G', 10)])

nonzero_ratio=0.95

v_eps = 4
critical_distance = 2

alpha_t_phi = 0.5

phi_thre = 0.5

alarm_thre_list = [0.5, 0.6, 0.7, 0.8, 0.9]

critical_d_list=[2,6,10,14]

alarm_thre=0.9
deceleration_a=-3
