import numpy as np
import random
from scipy.stats import norm
import matplotlib.pyplot as plt
import statistics
import math
import pickle
from tqdm import tqdm
import collections
import variables as var

T_Y = var.T_Y
T_G = var.T_G
T_R = var.T_R
total_T = T_Y + T_G + T_R



def random_init_state(d_lower, d_upper, v_lower, v_upper, T_Y, T_G, T_R): 
    '''
    randomly initialize a state in the given interval for the given initial distribution for state variables
    '''
    actual_d_0 = np.random.uniform(d_lower, d_upper, 1) #distance is uniformly distributed between lower and upper bound for distance
    actual_v_0 = np.random.uniform(v_lower, v_upper, 1) #velocity is also uniformly distributed between lower_v and upper_v
    actual_phi_probs = np.random.uniform(0, 1, 1) #to initialize the phase, we randomly generates a probabilty, and based on this probabilty and the timing of the phases, we decide which phase to assign

    actual_phi_0 = np.full((actual_phi_probs.shape), '-')
    actual_phi_0[actual_phi_probs<(T_Y/total_T)] = 'Y'
    actual_phi_0[((actual_phi_probs>=(T_Y/total_T)) & (actual_phi_probs<((T_Y+T_G)/total_T)))] = 'G'
    actual_phi_0[actual_phi_probs>=((T_Y+T_G)/total_T)] = 'R'

    #for the assigned phases, now we initilize the t_phi
    Y_occurance = np.count_nonzero(actual_phi_0 == 'Y')
    G_occurance = np.count_nonzero(actual_phi_0 == 'G')
    R_occurance = np.count_nonzero(actual_phi_0 == 'R')
    actual_t_phi_0 = np.full((actual_phi_probs.shape), 0, dtype=float)
    actual_t_phi_0[actual_phi_0=='Y'] = T_Y * np.random.random_sample(size= Y_occurance) 
    actual_t_phi_0[actual_phi_0=='G'] = T_G * np.random.random_sample(size= G_occurance) 
    actual_t_phi_0[actual_phi_0=='R'] = T_R * np.random.random_sample(size= R_occurance) 
    return actual_d_0[0], actual_v_0[0], actual_phi_0[0], actual_t_phi_0[0]



def map_idx_to_state(idx, discrete_d, discrete_v, discrete_phi_t):
    '''
    map the index(row number) in the Q-table to a discretized state
    '''
    len_1 = len(discrete_d)
    len_2 = len(discrete_v)
    len_3 = len(discrete_phi_t)
    
    idx_1 = int(np.floor(idx/(len_2*len_3)))
    distance = discrete_d[idx_1]
    
    idx_2 = int(np.floor((idx - idx_1*len_2*len_3)/len_3))
    velocity = discrete_v[idx_2]
    
    idx_3 = idx - idx_1*len_2*len_3 - idx_2*len_3

    phi = discrete_phi_t[idx_3].flatten()[0]
    t_phi = discrete_phi_t[idx_3].flatten()[1]
    
    return (distance, velocity, phi, int(t_phi))




def map_state_to_idx(state, discrete_d, discrete_v, discrete_phi_t, T_Y, T_R, phi_prev_phi_dict):
    '''
    maps the state to the closest discretized state, and returns its index in the Q-table
    '''
    distance = state[0]
    velocity = state[1]
    if state[3]<=0.5: #if the t_phi is less than 0.5, the closest pair of (phi, t_phi) is (previous_phi, 10)
        phi=phi_prev_phi_dict[state[2]]
        t_phi=10
    elif state[3]>0.5:
        phi = state[2]
        t_phi = state[3]
    distance_diff = np.asarray([(distance - d)**2 for d in discrete_d]) #find the closest distance in the discretized distance list
    distance_idx = np.argmin(distance_diff)

    velocity_diff = np.asarray([(velocity - v)**2 for v in discrete_v]) #find the closest velocity in the discretized velocity list
    velocity_idx = np.argmin(velocity_diff)
        
    t_phi_candidates = np.asarray([int(x[1]) for x in discrete_phi_t if x[0]==phi]) #find the closest t_phi 
    t_phi_diff = np.asarray([(t_phi - t)**2 for t in t_phi_candidates])
    t_phi_idx = np.argmin(t_phi_diff)
    if phi=='Y':
        phi_t_idx = t_phi_idx
    elif phi=='R':
        phi_t_idx = t_phi_idx + (T_Y)
    elif phi=='G':
        phi_t_idx = t_phi_idx + (T_Y+T_R)

    idx = distance_idx*len(discrete_v)*len(discrete_phi_t) + velocity_idx*len(discrete_phi_t) + phi_t_idx
    
    return idx
    




def action_selection(Q_row, velocity, actions, epsilon, v_max, training):
    '''
    given the Q-tabel and with epsilon-greedy approach, returns the action
    '''

    if (velocity>v_max) and training: #if it's in the training phase of the Q-learning and the velocity is greater the maximum allowed velocity, only negative actions are allowed
        feasible_index = [index for index,value in enumerate(actions) if value <0]
    
    if not(velocity>v_max) and training:#if it's in training phase, and the velocity is not greater than the v_max, only the actions are allowed that do not make the velocity negative
        feasible_index = [index for index,value in enumerate(actions) if (velocity+value >=0)]
        
    if not(training): #if it;s not the training phase, all the actions are considered
        feasible_index = [i for i in range(len(actions))]

    if (velocity<0) and (len(feasible_index)==0):#if the velocity is negative and there's no feasible action, select the largest action
            feasible_index = [np.argmax(np.asarray(actions))]
            
    feasible_actions = actions[feasible_index]
    
    rand = np.random.uniform(0,1,1)#epsilon greedy approach
    if rand>=epsilon:
        candidates = [i for i, x in enumerate(Q_row.flatten()[feasible_index]) if x == np.max(Q_row.flatten()[feasible_index])] #find the actions with the greates Q-values
        
        if training:
            candidates = [feasible_index[c] for c in candidates]
            action_idx = random.choice(candidates)
            action = actions[action_idx]
            
        elif not(training):
            candidates = [feasible_actions[c] for c in candidates]
            action_idx = np.random.randint(0,len(candidates))
            action = candidates[action_idx]
        
    elif rand<epsilon:
        action_idx =random.choice(feasible_index)
        action = actions[action_idx]
    
    return action, action_idx



def minmax_action_selection(Q_row, velocity, actions, epsilon, v_max, training):
    '''
    returns the action with epsilon-greedy approach 
    '''
    all_index = np.arange(len(actions))

    if (velocity>v_max) and training:#if it's in the training phase of the Q-learning and the velocity is greater the maximum allowed velocity, only negative actions are allowed
        feasible_index = [index for index,value in enumerate(actions) if value <0]
        not_feasible_index = [idx for idx in all_index if idx not in feasible_index] #all other actions that are not considered
    if not(velocity>v_max) and training:#if it's in training phase, and the velocity is not greater than the v_max, only the actions are allowed that do not make the velocity negative
        feasible_index = [index for index,value in enumerate(actions) if (velocity+value >=0)]
        not_feasible_index = [idx for idx in all_index if idx not in feasible_index]
        
    if not(training): #if it's not in the training phase, all actions in the action space are considered
        feasible_index = [i for i in range(len(actions))]
        not_feasible_index = [idx for idx in all_index if idx not in feasible_index]

    if (velocity<0) and (len(feasible_index)==0):#if the velocity is negative and there's no feasible action, select the largest action
            feasible_index = [np.argmax(np.asarray(actions))]
            not_feasible_index = [idx for idx in all_index if idx not in feasible_index]
    

    Q_row[not_feasible_index] = -1000 #for those non-feasible actions set the q-value to a large negative value
            
    feasible_actions = actions[feasible_index]
    
    rand = np.random.uniform(0,1,1) #epsilon greedy approach
    if rand>=epsilon:
        candidates = [i for i, x in enumerate(Q_row.flatten()[feasible_index]) if x == np.max(Q_row.flatten()[feasible_index])]
        
        if training:
            candidates = [feasible_index[c] for c in candidates]
            action_idx = random.choice(candidates)
            action = actions[action_idx]
            
        elif not(training):
            candidates = [feasible_actions[c] for c in candidates]
            action_idx = np.random.randint(0,len(candidates))
            action = candidates[action_idx]
        
    elif rand<epsilon:
        action_idx =random.choice(feasible_index)
        action = actions[action_idx]
    
    return action, action_idx, not_feasible_index



def update_state(state, action, delta_t, phi_to_T_dict, phi_to_next_phi_dict, std_d, std_v, std_t_phi, training):
    '''
    updates the state with given action and state noise
    '''
    distance = state[0]
    velocity = state[1]
    phi = state[2]
    t_phi = state[3]
   
    distance = distance - velocity*delta_t - 0.5*action*(delta_t**2) + np.random.normal(0, std_d, 1) #update the distance 
    velocity = velocity + action*delta_t + np.random.normal(0, std_v, 1) #update velocity
    rnd_tphi = np.random.normal(0, std_t_phi, 1) #random noise for the t_phi
    if (t_phi + delta_t + rnd_tphi) <= phi_to_T_dict[phi]: #if the updated t_phi (t_phi + delta_t + rnd_tphi) is less than the timing for that phase, we're still in the same phase
        t_phi = (t_phi) + delta_t + rnd_tphi
        phi = phi

    else: #if the updated t_phi (t_phi + delta_t + rnd_tphi) is greater than the timing for that phase, the phase changes
        t_phi = max(t_phi + delta_t + rnd_tphi - phi_to_T_dict[phi], 0)
        phi = phi_to_next_phi_dict[phi]        
    
    if (velocity[0]<0) and not(training):#if the updated velocity is negative due to the state noise, correct it
        
        distance[0]-=0.5*np.abs(velocity[0])
        velocity[0]=0
    return (distance[0], velocity[0], phi, t_phi)




def reward_function(distance, velocity, phi, t_phi, v_max, T_Y):
    '''
    returns the reward of going to the new state of (distance, velocity, phi, t_phi)
    '''
    reward = -2
    if distance<0:#passed the intersection
        if (phi=='G') or (phi=='Y'):
            if velocity>0:
                reward = reward + 100
        elif phi=='R':#if the vehicle passed the intersection while it was red
            if velocity>0:
                reward = reward - 200
    
    return reward


def TD_function(reward, discount_factor, Q_row, q):
    
    '''
    TD function in the training phase of the Q-learning
    '''
    return reward + discount_factor*np.max(Q_row.flatten()) - q

