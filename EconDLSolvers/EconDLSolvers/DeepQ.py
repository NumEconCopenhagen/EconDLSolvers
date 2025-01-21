from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F

# local
from . import auxilliary as aux

value_loss_f = aux.value_loss_f
scheduler_step = aux.scheduler_step

#########
# setup #
#########

def setup(model):
	""" setup training parameters"""

	train = model.train

	# a. targets
	train.tau = 0.20 # target smoothing coefficient - if 1 then no smoothing
	train.use_target_policy = True
	train.use_target_value = True

	# b. solver and exploration
	train.do_exo_actions_periods = 40 # number of iterations with uniform sampling
	train.start_train_policy = train.do_exo_actions_periods + 1 # start uniform sampling after this number of iterations

	# c. misc
	train.store_actions = True # store reward in replay buffer
	train.DoubleQ = False # whether to use double Q learning

def create_NN(model):
	""" create neural nets """

	aux.create_NN(model,Q=True)
	
#########
# solve #
#########

def update_NN(model):
	""" update neural networks """

	# unpack
	train = model.train

	# a. sample
	batch = model.rep_buffer.sample(train.batch_size)
	states, actions = batch.states, batch.actions

	if train.terminal_actions_known:
		states = states[:-1]
		actions = actions[:-1]
	else:
		raise NotImplementedError

	# b. update value
	aux.train_value(model,compute_target,value_loss_f,states,actions)

	# c. update target value
	if train.use_target_value:
		aux.update_target_network(train.tau,model.value_NN,model.value_NN_target)
	
	# d. update policy
	aux.train_policy(model,policy_loss_f,states)

	# e. update target policy
	if train.use_target_policy:
		aux.update_target_network(train.tau, model.policy_NN,model.policy_NN_target)

##################
# value training #
##################

def compute_target(model,states,actions):
	""" compute target """

	# a. unpack
	train = model.train
	use_target_policy = train.use_target_policy
	use_target_value = train.use_target_value

	# b. compute target
	with torch.no_grad():

		# i. reward
		outcomes = model.outcomes(states,actions)
		reward = model.reward(states,actions,outcomes)

		# ii. continuation value
		states_pd = model.state_trans_pd(states,actions,outcomes)
		continuation_value = compute_continuation_value(model,states_pd,use_target_value,use_target_policy)
		
		# iii. target
		discount_factor = model.discount_factor(states)
		target = reward + discount_factor*continuation_value 

		return target

def compute_continuation_value(model,states_pd,use_target_value=False,use_target_policy=False):
	""" compute continuation value given states and actions today"""

	# a. unpack
	par = model.par
	train = model.train
	
	if use_target_value:
		value_NN = model.value_NN_target
	else:
		value_NN = model.value_NN

	if use_target_policy:
		policy_NN = model.policy_NN_target
	else:
		policy_NN = model.policy_NN

	# b. future states
	states_plus = model.state_trans(states_pd,train.quad)

	# c. future action
	action_plus_before = model.eval_policy(policy_NN,states_plus[:-1],t0=1)
	action_plus_terminal = model.terminal_actions(states_plus[-1:])
	action_plus = torch.cat((action_plus_before, action_plus_terminal),dim=0)

	# d. future value
	if train.DoubleQ:
		value_before_1, value_before_2 = model.eval_actionvalue(states_plus[:-1],action_plus[:-1],value_NN,t0=1)
		value_before = torch.min(value_before_1,value_before_2)
	else:
		value_before = model.eval_actionvalue(states_plus[:-1],action_plus[:-1],value_NN,t0=1)
	
	# terminal
	outcomes_terminal = model.outcomes(states_plus[-1:],action_plus_terminal,t0=par.T-1)
	value_terminal = model.reward(states_plus[-1:],action_plus[-1:],outcomes_terminal,t0=par.T-1)	
	value_terminal = value_terminal.reshape(1,states_plus.shape[1],train.Nquad,1)	

	# combine
	value = torch.cat((value_before, value_terminal),dim=0).reshape(states_pd.shape[0],states_pd.shape[1],train.Nquad)

	# e. compute continuation value
	continuation_value = torch.sum(value*train.quad_w[None,None,:],dim=2,keepdim=False)
	continuation_value[-1] = model.terminal_reward_pd(states_pd[-1:])[0,...,0]

	return continuation_value.reshape(states_pd.shape[0],states_pd.shape[1])

def value_loss_f(model,target,states,actions):

	# unpack
	train = model.train
	value_NN = model.value_NN

	if train.DoubleQ:
		value_prediction_1,value_prediction_2 = model.eval_actionvalue(states,actions,value_NN,Q1=False)
		value_prediction_1 = value_prediction_1.reshape(target.shape)
		value_prediction_2 = value_prediction_2.reshape(target.shape)
		value_loss = F.mse_loss(value_prediction_1,target) + F.mse_loss(value_prediction_2,target)
	else:
		value_prediction = model.eval_actionvalue(states,actions,value_NN).reshape(target.shape)
		value_loss = F.mse_loss(value_prediction,target)

	return value_loss

###################
# policy training #
###################

def policy_loss_f(model,states):
	""" policy loss """

	# unpack
	train = model.train
	policy_NN = model.policy_NN
	value_NN = model.value_NN

	# a. compute action
	actions = model.eval_policy(policy_NN,states)
	
	# b. compute value
	value = model.eval_actionvalue(states,actions,value_NN,Q1=True) 

	# c. compute loss
	loss = -value.mean()

	if torch.cuda.is_available(): torch.cuda.synchronize(device=train.device)
	return loss