import numpy as np
import torch

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
	train.tau = 0.2 # target smoothing coefficient
	train.use_target_policy = True
	train.use_target_value = True

	# b. training
	train.start_train_policy = 10
	
def create_NN(model):

	aux.create_NN(model)
	
#########
# solve #
#########

def update_NN(model):
	""" update neural networks """

	# a. unpack
	train = model.train

	# b. sample
	batch = model.rep_buffer.sample(train.batch_size)
	states = batch.states

	# note: for value we formally don't need value in states[0] (not used)

	# c. update value
	aux.train_value(model,compute_target,value_loss_f,states)

	# d. update target value
	if train.use_target_value:
		aux.update_target_network(train.tau,model.value_NN,model.value_NN_target)
	
	# e. update policy
	aux.train_policy(model,policy_loss_f,states)

	# f. update target policy
	if train.use_target_policy:
		aux.update_target_network(train.tau,model.policy_NN,model.policy_NN_target)

#########
# value #
#########

def compute_target(model,states):
	""" compute target """

	# states_pd.shape = (T,N,Nstates_pd)
	
	# a. unpack
	train = model.train
	
	if train.use_target_policy:
		policy_NN = model.policy_NN_target
	else:
		policy_NN = model.policy_NN

	# b. compute target
	with torch.no_grad():

		# i. current action
		if train.terminal_actions_known:
			actions_before = model.eval_policy(policy_NN,states[:-1])
			actions_after = model.terminal_actions(states[-1:])
			actions = torch.cat([actions_before,actions_after],dim=0)
		else:
			actions = model.eval_policy(policy_NN,states)

		# ii. current reward
		outcomes = model.outcomes(states,actions)
		reward = model.reward(states,actions,outcomes).reshape(-1,1)

		# iii. continuation value
		continuation_value = compute_continuation_value(model,states,actions,outcomes)

		# iv. value of choice
		discount_factor = model.discount_factor(states).reshape(-1,1)
		value = reward + discount_factor*continuation_value

		# v. compute target
		target_value = value.reshape(-1,1)
		return target_value

def compute_continuation_value(model,states,actions,outcomes):
	""" compute post-decision value given current state and action"""
	
	# a. unpack
	train = model.train

	if train.use_target_value:
		value_NN = model.value_NN_target
	else:
		value_NN = model.value_NN
	
	# b. post-decision states and future states
	states_pd = model.state_trans_pd(states,actions,outcomes)
	states_plus = model.state_trans(states_pd[:-1],train.quad)

	# c. continous value - before terminal period
	value_plus = model.eval_value(value_NN,states_plus,t0=1).reshape(-1,train.Nquad)
	continuation_value_before = torch.sum(train.quad_w[None,:]*value_plus,dim=1,keepdim=True)

	# d. future value - after expectation
	continuation_value_after = model.terminal_reward_pd(states_pd[-1:]).reshape(-1,1)
	continuation_value = torch.cat([continuation_value_before,continuation_value_after],dim=0)

	return continuation_value
	
##########	
# policy #
##########

def policy_loss_f(model,states):
	""" policy loss """

	# a. unpack
	train = model.train
	policy_NN = model.policy_NN

	# b. actions
	if train.terminal_actions_known: states = states[:-1]
	actions = model.eval_policy(policy_NN,states)

	# c. reward
	outcomes = model.outcomes(states,actions)
	reward = model.reward(states,actions,outcomes).reshape(-1,1)

	# d. continuation value
	continuation_value = compute_continuation_value(model,states,actions,outcomes)

	# e. value of choice
	discount_factor = model.discount_factor(states).reshape(-1,1)
	value_of_choice = reward + discount_factor*continuation_value

	# f. loss
	loss = -torch.mean(value_of_choice)	
	
	if train.allow_synchronize and torch.cuda.is_available(): torch.cuda.synchronize(device=train.device)
	return loss