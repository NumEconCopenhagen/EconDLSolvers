import torch
import torch.nn.functional as F

# for distributed training
from torch.nn.parallel import DistributedDataParallel
import torch.distributed

# local
from . import auxilliary as aux

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

	# b. misc
	train.start_train_policy = 10
	train.store_pd = True # store pd states in replay buffer	

	train.use_FOC = False # use FOC in evaluation of policy

	train.value_weight_val = 1.0 # weight on value loss	
	train.FOC_weight_val = 1.0 # weight on FOC loss
	train.value_weight_pol = 1.0 # weight on value loss	
	train.FOC_weight_pol = 1.0 # weight on FOC loss
	train.NFOC_targets = 1 # number of FOC targets

def create_NN(model):
	""" create neural nets """

	Noutputs_value = 1 if not model.train.use_FOC else 1 + model.train.NFOC_targets
	aux.create_NN(model,Noutputs_value=Noutputs_value)

###########
# solving #
###########

def update_NN(model):
	""" update neural networks """

	# unpack
	train = model.train
	info = model.info

	# a. sample
	batch = model.rep_buffer.sample(train.batch_size)
	states, states_pd = batch.states, batch.states_pd

	states_pd = states_pd[:-1] # terminal reward is always known
	if train.terminal_actions_known: states = states[:-1] # policy not needed for terminal actions

	# b. update value
	aux.train_value(model,compute_target,value_loss_f,states_pd)

	# c. update target value
	if train.use_target_value: aux.update_target_value_network(model)
	
	if train.k >= train.start_train_policy:
	
		# d. update policy
		aux.train_policy(model,policy_loss_f,states)

		# e. update target policy
		if train.use_target_policy: aux.update_target_policy_network(model)
	
	else:

		info[('policy_epochs',train.k)] = 0

###################
# value training #
###################

def compute_target(model,states_pd):
	""" compute target """

	# states_pd.shape = (T,N,Nstates_pd)
	
	# a. unpack
	train = model.train

	if train.use_target_value:
		value_NN = model.value_NN_target
	else:
		value_NN = model.value_NN

	if train.use_target_policy:
		policy_NN = model.policy_NN_target
	else:
		policy_NN = model.policy_NN

	# b. compute target
	with torch.no_grad():

		# note: T = par.T-1 because last states_pd is left out
		
		# i. future states
		states_plus = model.state_trans(states_pd,train.quad) # shape = (T,N,Nquad,Nstates)

		# ii. future actions
		if train.terminal_actions_known:
			actions_plus_before = model.eval_policy(policy_NN,states_plus[:-1],t0=1)		
			actions_plus_after = model.terminal_actions(states_plus[-1:])
			actions_plus = torch.cat([actions_plus_before,actions_plus_after],dim=0)
		else:
			actions_plus = model.eval_policy(policy_NN,states_plus,t0=1)

		# iii. future reward
		outcomes_plus = model.outcomes(states_plus,actions_plus,t0=1)
		reward_plus = model.reward(states_plus,actions_plus,outcomes_plus,t0=1).reshape(-1,train.Nquad)
		
		# iv. future post decision states
		states_pd_plus = model.state_trans_pd(states_plus,actions_plus,outcomes_plus,t0=1)

		# v. future post-decision value
		value_pd_plus_before = model.eval_value_pd(value_NN,states_pd_plus[:-1],t0=1)
		value_pd_plus_before = value_pd_plus_before[...,0].reshape(-1,train.Nquad)	
		value_pd_plus_after = model.terminal_reward_pd(states_pd_plus[-1:]).reshape(-1,train.Nquad)
		value_pd_plus = torch.cat([value_pd_plus_before,value_pd_plus_after],dim=0)

		# vi. future value before expectation
		discount_factor = model.discount_factor(states_pd_plus,t0=1).reshape(-1,train.Nquad)
		value_plus = reward_plus + discount_factor*value_pd_plus

		# vii. expected future value
		target_value_pd = torch.sum(train.quad_w[None,:]*value_plus,dim=1,keepdim=True)

		if not train.use_FOC:
		
			return target_value_pd
		
		else:

			# viii. expected marginal reward
			marginal_reward_pd = model.marginal_reward(states_plus,actions_plus,outcomes_plus,t0=1)
			marginal_reward_pd = marginal_reward_pd.reshape(-1,train.Nquad,train.NFOC_targets)
			target_marginal_reward_pd = torch.sum(train.quad_w[None,:,None]*marginal_reward_pd,dim=1,keepdim=False)

			return target_value_pd, target_marginal_reward_pd

def value_loss_f(model,target,states_pd):
	""" value loss """

	# a. unpack
	train = model.train
	value_NN = model.value_NN

	if train.use_FOC:
		target_value_pd, target_marginal_reward_pd = target
	else:
		target_value_pd = target
		target_marginal_reward_pd = None

	# b. baseline
	pred = model.eval_value_pd(value_NN,states_pd)
	value_pd_pred = pred[...,0].reshape(-1,1)
	value_pd_loss = train.value_weight_val * F.mse_loss(value_pd_pred,target_value_pd)

	# c. FOC
	if train.use_FOC:		

		# i. FOC
		marginal_reward_pd_pred = pred[...,1:train.NFOC_targets+1].reshape(-1,train.NFOC_targets)
		marginal_reward_pd_loss = F.mse_loss(marginal_reward_pd_pred,target_marginal_reward_pd) * train.NFOC_targets # note: multiply with NFOC_targets to get same scale as value loss
		
		# ii. add to loss
		value_pd_loss += train.FOC_weight_val * marginal_reward_pd_loss

	if train.allow_synchronize and torch.cuda.is_available(): torch.cuda.synchronize(device=train.device)
	return value_pd_loss

##########
# policy #
##########

def policy_loss_f(model,states):
	""" policy loss """

	# a. unpack
	train = model.train
	policy_NN = model.policy_NN
	if train.N_value_NN is None:
		value_NN = model.value_NN
	else:
		value_NN = model.value_NNs

	# a. actions
	actions = model.eval_policy(policy_NN,states)

	# b. reward
	outcomes = model.outcomes(states,actions)
	reward = model.reward(states,actions,outcomes).reshape(-1,1)

	# c. post-decision value
	value_pd = compute_value_pd(model,states,actions,outcomes,value_NN)
	value_pd_v = value_pd[...,0].reshape(-1,1)

	# note: naming of value_pd could be improved

	# d. value of choice
	discount_factor = model.discount_factor(states).reshape(-1,1)
	value_of_choice = reward + discount_factor*value_pd_v

	# e. loss
	loss = train.value_weight_pol*-torch.mean(value_of_choice)

	# f. FOC
	if train.use_FOC:

		# i. FOC
		# marginal_reward_pd = value_pd[...,1].reshape(states.shape[0],states.shape[1],train.NFOC_targets)
		marginal_reward_pd = value_pd[...,1:].reshape(states.shape[0],states.shape[1],train.NFOC_targets)
		eq = model.eval_equations_VPD(states,actions,marginal_reward_pd)

		# ii. add to loss
		loss += train.FOC_weight_pol*torch.mean(eq)
	
	if train.allow_synchronize and torch.cuda.is_available(): torch.cuda.synchronize(device=train.device)
	return loss

def compute_value_pd(model,states,actions,outcomes,value_NN):
	""" compute post-decision value given current states and actions"""
	
	# unpack
	train = model.train
	
	# a. post-decision states
	states_pd = model.state_trans_pd(states,actions,outcomes)

	# b. post-decision value
	if train.terminal_actions_known:
		value_pd = model.eval_value_pd(value_NN,states_pd)
	else:
		value_pd_before = model.eval_value_pd(value_NN,states_pd[:-1])
		value_pd_after_ = model.terminal_reward_pd(states_pd[-1:])
		if train.use_FOC:
			marginal_reward_pd_after = model.marginal_terminal_reward(states_pd[-1:])
			value_pd_after = torch.cat([value_pd_after_,marginal_reward_pd_after],dim=-1)
		else:
			value_pd_after = value_pd_after_
		value_pd = torch.cat([value_pd_before,value_pd_after],dim=0)

	return value_pd

########################
# distributed training #
########################

def update_NN_DDP(model,rank):
	""" update neural net parameters - DDP version"""

	# a. unpack
	par = model.par
	train = model.train
	device = train.device = rank

	# b. setup policy and value with DDP
	old_policy_NN = model.policy_NN.to(rank)
	old_value_NN = model.value_NN.to(rank)

	model.policy_NN_target = model.policy_NN_target.to(rank)
	model.value_NN_target = model.value_NN_target.to(rank)	

	model.policy_NN = DistributedDataParallel(old_policy_NN,device_ids=[rank])
	model.value_NN = DistributedDataParallel(old_value_NN,device_ids=[rank])

	# c. sample and transfers
	if rank == 0:
		batch = model.rep_buffer.sample(train.batch_size)
		states = batch.states.to(rank)
		states_pd = batch.states_pd.to(rank)
	else:
		states = torch.zeros((par.T,train.batch_size,par.Nstates),dtype=train.dtype,device=device)
		states_pd = torch.zeros((par.T,train.batch_size,par.Nstates_pd),dtype=train.dtype,device=device)
	
	torch.distributed.barrier()
	torch.distributed.broadcast(states,src=0)
	torch.distributed.broadcast(states_pd,src=0)

	states_pd = states_pd[:-1]
	if train.terminal_actions_known: states = states[:-1]

	# d. split up batch
	batch_size = train.batch_size//train.Ngpus
	states_ = states[:,rank*batch_size:(rank+1)*batch_size,:]
	states_pd_ = states_pd[:,rank*batch_size:(rank+1)*batch_size,:]

	# e. train value
	aux.train_value(model,compute_target,value_loss_f,states_pd_)
	torch.distributed.barrier()

	# f. update target value
	if train.use_target_value:
		aux.update_target_network(train.tau,model.value_NN,model.value_NN_target)
	
	if train.k >= train.start_train_policy:
			
		# update policy
		aux.train_policy(model,policy_loss_f,states_)		
		torch.distributed.barrier()

		# update target policy
		if train.use_target_policy:
			aux.update_target_network(train.tau,model.policy_NN,model.policy_NN_target)

	# g. finalize
	model.policy_NN = old_policy_NN
	model.value_NN = old_value_NN
