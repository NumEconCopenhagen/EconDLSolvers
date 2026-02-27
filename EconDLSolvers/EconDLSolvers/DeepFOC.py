import numpy as np
import torch

# for distributed training
from torch.nn.parallel import DistributedDataParallel
import torch.distributed

# local
from . import auxilliary as aux

scheduler_step = aux.scheduler_step

####################
# setup and create #
####################

def setup(model):
	""" setup training parameters"""

	train = model.train

	# a. default
	train.track_sim_policy_loss = False
	train.N_sample_policy_loss = 100
	
	# b. not used
	train.clip_grad_value = None
	train.Delta_epoch_value = None
	train.epoch_value_min = None
	train.FOC_weight_pol = None
	train.FOC_weight_val = None
	train.learning_rate_value = None
	train.learning_rate_value_decay = None
	train.learning_rate_value_min = None
	train.Nepochs_value = None
	train.NFOC_targets = None
	train.Nneurons_value = None
	train.tau = None
	train.use_FOC = None
	train.use_target_policy = None
	train.use_target_value = None	
	train.value_weight_pol = None
	train.value_weight_val = None

def create_NN(model):
	""" create neural nets """

	aux.create_NN(model,value=False)

#########
# solve #
#########

def update_NN(model):
	""" update neural net parameters """

	# a. unpack
	train = model.train

	# b. sample
	batch = model.rep_buffer.sample(train.batch_size)
	states = batch.states # shape = (T,N,Nstates)

	if train.terminal_actions_known: states = states[:-1]

	# c. update policy
	aux.train_policy(model,policy_loss_f,states)

def policy_loss_f(model,states):
	""" policy loss """

	# a. unpack
	train = model.train
	policy_NN = model.policy_NN

	# b. actions today
	actions = model.eval_policy(policy_NN,states) # shape = (T,N,Nactions)
	outcomes = model.outcomes(states,actions) # shape = (T,N,Nactions)

	# c. post-decision states
	states_pd = model.state_trans_pd(states,actions,outcomes=outcomes) # shape = (T,N,Nstates_pd)

	# d. future states
	if train.terminal_actions_known:
		states_plus = model._state_trans(states_pd) # shape = (T,N,Nnumint,Nstates)
	else:
		states_plus = model._state_trans(states_pd[:-1]) # shape = (T,N,Nnumint,Nstates)
	
	# e. future actions
	if train.terminal_actions_known:
		actions_plus_before = model.eval_policy(policy_NN,states_plus[:-1],t0=1)
		actions_plus_terminal = model.terminal_actions(states_plus[-1:])
		actions_plus = torch.cat((actions_plus_before,actions_plus_terminal),dim=0)
		outcomes_plus = model.outcomes(states_plus,actions_plus,t0=1)
	else:
		actions_plus = model.eval_policy(policy_NN,states_plus,t0=1)
		outcomes_plus = model.outcomes(states_plus,actions_plus,t0=1)

	# f. evaluate equations

	# states.shape = (T,N,Nstates)
	# actions.shape = (T,N,Nactions)
	# outcomes.shape = (T,N,Nactions)
	# states_plus.shape = (T,N,Nnumint,Nstates)
	# actions_plus.shape = (T,N,Nnumint,Nactions)	
	# outcomes_plus.shape = (T,N,Nnumint,Nactions)	

	if train.terminal_actions_known:
		equations = model.eval_equations_FOC(states,states_plus,actions,actions_plus,outcomes,outcomes_plus)
	else:
		equations_ = model.eval_equations_FOC(states[:-1],states_plus,actions[:-1],actions_plus,outcomes[:-1],outcomes_plus)
		equations_terminal = model.eval_equations_FOC_terminal(states[-1:],actions[-1:],outcomes[-1:],states_pd[-1:])
		equations = torch.cat((equations_,equations_terminal),dim=0)

	# equations.shape = (T,N,Nequations)

	# take sum across equations weighted with eq_w
	equations = torch.sum(train.eq_w[None,None,:]*equations,dim=-1)

	# g. compute loss
	policy_loss = torch.mean(equations)

	if train.allow_synchronize and torch.cuda.is_available(): torch.cuda.synchronize(device=train.device)
	return policy_loss

#####################################
# solving with distributed training #
#####################################

def update_NN_DDP(model,rank):
	""" update neural net parameters - DDP version """

	# a. unpack
	par = model.par
	train = model.train
	dtype = train.dtype
	device = train.device = rank

	# b. setup policy with DDP
	old_policy_NN = model.policy_NN.to(rank)
	model.policy_NN = DistributedDataParallel(old_policy_NN,device_ids=[rank])

	# c. sample and transfer
	if rank == 0:
		batch = model.rep_buffer.sample(train.batch_size)
		states = batch.states
	else:
		states = torch.zeros((par.T,train.batch_size,par.Nstates),dtype=dtype,device=device)
	
	torch.distributed.barrier()
	torch.distributed.broadcast(states,src=0)

	if train.terminal_actions_known: states = states[:-1]

	# d. split up batch
	batch_size = train.batch_size//train.Ngpus
	states_ = states[:,rank*batch_size:(rank+1)*batch_size,:]

	# e. train model
	aux.train_policy(model,policy_loss_f,states_)

	# f. finalize
	model.policy_NN = old_policy_NN