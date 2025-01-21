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
	
	# a. not used
	train.Nneurons_value = None
	train.learning_rate_value = None
	train.learning_rate_value_decay = None
	train.learning_rate_value_min = None
	train.Nepochs_value = None
	train.Delta_epoch_value = None
	train.epoch_value_min = None
	train.tau = None
	train.use_target_policy = None
	train.use_target_value = None	
	train.clip_grad_value = None

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
	states = batch.states

	if train.terminal_actions_known: states = states[:-1]

	# c. update policy
	aux.train_policy(model,policy_loss_f,states)

def policy_loss_f(model,states):
	""" policy loss """

	# a. unpack
	train = model.train
	policy_NN = model.policy_NN

	# b. actions today
	actions = model.eval_policy(policy_NN,states)
	outcomes = model.outcomes(states,actions)

	# c. post-decision states
	states_pd = model.state_trans_pd(states,actions,outcomes=outcomes)

	# d. future states
	if train.terminal_actions_known:
		states_plus = model.state_trans(states_pd,train.quad)
	else:
		states_plus = model.state_trans(states_pd[:-1],train.quad)
	
	# e. future actions
	if train.terminal_actions_known:
		actions_plus_before = model.eval_policy(policy_NN,states_plus[:-1],t0=1)
		actions_plus_terminal = model.terminal_actions(states_plus[-1:])
		actions_plus = torch.cat((actions_plus_before,actions_plus_terminal),dim=0)
	else:
		actions_plus = model.eval_policy(policy_NN,states_plus,t0=1)

	outcomes_plus = model.outcomes(states_plus,actions_plus,t0=1)

	# f. evaluate equations

	# states.shape = (T,N,Nstates)
	# actions.shape = (T,N,Nactions)
	# states_plus.shape = (T,N,Nquad,Nstates)
	# actions_plus.shape = (T,N,Nquad,Nactions)	

	equations = model.eval_equations_FOC(states,states_plus,actions,actions_plus,outcomes,outcomes_plus)
	
	# take sum across equations weighted with Neq_w
	equations = torch.sum(equations*train.eq_w[None,None,:],dim=-1)

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