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

	# a. default
	train.start_train_policy = 10 # start training policy after these many iterations
	train.store_pd = True # store pd states in replay buffer

	# b. not used
	train.eq_w = None
	
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
	states, states_pd = batch.states, batch.states_pd # shape = (T,N,Nstates) and (T,N,Nstates_pd)

	states_pd = states_pd[:-1] # terminal reward is always known
	if train.terminal_actions_known: states = states[:-1] # policy not needed for terminal actions

	# b. update value
	aux.train_value(model,value_loss_f,states_pd)

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

def value_loss_f(model,target,states_pd):
	""" value loss """

	# a. unpack
	train = model.train
	value_NN = model.value_NN

	target_value_pd, target_q_pd = target

	# b. baseline
	pred = model.eval_value(value_NN,states_pd)

	value_pd_pred = pred[...,0].reshape(-1,1)
	value_pd_loss = train.value_weight_val * F.mse_loss(value_pd_pred,target_value_pd)

	# c. FOC
	if train.use_FOC:		

		# i. FOC
		q_pd_pred = pred[...,1:].reshape(-1,train.NFOC_targets)
		q_pd_loss = F.mse_loss(q_pd_pred,target_q_pd)*train.NFOC_targets 
		# note: multiply with NFOC_targets to get same scale as value loss
		
		# ii. add to loss
		value_pd_loss += train.FOC_weight_val*q_pd_loss

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
	
	if train.target_value_in_policy:

		if train.N_value_NN is None:
			value_NN = model.value_NN_target
		else:
			value_NN = model.value_NN_targets

	else:

		if train.N_value_NN is None:
			value_NN = model.value_NN
		else:
			value_NN = model.value_NNs

	# a. actions
	actions = model.eval_policy(policy_NN,states) # shape = (T,N,Nactions)

	# b. reward
	outcomes = model.outcomes(states,actions) # shape = (T,N,Nactions)
	reward = model.reward(states,actions,outcomes).reshape(-1,1) # shape = (T*N,1)

	# c. post-decision state and value
	states_pd = model.state_trans_pd(states,actions,outcomes) # shape = (T,N,Nstates_pd)
	value_pd,q_pd = compute_valueq_pd(model,states_pd,value_NN) # shape = (T*N,1) and (T*N,NFOC_targets)

	# d. value of choice
	discount_factor = model.discount_factor(states).reshape(-1,1)
	value_of_choice = reward + discount_factor*value_pd # shape = (T*N,1)

	# e. loss
	loss = train.value_weight_pol*-torch.mean(value_of_choice)

	# f. FOC
	if train.use_FOC:

		# i. FOC
		if train.terminal_actions_known:
			eq = model.eval_equations_VPD(states,actions,outcomes,states_pd,q_pd)
		else:
			eq_ = model.eval_equations_VPD(states[:-1],actions[:-1],outcomes[:-1],states_pd[:-1],q_pd)
			eq_terminal = model.eval_equations_VPD_terminal(states[-1:],actions[-1:],outcomes[-1:],states_pd[-1:])
			eq = torch.cat((eq_,eq_terminal),dim=0)

		# eq.shape = (T,N,Nequations)
		
		# ii. add to loss
		loss += train.FOC_weight_pol*torch.mean(eq)
	
	if train.allow_synchronize and torch.cuda.is_available(): torch.cuda.synchronize(device=train.device)
	return loss

def compute_valueq_pd(model,states_pd,value_NN):
	""" compute post-decision value given current states and actions"""
	
	# unpack
	train = model.train
	
	if train.terminal_actions_known:
		
		# a. eval
		valueq_pd = model.eval_value(value_NN,states_pd)

		# b. value
		value_pd = valueq_pd[...,0].reshape(-1,1)
		
		# c. marginal value
		if train.use_FOC:
			q_pd = valueq_pd[...,1:].reshape(states_pd.shape[0],states_pd.shape[1],train.NFOC_targets)
		else:
			q_pd = None

	else:

		# a. eval
		valueq_pd_ = model.eval_value(value_NN,states_pd[:-1])
		
		# b. value
		value_pd_ = valueq_pd_[...,0].reshape(-1,1)
		value_pd_terminal = model.terminal_reward_pd(states_pd[-1:]).reshape(-1,1)
		value_pd = torch.cat([value_pd_,value_pd_terminal],dim=0).reshape(-1,1)
		
		# c. marginal value
		if train.use_FOC:
			q_pd = valueq_pd_[...,1:].reshape(states_pd[:-1].shape[0],states_pd[:-1].shape[1],train.NFOC_targets)
		else:
			q_pd = None

	return value_pd,q_pd

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
	aux.train_value(model,value_loss_f,states_pd_)
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
