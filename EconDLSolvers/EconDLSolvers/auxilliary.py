import time
from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F

from . import neural_nets

############################
# random number generation #
############################

def torch_uniform(a,b,size):
	""" uniform random numbers in [a,b] """

	return a + (b-a)*torch.rand(size)

##############
# expansions #
##############

def expand_to_quad(x,Nquad):
	""" expand actions and states shapes for quadrature """
	if len(x.shape) == 2:
		T,N = x.shape
		return x.reshape(T,N,1).repeat((1,1,Nquad))
	else:
		T,N,X = x.shape
		return x.reshape(T,N,1,X).repeat((1,1,Nquad,1))

def expand_to_states(x,states):
	""" expand shocks shapes to states """
	T,N = states.shape[:-1]
	return x.tile((T,N,1))

##########
# create #
##########

def create_value_NN(model,value_target=True,Noutputs_value=1,Q=False):

	# a. unpack
	train = model.train
	par = model.par

	# b. neural net
	if not Q:
		
		model.value_NN = neural_nets.StateValue(par,train,outputs=Noutputs_value).to(train.dtype).to(train.device)
	
	else:

		if train.DoubleQ:
			model.value_NN = neural_nets.ActionValueDouble(par,train).to(train.device).to(train.dtype)
		else:
			model.value_NN = neural_nets.ActionValue(par,train).to(train.device).to(train.dtype)			

	if value_target: model.value_NN_target = deepcopy(model.value_NN)

	# c. optimizer	
	model.value_opt = torch.optim.Adam(
		model.value_NN.parameters(), 
		lr=train.learning_rate_value)

	# d. scheduler
	model.value_scheduler = torch.optim.lr_scheduler.ExponentialLR(
		model.value_opt,
		gamma=train.learning_rate_value_decay)

def create_NN(model,policy=True,value=True,policy_target=True,value_target=True,Noutputs_value=1,Q=False):
	""" create neural nets """

	# a. unpack
	train = model.train
	par = model.par
	
	# b. policy
	if policy:
		
		# i. neural net
		model.policy_NN = neural_nets.Policy(par,train).to(train.dtype).to(train.device)

		if policy_target: model.policy_NN_target = deepcopy(model.policy_NN)

		# ii. optimizer
		model.policy_opt = torch.optim.Adam(
			model.policy_NN.parameters(),
			lr=train.learning_rate_policy)

		# iii. scheduler
		model.policy_scheduler = torch.optim.lr_scheduler.ExponentialLR(
			model.policy_opt,
			gamma=train.learning_rate_policy_decay)
	
	# c. value
	if value:

		if train.N_value_NN is None:
		
			create_value_NN(model,value_target=value_target,Noutputs_value=Noutputs_value,Q=Q)
		
		else:

			# a. containers
			model.value_NNs = []
			if value_target: model.value_NN_targets = []
			model.value_opts = []
			model.value_schedulers = []

			# b. create all
			for j in range(train.N_value_NN):
				
				create_value_NN(model,value_target=value_target,Noutputs_value=Noutputs_value,Q=Q)

				model.value_NNs.append(model.value_NN)
				if value_target: model.value_NN_targets.append(model.value_NN_target)
				model.value_opts.append(model.value_opt)
				model.value_schedulers.append(model.value_scheduler)

			# c. reset
			model.value_NN = None
			if value_target: model.value_NN_target = None
			model.value_opt = None
			model.value_scheduler = None

###########
# NN step #
###########

def NN_step(NN,opt,loss,clip_grad_value=None):
	""" single step of training """
	
	opt.zero_grad()
	loss.backward()
	
	if clip_grad_value is not None: torch.nn.utils.clip_grad_norm_(NN.parameters(),clip_grad_value)
	
	opt.step()

##########
# policy #
##########

def train_policy(model,policy_loss_f,*args):
	""" update policy network """

	t0 = time.perf_counter()

	# a. unpack
	train = model.train
	policy_NN = model.policy_NN
	policy_opt = model.policy_opt
	info = model.info

	if train.k < train.start_train_policy:
		return
	
	# b. prepare
	best_loss = np.inf
	best_policy_NN = None
	best_optim_state = None
	no_improvement = 0

	# c. loop over epochs
	for epoch in range(train.Nepochs_policy):

		# i. loss and update
		policy_loss = policy_update_step(model,policy_loss_f,*args)

		# ii. save best
		if policy_loss < best_loss:
			no_improvement = 0
			best_loss = policy_loss
			if train.epoch_termination:
				best_policy_NN = deepcopy(policy_NN.state_dict())
				best_optim_state = deepcopy(policy_opt.state_dict())
		else:
			no_improvement += 1
		
		# iii. terminate
		if train.epoch_termination:
			if no_improvement >= train.Delta_epoch_policy and epoch >= train.epoch_policy_min:
				break

		info[('policy_loss',train.k,epoch)] = policy_loss

	# c. load best
	if best_loss < np.inf and train.epoch_termination:
		policy_NN.load_state_dict(best_policy_NN)
		policy_opt.load_state_dict(best_optim_state)

	# d. finalize
	info[('policy_epochs',train.k)] = epoch+1
	info[('policy_loss',train.k)] = best_loss
	info['time.update_NN.train_policy'] += time.perf_counter()-t0

def policy_update_step(model,policy_loss_f,*args):

	# a. unpack
	train = model.train
	policy_NN = model.policy_NN
	policy_opt = model.policy_opt

	# b. loss
	policy_loss = policy_loss_f(model,*args)

	# c. update
	NN_step_policy(train,policy_NN,policy_opt,policy_loss)

	return policy_loss.item()

def NN_step_policy(train,NN,opt,loss):
	""" single step of training - for timing"""

	NN_step(NN,opt,loss,train.clip_grad_policy)
	if train.allow_synchronize and torch.cuda.is_available(): torch.cuda.synchronize(device=train.device)

#########
# value #
#########

def train_value_(model,compute_target,value_loss_f,*args,info_details=True):

	# a. unpack
	train = model.train
	value_NN = model.value_NN
	value_opt = model.value_opt
	info = model.info

	# b. compute target
	target = compute_target(model,*args)

	# c. prepare
	best_loss = np.inf
	best_value_NN = None
	best_optim_state = None
	no_improvement = 0

	# d. loop over epochs
	for epoch in range(train.Nepochs_value):

		# i. loss
		value_loss = value_loss_f(model,target,*args)

		# ii. update
		NN_step_value(train,value_NN,value_opt,value_loss)

		# iii. save best
		value_loss = value_loss.item()
		if value_loss < best_loss:
			no_improvement = 0 # reset no improvement
			best_loss = value_loss
			if train.epoch_termination:
				best_value_NN = deepcopy(value_NN.state_dict())
				best_optim_state = deepcopy(value_opt.state_dict())
		else:
			no_improvement += 1 # increase no improvement
				
		# iv. terminate
		if train.epoch_termination:
			if no_improvement >= train.Delta_epoch_value and epoch >= train.epoch_value_min:
				break

		if info_details: info[('value_loss',train.k,epoch)] = value_loss
	
	# d. load best
	if best_loss < np.inf and train.epoch_termination:
		value_NN.load_state_dict(best_value_NN)
		value_opt.load_state_dict(best_optim_state)

	# e. finalize
	info[('value_epochs',train.k)] = epoch+1
	info[('value_loss')] = best_loss

def train_value(model,compute_target,value_loss_f,*args):
	""" update value network """

	t0 = time.perf_counter()
	train = model.train
	info = model.info

	if train.N_value_NN is None:
		
		train_value_(model,compute_target,value_loss_f,*args)
	
	else:

		# a. initialize
		epochs = 0
		best_loss = -np.inf

		# b. loop
		for j in range(train.N_value_NN):

			# i. set
			model.value_NN = model.value_NNs[j]
			if train.use_target_value: model.value_NN_target = model.value_NN_targets[j]
			model.value_opt = model.value_opts[j]
			model.value_scheduler = model.value_schedulers[j]

			# ii. train
			train_value_(model,compute_target,value_loss_f,*args,info_details=False)

			# iii. update
			epochs += info[('value_epochs',train.k)]
			best_loss = max(best_loss,info[('value_loss')])

			# iv. reset
			model.value_NN = None
			if train.use_target_value: model.value_NN_target = None
			model.value_opt = None
			model.value_scheduler = None			

		# c. save
		info[('value_epochs',train.k)] = epochs
		info[('value_loss',train.k)] = best_loss

	info['time.update_NN.train_value'] += time.perf_counter()-t0

def value_loss_f(model,target,*args):
	""" value loss """

	# a. unpack
	train = model.train
	value_NN = model.value_NN

	value_pred = model.eval_value(value_NN,*args)
	value_loss = F.mse_loss(value_pred[...,0].reshape(-1,1),target)

	if train.allow_synchronize and torch.cuda.is_available(): torch.cuda.synchronize(device=train.device)
	return value_loss

def NN_step_value(train,NN,opt,loss):
	""" single step of training - for timing"""

	NN_step(NN,opt,loss,train.clip_grad_value)
	if train.allow_synchronize and torch.cuda.is_available(): torch.cuda.synchronize(device=train.device)

###################
# target networks #
###################

def update_target_value_network(model):
	""" update target value network """

	train = model.train

	if train.N_value_NN is None:
		update_target_network(train.tau,model.value_NN,model.value_NN_target)
	else:
		for j in range(train.N_value_NN):
			update_target_network(train.tau,model.value_NNs[j],model.value_NN_targets[j])

def update_target_policy_network(model):
	""" update target value network """

	update_target_network(model.train.tau,model.policy_NN,model.policy_NN_target)

def update_target_network(adjustment_param,net,target_net):
    """ update target network """

    for target_param,param in zip(target_net.parameters(),net.parameters()):
        target_param.data.copy_(adjustment_param*param.data+(1-adjustment_param)*target_param.data)

#############
# scheduler #
#############

def scheduler_step(model):
	""" step scheduler """

	train = model.train

	if hasattr(model,'policy_scheduler') and (train.start_train_policy is None or train.k > train.start_train_policy): # JD: Should perhaps be >=

		if model.policy_scheduler.get_last_lr()[0] > train.learning_rate_policy_min:
			model.policy_scheduler.step()

	if hasattr(model,'value_scheduler'):
		if train.N_value_NN is None:
			if model.value_scheduler.get_last_lr()[0] > train.learning_rate_value_min:
				model.value_scheduler.step()

		else:
			for j in range(train.N_value_NN):
				if model.value_schedulers[j].get_last_lr()[0] > train.learning_rate_value_min:
					model.value_schedulers[j].step()

############
# transfer #
############

def compute_transfer(R_transfer,transfer_grid,R,do_extrap=False):

    # print(R - R_transfer[0],R_transfer[-1] - R)

    if R < R_transfer[0]:
        if not do_extrap: return np.nan
        fac = (R_transfer[0]-R) / (R_transfer[1]-R_transfer[0])
        transfer = transfer_grid[0] - fac * (transfer_grid[1]-transfer_grid[0])
    elif R > R_transfer[-1]:
        if not do_extrap: return np.nan
        fac = (R-R_transfer[-1]) / (R_transfer[-1]-R_transfer[-2])
        transfer = transfer_grid[-1] + fac * (transfer_grid[-1]-transfer_grid[-2])
    else:
        transfer = np.interp(R,R_transfer,transfer_grid)

    return transfer



######################################
# default model-specific functions #
######################################


def discount_factor(model,states,t0=0,t=None):
	""" discount factor """

	par = model.par

	beta = par.beta*torch.ones_like(states[...,0])
	
	return beta 


def terminal_reward_pd(model,states_pd):
	""" terminal reward """

	train = model.train
	h = torch.zeros((*states_pd.shape[:-1],1),dtype=train.dtype,device=states_pd.device)

	return h

def outcomes(model,states,actions,t0=0,t=None):
	""" default outcomes  - outcomes = actions """

	return actions

def draw_exploration_shocks(self,epsilon_sigma,N):
    """ draw exploration shocks - need to be changed """

    par = self.par

    eps = torch.zeros((par.T,N,par.Nactions))

    for i_action in range(par.Nactions):
        epsilon_sigma_i = epsilon_sigma[i_action]
        eps[:,:,i_action] = torch.normal(0,epsilon_sigma_i,(par.T,N))

    return eps

def exploration(model,states,actions,eps,t=None):

	return actions + eps