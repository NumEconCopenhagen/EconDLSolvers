import numpy as np
import torch
import time
from copy import deepcopy

# local
from . import auxilliary as aux
from . import neural_nets

####################
# setup and create #
####################

def setup(model):
	""" setup training parameters"""

	par = model.par
	train = model.train
	device = train.device
	
	# a. algorithm specific
	train.backward = True
	train.use_simult_in_backward = True

	train.Nneurons_policy_t = np.array([50,50])
	train.learning_rate_policy_t = 1e-3*torch.ones((par.T),device=device)
	train.learning_rate_policy_decay_t = 0.99*torch.ones((par.T),device=device)
	train.Nepochs_policy_t = 1000*torch.ones((par.T),device=device,dtype=torch.int64)

	train.Nneurons_policy_t = np.array([50,50])
	train.learning_rate_value_t = 1e-3*torch.ones((par.T),device=device)
	train.learning_rate_value_decay_t = 0.99*torch.ones((par.T),device=device)
	train.Nepochs_value_t = 10_000*torch.ones((par.T),device=device,dtype=torch.int64)

	# b. not used
	train.eq_w = None
	
def create_NN(model):
	""" create neural nets """

	# a. unpack
	train = model.train
	par = model.par
	
	# b. policy and value for simultaneous solve
	aux.create_NN(model,value=False)
	aux.create_NN(model,value=True)
	
	# c. policy for backwards solve
	model.policy_NN_t  = [None for _ in range(par.T)]
	model.policy_opt_t = [None for _ in range(par.T)]
	model.policy_scheduler_t = [None for _ in range(par.T)]

	# d. value for Backwards solve
	model.value_NN_t  = [None for _ in range(par.T)]
	model.value_opt_t = [None for _ in range(par.T)]
	model.value_scheduler_t = [None for _ in range(par.T)]

	for t in range(par.T):

		# i. neural net
		norma_init = train.use_simult_in_backward

		model.policy_NN_t[t] = neural_nets.Policy(par,train,
											backward=True,
											normal_init=norma_init).to(train.dtype).to(train.device)
		
		model.value_NN_t[t] = neural_nets.StateValue(par,train,
											   backward=True,
											   normal_init=norma_init).to(train.dtype).to(train.device)

		# ii. optimizer
		model.policy_opt_t[t] = torch.optim.Adam(
			model.policy_NN_t[t].parameters(),
			lr=train.learning_rate_policy_t[t])
	
		model.value_opt_t[t] = torch.optim.Adam(
			model.value_NN_t[t].parameters(),
			lr=train.learning_rate_value_t[t])

		# iii. scheduler
		model.policy_scheduler_t[t] = torch.optim.lr_scheduler.ExponentialLR(
			model.policy_opt_t[t],
			gamma=train.learning_rate_policy_decay_t[t])
	
		model.value_scheduler_t[t] = torch.optim.lr_scheduler.ExponentialLR(
			model.value_opt_t[t],
			gamma=train.learning_rate_value_decay_t[t])
		
def scheduler_step(model, scheduler, t):
	""" step scheduler """

	train = model.train

	if scheduler.get_last_lr()[0] > train.learning_rate_policy_min:
		scheduler.step()

#########
# solve #
#########

def solve_backward(model,do_print=False,model_simult=None):
	""" Solve with backward induction """
	
	info = model.info

	# a. load simultaneous policy from previous solve
	if model.train.use_simult_in_backward:
		
		model.policy_NN.load_state_dict(model_simult.policy_NN.state_dict())

		if model.train.N_value_NN is None:
			model.value_NN.load_state_dict(model_simult.value_NN.state_dict())
		else:
			for j in range(model.train.N_value_NN):
				model.value_NNs[j].load_state_dict(model_simult.value_NNs[j].state_dict())

	# b. generate training sample for backward solve
	model._simulate_training_sample(model.train.epsilon_sigma)

	# c. train neural networks
	model.set_time_per_t(value=True)
	update_NN(model,do_print=do_print)

	model.info['time'] += time.perf_counter() - info['t0_solve']

def update_NN(model,do_print=False):
	""" update neural net parameters """

	# a. unpack
	par = model.par
	train = model.train
	info = model.info

	if train.use_simult_in_backward:

		for params in model.policy_NN.parameters(): # kill gradient tracking for simultaneous network
			params.requires_grad = False

		if model.train.N_value_NN is None:
			for params in model.value_NN.parameters():
				params.requires_grad = False
		else:
			for j in range(model.train.N_value_NN):
				for params in model.value_NNs[j].parameters():
					params.requires_grad = False

	# b. sample
	states = train.states
	states_pd = train.states_pd
	if not train.use_simult_in_backward: model.draw_endogenous_states()
	states_pd = states_pd[:-1]
	if train.terminal_actions_known: states = states[:-1]

	# c. update policy

	## loop backward
	T_start = par.T-2 if train.terminal_actions_known else par.T-1
	for t in range(T_start,-1,-1):

		# i. load previous NN state-dict as starting point 
		if do_print: print(f'{t = :3d}')
		if t < T_start:
			model.policy_NN_t[t].load_state_dict(model.policy_NN_t[t+1].state_dict())
			model.value_NN_t[t].load_state_dict(model.value_NN_t[t+1].state_dict())
		
		# ii. train value and policy NN
		states_t = states[t]

		if t < par.T-1: 
			
			states_pd_t = states_pd[t]

			if do_print: print(' training value network ')
			
			info['t0_t'] = time.perf_counter()
			train_value_backward(t,model,states_pd_t,do_print=do_print)

		if do_print: print(' training policy network ')
		info['t0_t'] = time.perf_counter()
		train_policy_backward(t,model,states_t,do_print=do_print)

		# iii. kill gradient tracking
		for params in model.policy_NN_t[t].parameters():
			params.requires_grad = False
		for params in model.value_NN_t[t].parameters():
			params.requires_grad = False
	
#########
# train #
#########

def batch_iterator(*tensors, batch_size, drop_last=False):
    """
    Mini-batch iterator over one or more same-length tensors.
    
    Yields a tuple of sliced tensors:
        (tensors[0][idx], tensors[1][idx], …)
    """

    # sanity checks
    N = tensors[0].size(0)
    device = tensors[0].device
    for t in tensors:
        if t.size(0) != N:
            raise ValueError("All tensors must have the same first dimension")

    # random permutation on the right device
    perm = torch.randperm(N, device=device)

    # slice into batches
    for start in range(0, N, batch_size):
        idx = perm[start : start + batch_size]
        if drop_last and idx.numel() < batch_size:
            break
        yield tuple(t[idx] for t in tensors)

##########
# policy #
##########

def train_policy_backward(t,model,states,do_print=False):
	""" update policy network """

	t0 = time.perf_counter()

	# a. unpack
	train = model.train
	policy_opt_t = model.policy_opt_t[t]
	policy_NN = model.policy_NN_t[t]
	info = model.info
	
	# b. prepare
	best_loss = np.inf
	best_policy_NN = None
	best_optim_state = None
	no_improvement = 0

	# c. loop over epochs
	for epoch in range(train.Nepochs_policy_t[t]):

		policy_loss_average = 0
		
		for batch in batch_iterator(states,batch_size=train.batch_size,drop_last=False):

			# i. loss and update
			policy_loss = policy_update_step_t(t,model, batch[0])
			policy_loss_average += policy_loss/(train.N//train.batch_size)

		# ii. save best
		if policy_loss < best_loss:
			no_improvement = 0
			best_loss = policy_loss
			if train.epoch_termination:
				best_policy_NN = deepcopy(policy_NN.state_dict())
				best_optim_state = deepcopy(policy_opt_t.state_dict())
		else:
			no_improvement += 1
		
		# iii. scheduler step
		model.algo.scheduler_step(model,model.policy_scheduler_t[t], t) # is different for each algorithm

		# iv. save and print
		info[('avg_policy_loss',t,epoch)] = policy_loss_average
		if epoch%10 == 0:
			if do_print: print(f'\r {epoch = :5d}: {policy_loss_average:7.1f}',end='')

		# v. check early stopping
		t_spend = time.perf_counter() - info['t0_t']
		if t_spend > info['time_per_t']:
			if do_print: print(f'\r  {epoch = :5d}: {policy_loss_average:7.1f}',end='')
			if do_print: print(f' time limit reached [{t_spend:5.1f} secs]')
			break

	# c. load best
	if best_loss < np.inf and train.epoch_termination:
		policy_NN.load_state_dict(best_policy_NN)
		policy_opt_t.load_state_dict(best_optim_state)

def policy_update_step_t(t,model, states):

	# a. unpack
	train = model.train
	policy_opt_t = model.policy_opt_t[t]
	policy_NN_t = model.policy_NN_t[t]

	# b. loss
	policy_loss = policy_loss_f(model,states,t)

	# c. update
	aux.NN_step_policy(train,policy_NN_t,policy_opt_t,policy_loss)

	return policy_loss.item()

def policy_loss_f(model,states,t):
	""" policy loss """

	# a. unpack
	train = model.train
	par = model.par
	policy_NN = model.policy_NN
	value_NN = model.value_NN
	policy_NN_t = model.policy_NN_t[t]
	value_NN_t = model.value_NN_t[t]
	
	# b. actions today
	actions = model.eval_policy_t(policy_NN_t,states,t=t)
	outcomes = model.outcomes(states,actions,t=t)

	# c. reward
	reward = model.reward(states,actions,outcomes=outcomes,t=t).reshape(-1,1)

	# e. post-decision states
	states_pd = model.state_trans_pd(states,actions,outcomes=outcomes,t=t)

	# f. compute post-decision value
	if t == par.T-1:
		value_pd = model.terminal_reward_pd(states_pd)
	else:
		value_pd = model.eval_value_t(value_NN_t,states_pd,t=t)

	# g. value of choice
	discount_factor = model.discount_factor(states).reshape(-1,1)
	value_of_choice = reward + discount_factor*value_pd

	policy_loss = - torch.mean(value_of_choice)

	return policy_loss

#########
# value #
#########

def train_value_backward(t,model,states_pd,do_print=False):
	""" update policy network """

	t0 = time.perf_counter()

	# a. unpack
	train = model.train
	value_opt = model.value_opt_t[t]
	value_NN = model.value_NN_t[t]
	info = model.info
	
	# b. prepare
	best_loss = np.inf
	best_value_NN = None
	best_optim_state = None
	no_improvement = 0

	assert train.N % train.N_target_batches == 0, 'N must be divisable by batch size'
	Nb = train.N // train.N_target_batches

	# c. loop over epochs
	for epoch in range(train.Nepochs_value_t[t]):

		value_loss_average = 0

		# i. compute target
		with torch.no_grad():

			value_targets = []
			for batch in range(train.N_target_batches):
				states_pd_b = states_pd[batch*Nb:(batch+1)*Nb,:]
				value_target_now = compute_target(model,states_pd_b,t)
				value_targets.append(value_target_now)

			value_target = torch.cat(value_targets)
		
		# ii. loss
		for states, targets in batch_iterator(states_pd,value_target,batch_size=train.batch_size,drop_last=False):

			value_loss = value_update_step_t(t,model,states,targets)
			value_loss_average += value_loss/(train.N//train.batch_size)

		# iii. save best
		if value_loss_average < best_loss:
			no_improvement = 0
			best_loss = value_loss_average
			if train.epoch_termination:
				best_value_NN = deepcopy(value_NN.state_dict())
				best_optim_state = deepcopy(value_opt.state_dict())
		else:
			no_improvement += 1
		
		# iv. scheduler step				
		model.algo.scheduler_step(model,model.value_scheduler_t[t],t)

		# v. save and print
		info[('avg_value_loss',t,epoch)] = value_loss_average
		if epoch%10 == 0:
			if do_print: print(f'\r {epoch = :5d}: {value_loss_average:6.1e}',end='')

		# vi. check early stopping
		t_spend = time.perf_counter() - info['t0_t']
		if t_spend > info['time_per_t']:
			if do_print: print(f'\r  {epoch = :5d}: {value_loss_average:6.1e}',end='')
			if do_print: print(f' time limit reached [{t_spend:5.1f} secs]')
			break

	# c. load best
	if best_loss < np.inf and train.epoch_termination:
		value_NN.load_state_dict(best_value_NN)
		value_opt.load_state_dict(best_optim_state)

def compute_target(model,states_pd, t):
	""" compute target """

	# states_pd.shape = (T,N,Nstates_pd) = (par.T-1,N,Nstates_pd) 
	
	# a. unpack
	par = model.par
	train = model.train

	value_NN_tplus = model.value_NN_t[t+1]
	policy_NN_tplus = model.policy_NN_t[t+1]

	# b. compute target
	numint_weights = train.numint_weights

	# i. future
	states_plus = model._state_trans_t(states_pd, t) # shape = (T,N,Nnumint,Nstates)
	if t == par.T-2 and train.terminal_actions_known:
		actions_plus = model.terminal_actions(states_plus)
	else:
		actions_plus = model.eval_policy_t(policy_NN_tplus,states_plus,t=t+1)
	outcomes_plus = model.outcomes(states_plus,actions_plus,t=t+1)
	reward_plus = model.reward(states_plus,actions_plus,outcomes_plus,t=t+1)

	states_pd_plus = model.state_trans_pd(states_plus,actions_plus,outcomes_plus,t=t+1)

	if t == par.T-2:
		value_pd_plus = model.terminal_reward_pd(states_pd_plus)
	else:
		value_pd_plus = model.eval_value_t(value_NN_tplus,states_pd_plus,t=t+1).squeeze(-1)
	
	# ii. future value before expectation
	discount_factor_plus = model.discount_factor(states_plus)
	value_plus = reward_plus + discount_factor_plus*value_pd_plus
	
	# iii. expected future value
	target_value_pd = torch.sum(value_plus*numint_weights[None,:],dim=-1,keepdim=False) 
	target_value_pd = target_value_pd

	return target_value_pd

def value_update_step_t(t,model, states, target):

	# a. unpack
	train = model.train
	value_opt = model.value_opt_t[t]
	value_NN_t = model.value_NN_t[t]

	# b. loss
	value_loss = value_loss_f(model,target,states,t)

	# c. update
	aux.NN_step_value(train,value_NN_t,value_opt,value_loss)

	return value_loss.item()

def value_loss_f(model,target,states_pd, t):
	""" value loss """

	# a. unpack
	train = model.train
	par = model.par
	value_NN_t = model.value_NN_t[t]

	# b. compute value
	value_pd = model.eval_value_t(value_NN_t,states_pd,t=t).squeeze(-1)

	# c. compute loss
	value_loss = torch.mean((value_pd-target)**2)

	return value_loss