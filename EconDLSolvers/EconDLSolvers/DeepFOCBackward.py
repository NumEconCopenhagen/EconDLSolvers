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
	train.Nepochs_policy_t = 10_000*torch.ones((par.T),device=device,dtype=torch.int64)

	# b. not used
	train.clip_grad_value = None
	train.Delta_epoch_value = None
	train.epoch_value_min = None
	train.FOC_value_weight_pol = None
	train.FOC_value_weight_val = None
	train.learning_rate_value = None
	train.learning_rate_value_decay = None
	train.learning_rate_value_min = None
	train.learning_rate_value_schedule = None
	train.learning_rate_value_t	= None
	train.N_value_NN = None
	train.Nepochs_value = None
	train.Nepochs_value_t = None
	train.NFOC_targets = None
	train.Nneurons_value = None
	train.Nneurons_value_t = None
	train.tau = None
	train.tau_schedule = None
	train.use_FOC = None
	train.use_target_policy = None
	train.use_target_value = None	
	train.value_weight_pol = None
	train.value_weight_val = None

def create_NN(model):
	""" create neural nets """

	# a. unpack
	train = model.train
	par = model.par
	
	# b. policy for simultaneous solve
	aux.create_NN(model,value=False)
	
	# c. policy for backwards solve
	model.policy_NN_t  = [None for _ in range(par.T)]
	model.policy_opt_t = [None for _ in range(par.T)]
	model.policy_scheduler_t = [None for _ in range(par.T)]

	for t in range(par.T):

		# i. neural net
		normal_init = train.use_simult_in_backward
		model.policy_NN_t[t] = neural_nets.Policy(par,train,
											backward=True,normal_init=normal_init).to(train.dtype).to(train.device)

		# ii. optimizer
		model.policy_opt_t[t] = torch.optim.Adam(
			model.policy_NN_t[t].parameters(),
			lr=train.learning_rate_policy_t[t])

		# iii. scheduler
		model.policy_scheduler_t[t] = torch.optim.lr_scheduler.ExponentialLR(
			model.policy_opt_t[t],
			gamma=train.learning_rate_policy_decay_t[t])

def scheduler_step(model,t):
	""" step scheduler """

	train = model.train

	if model.policy_scheduler_t[t].get_last_lr()[0] > train.learning_rate_policy_min:
		model.policy_scheduler_t[t].step()

#########
# solve #
#########

def solve_backward(model,do_print=False,model_simult=None):
	""" Solve with backward induction """
	
	info = model.info

	# a. load simultaneous policy from previous solve
	if model.train.use_simult_in_backward:
		model.policy_NN.load_state_dict(model_simult.policy_NN.state_dict())

	# b. generate training sample for backward solve
	model._simulate_training_sample(model.train.epsilon_sigma)

	# c. train neural networks
	model.set_time_per_t(value=False)
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

	# b. sample
	states = train.states
	if not train.use_simult_in_backward: model.draw_endogenous_states()
	if train.terminal_actions_known: states = states[:-1]

	# c. update policy - loop backward
	T_start = par.T-2 if train.terminal_actions_known else par.T-1
	for t in range(T_start,-1,-1):

		info['t0_t'] = time.perf_counter()

		# i. load previous NN state-dict as starting point 
		if do_print: print(f'{t = :3d}')
		if t < T_start:
			model.policy_NN_t[t].load_state_dict(model.policy_NN_t[t+1].state_dict())

		# ii. train policy NN
		states_t = states[t]
		train_policy_backward(t,model,states_t,do_print=do_print)
		
		# iii. kill gradient tracking
		for params in model.policy_NN_t[t].parameters():
			params.requires_grad = False

#########
# train #
#########

def batch_iterator(states,batch_size,*,drop_last=False):
	""" batch iter for minibatch training """

	N = states.size(0)

	# a. produce a fresh random permutation each epoch **on the same device**.
	perm = torch.randperm(N,device=states.device)

	# b. slice the permutation into contiguous blocks.
	for start in range(0,N,batch_size):
		idx = perm[start:start+batch_size]
		if idx.numel() < batch_size and drop_last:
			break
		yield states[idx]

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
		for batch in batch_iterator(states,train.batch_size,drop_last=False):

			# i. loss and update
			policy_loss = policy_update_step_t(t,model,batch)
			policy_loss_average += policy_loss/(train.N//train.batch_size)

		# ii. save best
		if policy_loss_average < best_loss:
			no_improvement = 0
			best_loss = policy_loss_average
			if train.epoch_use_best:
				best_policy_NN = deepcopy(policy_NN.state_dict())
				best_optim_state = deepcopy(policy_opt_t.state_dict())
		else:
			no_improvement += 1
		
		# iii. scheduler step
		model.algo.scheduler_step(model,t)

		# iv. save and print
		info[('avg_policy_loss',t,epoch)] = policy_loss_average
		if epoch%10 == 0:
			if do_print: print(f'\r {epoch = :5d}: {policy_loss_average:6.1e}',end='')

		# v. check early stopping
		t_spend = time.perf_counter() - info['t0_t']
		if t_spend > info['time_per_t']:
			if do_print: print(f'\r {epoch = :5d}: {policy_loss_average:6.1e}',end='')
			if do_print: print(f' time limit reached [{t_spend:5.1f} secs]')
			break
	
	else:

		print(' done')

	# d. load best
	if best_loss < np.inf and train.epoch_termination:
		policy_NN.load_state_dict(best_policy_NN)
		policy_opt_t.load_state_dict(best_optim_state)

def policy_update_step_t(t,model,states):

	# a. unpack
	train = model.train
	policy_opt_t = model.policy_opt_t[t]
	policy_NN_t = model.policy_NN_t[t]

	# b. loss
	policy_loss= policy_loss_f(model,states,t)

	# c. update
	aux.NN_step_policy(train,policy_NN_t,policy_opt_t,policy_loss)

	return policy_loss.item()

def policy_loss_f(model,states,t):
	""" policy loss """

	# a. unpack
	train = model.train
	par = model.par
	policy_NN = model.policy_NN
	policy_NN_t = model.policy_NN_t[t]
	terminal_T = par.T-2 if train.terminal_actions_known else par.T-1
	policy_NN_tplus = model.policy_NN_t[t+1] if t < terminal_T else None

	# b. actions today
	actions = model.eval_policy_t(policy_NN_t,states,t=t)
	outcomes = model.outcomes(states,actions,t=t)

	# c. post-decision states
	states_pd = model.state_trans_pd(states,actions,outcomes=outcomes,t=t)

	# d. future states
	if t < par.T-1:
		states_plus = model._state_trans_t(states_pd,t)
		if train.terminal_actions_known and t == par.T-2:
			actions_plus = model.terminal_actions(states_plus)
		else:
			actions_plus = model.eval_policy_t(policy_NN_tplus,states_plus,t=t+1)
		outcomes_plus = model.outcomes(states_plus,actions_plus,t=t+1)
	else:
		states_plus = None
		actions_plus = None
		outcomes_plus = None

	# f. evaluate equations
	equations = model.eval_equations_FOC_t(states,states_plus,states_pd,actions,actions_plus,outcomes,outcomes_plus,t)

	# take sum across equations weighted with Neq_w
	equations = torch.sum(train.eq_w[None,:]*equations,dim=-1)

	# g. compute loss
	policy_loss = torch.mean(equations)

	if train.allow_synchronize and torch.cuda.is_available(): torch.cuda.synchronize(device=train.device)
	return policy_loss