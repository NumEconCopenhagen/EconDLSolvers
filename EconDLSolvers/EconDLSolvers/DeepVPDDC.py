import torch
import torch.nn.functional as F

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
	train.use_target_policy = False
	train.use_target_value = True

	# b. misc
	train.store_pd = True # store pd states in replay buffer

	# c. FOC
	train.start_train_policy = 10
	train.use_FOC = False

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

	assert not train.terminal_actions_known, "Terminal actions can not be known for DeepVPDDC"

	# a. sample
	batch = model.rep_buffer.sample(train.batch_size)
	states, states_pd = batch.states, batch.states_pd

	states_pd = states_pd[:-1]

	# b. update value
	aux.train_value(model,compute_target,value_loss_f,states_pd)

	# c. update target value
	if train.use_target_value: aux.update_target_value_network(model)

	if train.k >= train.start_train_policy:
	
		# d. update policy
		aux.train_policy(model,policy_loss_f,states)

		# e. update target policy
		if train.use_target_policy:
			aux.update_target_network(train.tau,model.policy_NN,model.policy_NN_target)

###################
# value training #
###################

def choice_prob(par,v0):
	""" compute choice probabilities """

	choice_probs = torch.zeros_like(v0)
	vmax = torch.max(v0,dim=-1,keepdim=True)[0]
	v_vmax = (v0-vmax)/par.sigma_eps
	for d in range(par.NDC):
		choice_probs[...,d] = torch.exp(v_vmax[...,d])/torch.sum(torch.exp(v_vmax),dim=-1)
	
	return choice_probs

def compute_target(model,states_pd):
	""" compute target """

	# a. unpack
	par = model.par
	train = model.train
	dtype = train.dtype
	device = train.device
	
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

		# i. future states
		states_plus = model.state_trans(states_pd,train.quad) # shape = (T,N,Nquad,Nstates)

		# ii. future actions
		actions_plus = model.eval_policy(policy_NN,states_plus,t0=1)

		# iii. future reward
		outcomes_plus = model.outcomes(states_plus,actions_plus,t0=1) # shape = (T,N,Nquad,NDC)
		reward_plus = model.reward(states_plus,actions_plus,outcomes_plus,t0=1) # shape = (T,N,Nquad,NDC)

		# iv. future post-decision states
		states_pd_plus = model.state_trans_pd(states_plus,actions_plus,outcomes_plus,t0=1).permute(0,1,2,4,3) # swap last two dimensions

		# v. future post-decision value
		value_pd_plus_before = model.eval_value_pd(value_NN,states_pd_plus[:-1],t0=1)[...,0]
		value_pd_plus_after = model.terminal_reward_pd(states_pd_plus[-1:])[...,0]
		value_pd_plus = torch.cat([value_pd_plus_before,value_pd_plus_after],dim=0)

		# vi. future value function
		discount_factor = model.discount_factor(states_plus,t0=1)[...,None]
		value_plus = (reward_plus + discount_factor*value_pd_plus).reshape(-1,train.Nquad,par.NDC)


		# vii. expected future value
		target_value_quad = par.sigma_eps*torch.logsumexp(value_plus/par.sigma_eps,dim=-1,keepdim=False)
		target_value = torch.sum(target_value_quad*train.quad_w[None,:],dim=-1,keepdim=True) # target post-decision value function

		# iv. FOC
		if train.use_FOC:
			
			choice_probs = choice_prob(par,value_plus)

			q = model.marginal_reward(states_plus,actions_plus,outcomes_plus,t0=1).reshape(-1,train.Nquad,par.NDC)
			target_q = torch.sum(torch.sum(q*choice_probs,dim=-1,keepdim=False)*train.quad_w[None,:],dim=1,keepdim=True)
			target_value = torch.cat([target_value,target_q],dim=1)	

	return target_value

def value_loss_f(model,target,states_pd):
	""" value loss """

	# a. unpack
	train = model.train
	value_NN = model.value_NN

	# b. prediction
	pred = model.eval_value_pd(value_NN,states_pd)
	value_pd_pred = pred[...,0].reshape(-1,1)

	# c. targets
	target_val = target[...,0].reshape(-1,1)

	# d. loss
	loss = F.mse_loss(value_pd_pred,target_val)
	if train.use_FOC:
		q_pred = pred[...,1].reshape(-1,1)
		target_q = target[...,1].reshape(-1,1)
		loss += F.mse_loss(q_pred,target_q)	

	return loss

###################
# policy training #
###################

def policy_loss_f(model,states):
	""" policy loss """

	# a. unpack
	par = model.par
	train = model.train
	policy_NN = model.policy_NN
	if train.N_value_NN is None:
		value_NN = model.value_NN
	else:
		value_NN = model.value_NNs

	if train.use_input_scaling:
		raise NotImplementedError("Input scaling not implemented")
	
	# b. compute loss
	
	# i. actions
	actions = model.eval_policy(policy_NN,states)

	# ii. current reward
	outcomes = model.outcomes(states,actions)
	reward = model.reward(states,actions,outcomes)
	if train.use_FOC:
		marginal_reward = model.marginal_reward(states,actions,outcomes)

	# iii. post-decision states
	states_pd = model.state_trans_pd(states,actions,outcomes)

	# iv. post-decision value
	states_pd_ = states_pd.permute(0,1,3,2) # swap dim 2 and 3

	val_pred_before = model.eval_value_pd(value_NN,states_pd_[:-1])
	val_pred_after = model.terminal_reward_pd(states_pd_[-1:])[...,0] # terminal period value
	value_pd_pred = torch.cat([val_pred_before[...,0],val_pred_after],dim=0)


	# v. choices-specific value
	discount_factor = model.discount_factor(states)[...,None] # add

	value = reward + discount_factor * value_pd_pred

	# vi. loss 
	loss = - torch.mean(value)

	if train.use_FOC:
		pred_ = val_pred_before[...,1]
		eq_error = model.eval_equations_DeepVPDDC(states,actions,marginal_reward,pred_)
		loss += torch.mean(eq_error)



	return loss