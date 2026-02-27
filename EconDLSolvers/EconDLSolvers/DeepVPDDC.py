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

	train.use_target_policy = False

	# a. default
	train.store_pd = True

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

	if train.terminal_actions_known:
		raise NotImplementedError('terminal_actions_known=True is not supported for DeepVPDDC')

	# a. sample
	batch = model.rep_buffer.sample(train.batch_size)
	states, states_pd = batch.states, batch.states_pd

	states_pd = states_pd[:-1] # terminal reward is always known

	# b. update value
	aux.train_value(model,value_loss_f,states_pd)

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
			
def value_loss_f(model,target,states_pd):
	""" value loss """

	# a. unpack
	train = model.train
	value_NN = model.value_NN

	target_value_pd, target_q_pd = target

	# b. baslie
	pred = model.eval_value(value_NN,states_pd)

	value_pd_pred = pred[...,0].reshape(-1,1)
	value_pd_loss = train.value_weight_val * F.mse_loss(value_pd_pred,target_value_pd)

	if train.use_FOC:

		# i. FOC
		q_pd_pred = pred[...,1:].reshape(-1,train.NFOC_targets)
		q_pd_loss = F.mse_loss(q_pd_pred,target_q_pd)*train.NFOC_targets 
		# note: multiply with NFOC_targets to get same scale as value loss
		
		# ii. add to loss
		value_pd_loss += train.FOC_weight_val*q_pd_loss

	if train.allow_synchronize and torch.cuda.is_available(): torch.cuda.synchronize(device=train.device)
	return value_pd_loss
	
###################
# policy training #
###################

def policy_loss_f(model,states):
	""" policy loss """

	# a. unpack
	par = model.par
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
	
	# b. actions
	actions = model.eval_policy(policy_NN,states) # shape = (T,N,Nactions)

	# c. current reward
	outcomes = model.outcomes(states,actions) # shape = (T,N,Noutcomes)
	reward = model.reward(states,actions,outcomes) # shape = (T,N,NDC)
	mask_feasible = reward > -1e6

	# d. post-decision states
	states_pd = model.state_trans_pd(states,actions,outcomes) # shape = (T,N,Nstates_pd,NDC)
	states_pd = states_pd.permute(0,1,3,2) # swap last two dimensions, shape = (T,N,NDC,Nstates_pd)

	# e. post-decision value
	value_pd,q_pd = compute_valueq_pd(model,states_pd,value_NN) # shape = (T,N,NDC)

	# f. value of choice
	discount_factor = model.discount_factor(states)[...,None]
	value_of_choice = reward + discount_factor * value_pd # shape = (T,N,NDC)

	# g. loss 
	#loss =  train.value_weight_pol*-torch.mean(value_of_choice)
	loss = train.value_weight_pol*-torch.mean(value_of_choice[mask_feasible])

	# h. FOC
	if train.use_FOC:
		# Non-terminal periods use q_pd from value network
		eq_ = model.eval_equations_DeepVPDDC(states[:-1], actions[:-1], outcomes[:-1], states_pd[:-1], q_pd)
		# Terminal period uses analytical terminal condition
		eq_terminal = model.eval_equations_DeepVPDDC_terminal(states[-1:], actions[-1:], outcomes[-1:], states_pd[-1:])
		eq = torch.cat((eq_, eq_terminal), dim=0)

		# eq.shape = (T, N, NDC)
		loss += train.FOC_weight_pol * torch.mean(eq)

	return loss

def compute_valueq_pd(model,states_pd,value_NN):
	""" compute post-decision value given current states and actions"""
	
	# unpack
	train = model.train

	# a. eval
	valueq_pd_ = model.eval_value(value_NN,states_pd[:-1])
	
	# b. value
	value_pd_ = valueq_pd_[...,0]
	value_pd_terminal = model.terminal_reward_pd(states_pd[-1:])
	value_pd = torch.cat([value_pd_,value_pd_terminal],dim=0)
	
	# c. marginal value
	if train.use_FOC:
		q_pd = valueq_pd_[...,1:]
	else:
		q_pd = None

	return value_pd,q_pd