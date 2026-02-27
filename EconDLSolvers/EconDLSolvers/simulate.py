import torch

def simulate(model,ns,eps=None):
	""" Simulate model to get expected discounted life-time utility """  

	with torch.no_grad():

		# a. unpack
		par = model.par
		train = model.train
		dtype = train.dtype
		device = train.device
		policy_NN = model.policy_NN
		if train.backward:
			policy_NN_t = model.policy_NN_t

		states = ns.states
		states_pd = ns.states_pd
		outcomes = ns.outcomes
		actions = ns.actions
		shocks = ns.shocks
		reward = ns.reward

		if par.NDC > 0: # if discrete choices

			if train.N_value_NN is None:
				value_NN = model.value_NN
			else:
				value_NN = model.value_NNs	

			DC = ns.DC
			taste_shocks = ns.taste_shocks
		
		# b. simulate
		N = states.shape[1]
		cum_discount_factor = torch.ones((par.T,N),dtype=dtype,device=device)	

		for t in range(par.T):

			# i. actions
			if t == par.T-1 and train.terminal_actions_known:
				actions[t] = model.terminal_actions(states[t])
			else:
				if not train.backward:
					actions[t] = model.eval_policy(policy_NN,states[t],t=t)
				else:
					actions[t] = model.eval_policy_t(policy_NN_t[t],states[t],t=t)

			# iii. exploration
			if not eps is None:
				actions[t] = model.exploration(states[t],actions[t],eps[t],t=t)
				actions[t] = actions[t].clamp(train.min_actions,train.max_actions)

			# iv. reward and discount factor
			outcomes[t] = model.outcomes(states[t],actions[t],t=t)
			reward[t] = model.reward(states[t],actions[t],outcomes[t],t=t)
			if par.NDC > 0: reward[t] += taste_shocks[t]
			discount_factor = model.discount_factor(states[t],t=t)
			if t < par.T-1: cum_discount_factor[t+1] = cum_discount_factor[t]*discount_factor
			
			# v. transition
			if par.NDC == 0:

				states_pd[t] = model.state_trans_pd(states[t],actions[t],outcomes[t],t=t)
				if t < par.T-1:	
					states[t+1] = model.state_trans(states_pd[t],shocks[t+1],t=t)
				
			else:

				# o. post-decision states
				states_pd_DC = model.state_trans_pd(states[t],actions[t],outcomes[t],t=t)
				
				# oo. continuation values
				values_DC = reward[t].clone()
				if t < par.T-1:
					for i_DC in range(par.NDC):
						states_pd_ = states_pd_DC[:,:,i_DC]
						cont_value = model.eval_value(value_NN,states_pd_,t=t)[...,0]
						values_DC[:,i_DC] += discount_factor*cont_value
				else:
					for i_DC in range(par.NDC):
						states_pd_ = states_pd_DC[:,:,i_DC]
						terminal_value = discount_factor*model.terminal_reward_pd(states_pd_)
						reward[t,:,i_DC] += discount_factor*terminal_value
						values_DC[:,i_DC] += discount_factor*terminal_value
				
				# ooo. discrete choice
				DC[t] = torch.argmax(values_DC,dim=-1)
				for i_DC in range(par.NDC):
					I = DC[t] == i_DC
					states_pd[t][I,:] = states_pd_DC[I,:,i_DC]
					reward[t][~I,i_DC] = 0.0

				# oooo. next period
				if t < par.T-1:
					states[t+1] = model.state_trans(states_pd[t],shocks[t+1],t=t)

			# vi. terminal value
			if par.NDC == 0:
				if t == par.T-1:
					reward[t] += discount_factor*model.terminal_reward_pd(states_pd[t])

	if par.NDC == 0:
		ns.R = torch.sum(cum_discount_factor*reward)/ns.N
	else:
		ns.R = torch.sum(cum_discount_factor[:,:,None]*reward)/ns.N

def simulate_DeepSimulate(model,policy_NN,initial_states,shocks,eps=None):
	""" Simulate to get loss in DeepSimulate """

	# a. unpack
	par = model.par
	train = model.train
	dtype = train.dtype
	device = train.device

	# b. allocate
	N = initial_states.shape[0]
	cum_discount_factor = torch.ones((par.T,N),dtype=dtype,device=device)	
	reward = torch.zeros((par.T,N),dtype=dtype,device=device)
	new_states_t = torch.zeros((N,par.Nstates),dtype=dtype,device=device)
	
	# c. simulate
	for t in range(par.T):

		# i. states in period t 
		if t > 0:
			states_t = new_states_t.clone()
		else:
			states_t = initial_states
		
		# ii. endogenous actions
		if t == par.T-1 and train.terminal_actions_known:
			actions_t = model.terminal_actions(states_t)
		else:
			actions_t = model.eval_policy(policy_NN,states_t,t=t)
			
		# iii. exploration
		N_explore = (train.explore_frac[t]*N).int()
		if N_explore > 0 and not eps is None:

			# a. explorers
			states_t_explore = states_t[:N_explore]
			actions_t_explore = actions_t[:N_explore].detach()
			eps_t_explore = eps[t,:N_explore]

			# o) always the first agents are explorers
			# oo) detaching actions detach future states
			# ooo) exploration imply that R can not be interpreted as expected utility

			actions_t_explore = model.exploration(states_t_explore,actions_t_explore,eps_t_explore,t=t)
			actions_t_explore = actions_t_explore.clamp(train.min_actions,train.max_actions).clamp(train.min_actions,train.max_actions)

			# b. combine with non-explorers
			actions = torch.cat([
				actions_t_explore,
				actions_t[N_explore:]
			], dim=0)

		else: # no exploration at all

			actions = actions_t

		# iii. reward and discount factor
		outcomes_t = model.outcomes(states_t,actions,t=t)
		reward[t] = model.reward(states_t,actions,outcomes_t,t=t)

		if N_explore > 0 and not eps is None:
			
			discount_factor_explore = torch.ones((N_explore,),dtype=dtype,device=device)
			discount_factor_no_explore = model.discount_factor(states_t[N_explore:],t=t)
			
			discount_factor = torch.cat([
				discount_factor_explore,
				discount_factor_no_explore
			], dim=0)

		else:

			discount_factor = model.discount_factor(states_t,t=t)

		if t < par.T-1: cum_discount_factor[t+1] = cum_discount_factor[t]*discount_factor

		# iv. transition
		states_pd_t = model.state_trans_pd(states_t,actions,outcomes_t,t=t)
		if t < par.T-1:
			new_states_t[:,:] = model.state_trans(states_pd_t,shocks[t+1],t=t)
		else:
			reward[t] += discount_factor*model.terminal_reward_pd(states_pd_t)
			
	# d. compute discounted utility

	R = torch.sum(cum_discount_factor*reward)/train.N
	loss = -R

	if torch.cuda.is_available() and train.allow_synchronize: torch.cuda.synchronize(device=device)	
	return loss