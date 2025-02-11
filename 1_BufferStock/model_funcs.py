import numpy as np
import torch

from EconDLSolvers import expand_to_quad, expand_to_states

#####################
# policy prediction #
#####################

def predict_consumption(par,states,actions):
	""" predict consumption """

	if par.policy_predict == 'savings_rate':
		savings_rate = actions[...,0]
		consumption = states[...,0]*(1-savings_rate)
	elif par.policy_predict == 'consumption':
		consumption = actions[...,0]
	else:
		raise ValueError('policy_predict must be either savings_rate or consumption')

	return consumption

def predict_savings_rate(par,states,actions):
	""" predict savings rate """

	if par.policy_predict == 'savings_rate':
		savings_rate = actions[...,0]
	elif par.policy_predict == 'consumption':
		consumption = actions[...,0]
		savings_rate = 1-consumption/states[...,0]
	else:
		raise ValueError('policy_predict must be either savings_rate or consumption')

	return savings_rate

###########
# utility #
###########

def util(c):
	""" utility """

	return torch.log(c)

def marg_util_c(c):
	""" marginal utility of consumption """

	return 1/c

def inverse_marg_util(u):
	"""Inverse function of marginal utility of consumption """

	return 1/u

###########
# reward #
###########

def outcomes(model,states,actions,t0=0,t=None):
	""" outcomes """

	# Case I: t is None -> t in t0,...,t0+T-1 <= par.T-1:
	#  states.shape = (T,...,Nstates)
	#  actions.shape = (T,...,Nactions)
	# Case II: t in 0,...,T-1, t0 irrelevant:
	#  states.shape = (N,Nstates)
	#  actions.shape = (N,Nactions)

	par = model.par

	c = predict_consumption(par,states,actions)

	return torch.stack((c,),dim=-1)
	# Case: I: shape = (T,...,Noutcomes)
	# Case: II: shape = (N,Noutcomes)

def reward(model,states,actions,outcomes,t0=0,t=None):
	""" reward """

	# Case I: t is None -> t in t0,...,t0+T-1 <= par.T-1:
	# 	states.shape = (T,...,Nstates)
	# 	actions.shape = (T,...,Nactions)
	# Case II: t in 0,...,T-1, t0 irrelevant:
	# 	states.shape = (N,Nstates)
	# 	actions.shape = (N,Nactions)

	par = model.par
	train = model.train

	# a. consumption
	c = outcomes[...,0]

	# b. utility
	u = util(c)

    # create a penalty mask
	if par.policy_predict == 'consumption' and train.algoname == 'DeepVPD': 

		mask = c > states[...,0]

		# calculate the penalty for all elements
		penalty = -1e3 * (c - states[...,0])**2

		# apply the penalty only where the mask is True
		u = u + mask * penalty

	# c. finalize
	return u 
	# Case I: shape = (T,...)
	# Case II: shape = (N,)

def marginal_reward(model,states,actions,outcomes,t0=0,t=None):
	""" marginal reward """

	# Case I: t is None -> t in t0,...,t0+T-1 <= par.T-1:
	#  states.shape = (T,...,Nstates)
	#  actions.shape = (T,...,Nactions)
	# Case II: t in 0,...,T-1, t0 irrelevant:
	#  states.shape = (N,Nstates)
	#  actions.shape = (N,Nactions)

	# a. consumption
	c = outcomes[...,0]

	# b. finalize
	return marg_util_c(c)
	# Case I: shape = (T,...)
	# Case II: shape = (N,)

def discount_factor(model,states,t0=0,t=None):
	""" discount factor """

	# Case I: t is None -> t in t0,...,t0+T-1 <= par.T-1:
	#  states_pd.shape = (T,...,Nstates_pd)
	# Case II: t in 0,...,T-1, t0 irrelevant:
	#  states_pd.shape = (N,Nstates_pd)

	par = model.par

	beta = par.beta*torch.ones_like(states[...,0])
	
	return beta 	
	# Case I: shape (T,...)
	# Case II: shape (N,)
	
############
# terminal #
############

def terminal_actions(model,states):
	""" terminal actions """

	# Case I: states.shape = (1,...,Nstates)
	# Case II: states.shape = (N,Nstates)
	
	par = model.par
	train = model.train
	dtype = train.dtype
	device = train.device

	if par.policy_predict == 'savings_rate':
		actions = (1-((states[...,0])/(1+par.bequest))/states[...,0]).reshape((*states.shape[:-1],1))
	elif par.policy_predict == 'consumption':
		actions = actions = ((states[...,0])/(1+par.bequest)).reshape((*states.shape[:-1],1))
	else:
		raise ValueError('policy_predict must be either savings_rate or consumption')

	if par.KKT:
		multipliers = torch.zeros((*states.shape[:-1],1),dtype=dtype,device=device)
		actions = torch.cat((actions,multipliers),dim=-1)

	return actions 
	# Case I: shape = (1,...,Nactions)
	# Case II: shape = (N,Nactions)

def terminal_reward_pd(model,states_pd):
	""" terminal reward """

	# Case I: states_pd.shape = (1,...,Nstates_pd)
	# Case II: states_pd.shape = (N,Nstates_pd)

	train = model.train
	par = model.par
	dtype = train.dtype
	device = train.device

	m_pd = states_pd[...,0]
	if par.bequest == 0:
		u = torch.zeros_like(m_pd)
	else:
		u = par.bequest*torch.log(m_pd)
	value_pd = u.unsqueeze(-1)
	return value_pd 
	# Case I: shapes = (1,...,1)
	# Case II: shapes = (N,1)

##############
# transition #
##############

def state_trans_pd(model,states,actions,outcomes,t0=0,t=None):
	""" transition to post-decision state """

	# Case I: t is None -> t in t0,...,t0+T-1 <= par.T-1:
	#  states.shape = (T,...,Nstates)
	#  actions.shape = (T,...,Nactions)
	#  outcomes.shape = (T,...,Noutcomes)
	# Case II: t in 0,...,T-1, t0 irrelevant:
	#  states.shape = (N,Nstates)
	#  actions.shape = (N,Nactions)
	#  outcomes.shape = (N,Noutcomes)

	par = model.par

	# a. unpack
	m = states[...,0]
	p = states[...,1]

	# b. consumption
	c = outcomes[...,0]

	# c. post-decision
	m_pd = m-c
	p_pd = p

	# c. finalize
	states_pd = torch.stack((m_pd,p_pd),dim=-1) 
	if par.Nstates_fixed > 0: 
		states_pd = torch.cat((states_pd,states[...,2:]),dim=-1) 
	
	return states_pd 
	# Case I: shape = (T,...,Nstates_pd)
	# Case II: shape = (N,Nstates_pd)

def state_trans(model,states_pd,shocks,t=None):
	""" state transition with quadrature """

	# Case I: t is None -> t in 0,...,T-1 <= par.T-1:
	#  states_pd.shape = (T,N,Nstates_pd)
	#  shocks.shape = (Nquad,Nshocks) [this is quadrature nodes]
	# Case II: t in 0,...,T-1, t0 irrelevant:
	#  states_pd.shape = (N,Nstates_pd)
	#  shocks.shape = (N,Nshocks) [this is actual shocks]

	# a. unpack
	par = model.par
	train = model.train

	m_pd = states_pd[...,0]
	p_pd = states_pd[...,1]

	if par.Nstates_fixed == 0:
		sigma_xi = torch.ones_like(m_pd)*par.sigma_xi_base
		sigma_psi = torch.ones_like(m_pd)*par.sigma_psi_base
		rho_p = torch.ones_like(m_pd)*par.rho_p_base
	elif par.Nstates_fixed == 1:
		sigma_xi = states_pd[...,2]
		sigma_psi = torch.ones_like(m_pd)*par.sigma_psi_base
		rho_p = torch.ones_like(m_pd)*par.rho_p_base
	elif par.Nstates_fixed == 2:
		sigma_xi = states_pd[...,2]
		sigma_psi = states_pd[...,3]
		rho_p = torch.ones_like(m_pd)*par.rho_p_base
	else:
		sigma_xi = states_pd[...,2]
		sigma_psi = states_pd[...,3]
		rho_p = states_pd[...,4]

	xi = shocks[:,0]
	psi = shocks[:,1]

	# b. adjust shape and scale quadrature nodes (when solving)
	if t is None:

		T,N = states_pd.shape[:-1]

		m_pd = expand_to_quad(m_pd,train.Nquad)
		p_pd = expand_to_quad(p_pd,train.Nquad)
		sigma_xi = expand_to_quad(sigma_xi,train.Nquad)
		sigma_psi = expand_to_quad(sigma_psi,train.Nquad)
		rho_p = expand_to_quad(rho_p,train.Nquad)
		xi = expand_to_states(xi,states_pd)
		psi = expand_to_states(psi,states_pd)

	xi = torch.exp(sigma_xi*xi-0.5*sigma_xi**2)
	psi = torch.exp(sigma_psi*psi-0.5*sigma_psi**2)

	# c. next period

	# i. persistent income
	p_plus = p_pd**rho_p*xi

	# ii. actual income
	if t is None: # when solving

		kappa = par.kappa.reshape(par.T,1,1).repeat(1,N,train.Nquad)

		if par.T_retired == par.T:
			y_plus = kappa[:par.T-1]*p_plus*psi
		else:
			y_plus_before = kappa[:par.T_retired]*p_plus[:par.T_retired]*psi[:par.T_retired]
			y_plus_after = kappa[par.T_retired] * torch.ones_like(p_plus[par.T_retired:])
			y_plus = torch.cat((y_plus_before,y_plus_after),dim=0)
	
	else: # when simulating

		if t < par.T_retired:
			y_plus = par.kappa[t]*p_plus*psi
		else:
			y_plus = par.kappa[t]

	# iii. cash-on-hand
	m_plus = par.R*m_pd + y_plus # shape = (T,N,Nquad) or (T,N)
	
	# iv. fixed states
	if par.Nstates_fixed == 0:
		fixed_states_tuple = ()
	elif par.Nstates_fixed == 1:
		fixed_states_tuple = (sigma_xi,)
	elif par.Nstates_fixed == 2:
		fixed_states_tuple = (sigma_xi,sigma_psi)
	else:
		fixed_states_tuple = (sigma_xi,sigma_psi,rho_p)

	# d. finalize
	states_plus = torch.stack((m_plus,p_plus) + fixed_states_tuple,dim=-1)
	return states_plus
	# Case I: shape = (T,N,Nquad,Nstates)
	# Case II: shape = (N,Nstates)

def exploration(model,states,actions,eps,t=None):

	# Case I: t is None -> t in t0,...,t0+T-1 <= par.T-1:
	#  states.shape = (T,...,Nstates)
	#  actions.shape = (T,...,Nactions)
	#  eps.shape = (T,...,Nactions)
	# Case II: t in 0,...,T-1, t0 irrelevant:
	#  states.shape = (N,Nstates)
	#  actions.shape = (N,Nactions)
	#  eps.shape = (N,Nactions)

	return actions + eps
	# Case I: shape = (T,...,Nactions)
	# Case II: shape = (N,Nactions)

###########################
# equations for DeepFOC #
###########################

def eval_equations_FOC(model,states,states_plus,actions,actions_plus,outcomes,outcomes_plus):
	""" evaluate equations for DeepFOC """

	# states.shape = (T,N,Nstates)
	# states_plus.shape = (T,N,Nquad,Nstates)
	# actions.shape = (T,N,Nactions)
	# actions_plus.shape = (T,N,Nquad,Nactions)
	# outcomes.shape = (T,N,Noutcomes)
	# outcomes_plus.shape = (T,N,Nquad,Noutcomes)

	par = model.par

	if par.KKT:
		sq_equations = eval_KKT(model,states,states_plus,actions,actions_plus)
	else:
		sq_equations = eval_burmeister(model,states,states_plus,actions,actions_plus)

	return sq_equations

def eval_burmeister(model,states,states_plus,actions,actions_plus):
	""" evaluate equations for DeepFOC using Fischer Burmeister function """

	par = model.par
	train = model.train

	# a. compute consumption at time t
	c_t = predict_consumption(par,states,actions)

	# b. compute consumption at time t+1
	c_tplus = predict_consumption(par,states_plus,actions_plus)

	# c. compute marginal utility at time t+1
	marg_util_tplus = marg_util_c(c_tplus)

	# d. compute expected marginal utility at time t+1
	exp_marg_util_t1 = torch.sum(train.quad_w[None,None,:]*marg_util_tplus,dim=-1)
	
	# e. euler equation
	beta = discount_factor(model,states)
	FOC = inverse_marg_util(beta*par.R*exp_marg_util_t1)/c_t-1
	
	# f. borrowing constraint
	savings_rate = predict_savings_rate(par,states,actions)
	constraint = savings_rate # >= 0

	# g. combine with fischer burmeister
	eq = fischer_burmeister(FOC,constraint)**2

	return eq.unsqueeze(-1) # shape = (T,N) -> (T,N,1)

def eval_KKT(model,states,states_plus,actions,actions_plus):
	""" evaluate equations for DeepFOC using KKT conditions """

	par = model.par
	train = model.train

	# a. compute consumption and multiplier at time t
	c_t = predict_consumption(par,states,actions)
	multiplier_t = actions[...,1]

	# b. compute consumption at time t+1
	c_tplus = predict_consumption(par,states_plus,actions_plus)

	# c. compute marginal utility at time t+1
	marg_util_tplus = marg_util_c(c_tplus)

	# d. compute expected marginal utility at time t+1
	exp_marg_util_t1 = torch.sum(train.quad_w[None,None,:]*marg_util_tplus,dim=-1)
	
	# e. compute euler equation
	beta = discount_factor(model,states)
	FOC = inverse_marg_util(beta*par.R*exp_marg_util_t1+actions[...,1]) / c_t - 1
	
	# f. borrowing constraint (slackness condition)
	savings_rate = predict_savings_rate(par,states,actions).clamp(0.0,0.9999)
	slackness = savings_rate*multiplier_t # <= 0

	eq = torch.stack((FOC**2,slackness),dim=-1) # shape = (T,N,2)

	return eq # shape = (T,N,2)

#########################
# equations for DeepVPD # 
#########################

def eval_equations_VPD(model,states,action,dvalue_pd):
	""" evaluate equation for DeepVPD with FOC """

	par = model.par
	
	# a. compute consumption at time t
	c_t = predict_consumption(par,states,action)

	# b. compute marginal utility at time t
	marg_util_t = marg_util_c(c_t)

	# c. compute euler equation
	beta = discount_factor(model,states)
	FOC = marg_util_t/(beta*par.R*dvalue_pd[...,0])-1

	# d. constraint
	savings_rate = predict_savings_rate(par,states,action)
	constraint = savings_rate

	# e. combine with fischer burmeister
	eq = fischer_burmeister(FOC,constraint)**2

	return eq

##############
# auxilliary #
##############

def fischer_burmeister(a,b):
	""" Fischer-Burmeister function """
	 
	return torch.sqrt(a**2 + b**2) - a - b