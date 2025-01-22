import numpy as np
import torch

from EconDLSolvers import expand_to_quad, expand_to_states, discount_factor, terminal_reward_pd, exploration

###########
# utility #
###########

def util(c,ell,par):
	""" utility """

	return torch.log(c) + par.vphi * ((1-ell)**(1-par.nu))/(1-par.nu)

def marg_util_c(c):
	""" marginal utility of consumption """

	return 1/c

def marg_util_ell(ell, par):
	""" marginal disutility of work """

	return -par.vphi * (1-ell)**(-par.nu)

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

	# a. unpack
	par = model.par
	train = model.train

	# b. get labor supply and savings_rate from actions
	savings_rate = actions[...,0]
	ell = actions[...,1]

	# c. get income
	b = states[...,0]
	p = states[...,1]
	hk = states[...,2]
	psi = states[...,3]

	if t is None: # when solving with DeepVPD or DeepFOC

		if len(states.shape) == 3: # when solving
			T,N = states.shape[:-1]
			y = par.y.reshape(par.T,1).repeat(1,N)

		else:
			T = states.shape[0]
			N = states.shape[1]
			y = par.y.reshape(par.T,1,1).repeat(1,N,train.Nquad)

		T = p.shape[0]

		income = y[t0:T+t0]*p*psi*ell*(1+par.alpha*hk)
			# income = y[:T]*p*psi*ell*(1+par.alpha*hk)

	else:
		income = par.y[t]*p*psi*ell*(1+par.alpha*hk)

	# d. consumption
	m = b*par.R + income
	c = m*(1-savings_rate)


	return torch.stack((c,ell,income),dim=-1)

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
	ell = outcomes[...,1]

	# b. utility
	u = util(c,ell,par)

	# c. finalize
	return u 
	# Case I: shape = (T,...)
	# Case II: shape = (N,)
 

##############
# transition #
##############

def state_trans_pd(model,states,actions,outcomes,t0=0,t=None):
	""" transition to post-decision state """

	# Case I: t is None -> t in t0,...,t0+T-1 <= par.T-1:
	#  states.shape = (T,...,Nstates)
	#  actions.shape = (T,...,Nactions)
	# Case II: t in 0,...,T-1, t0 irrelevant:
	#  states.shape = (N,Nstates)
	#  actions.shape = (N,Nactions)

	par = model.par
	train = model.train

	# a. unpack
	b = states[...,0]
	p = states[...,1]
	hk = states[...,2]
	psi = states[...,3]

	# b. consumption
	c = outcomes[...,0]
	ell = outcomes[...,1]
	income = outcomes[...,2]

	# c. post-decision
	m = b * par.R + income
	
	b_pd = m - c 
	p_pd = p
	hk_pd = hk + ell

	# c. finalize
	states_pd = torch.stack((b_pd,p_pd,hk_pd),dim=-1) 

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

	b_pd = states_pd[...,0]
	p_pd = states_pd[...,1]
	hk_pd = states_pd[...,2]

	xi = shocks[:,0]
	psi = shocks[:,1]

	# b. adjust shape and scale quadrature nodes (when solving)
	if t is None:


		b_pd = expand_to_quad(b_pd,train.Nquad)
		p_pd = expand_to_quad(p_pd,train.Nquad)
		hk_pd = expand_to_quad(hk_pd,train.Nquad)

		xi = expand_to_states(xi,states_pd)
		psi = expand_to_states(psi,states_pd)

	# c. next period

	# i. assets 
	b_plus = b_pd 

	# ii. persistent income
	p_plus = p_pd**par.rho_xi*xi

	# iii. cash-on-hand
	psi_plus = psi # shape = (T,N,Nquad) or (T,N)
	hk_plus = hk_pd
	
	# d. finalize
	states_plus = torch.stack((b_plus,p_plus,hk_plus,psi_plus),dim=-1)
	return states_plus
	# Case I: shape = (T,N,Nquad,Nstates)
	# Case II: shape = (N,Nstates)


###########################
# equations for DeepFOC #
###########################

def eval_equations_FOC(model,states,states_plus,actions,actions_plus,outcomes,outcomes_plus):
	""" evaluate equations for DeepFOC """

	# states.shape = (T,N,Nstates)
	# states_plus.shape = (T,N,Nstates)
	# actions.shape = (T,N,Nactions)
	# actions_plus.shape = (T,N,Nactions)

	par = model.par

	sq_equations = eval_KKT(model,states,states_plus,actions,actions_plus,outcomes,outcomes_plus)

	return sq_equations


def eval_KKT(model,states,states_plus,actions,actions_plus,outcomes,outcomes_plus):
	""" evaluate equations for DeepFOC using KKT conditions """

	par = model.par
	train = model.train
	beta = discount_factor(model,states)

	# a. states at time t
	b = states[...,0]
	p = states[...,1]
	hk = states[...,2]
	psi = states[...,3]

	# b. states at time t+1
	b_plus = states_plus[...,0]
	p_plus = states_plus[...,1]
	hk_plus = states_plus[...,2]
	psi_plus = states_plus[...,3]


	# c. outcomes at time t
	c_t = outcomes[...,0]
	ell_t = outcomes[...,1]
	income_t = outcomes[...,2]

	# d. outcomes at time t+1
	c_plus = outcomes_plus[...,0]
	ell_plus = outcomes_plus[...,1]
	income_plus = outcomes_plus[...,2]

	# e. multiplier at time t
	lambda_t = actions[...,2]

	# f. compute marginal utility at time t
	marg_util_c_t = marg_util_c(c_t)
	marg_util_ell_t = marg_util_ell(ell_t,par)

	# g. compute marginal utility at time t+1
	marg_util_c_plus = marg_util_c(c_plus)
	marg_util_ell_plus = marg_util_ell(ell_plus,par)


	# h. consumption euler equation
	# 1. compute expected marginal utility at time t+1
	exp_marg_util_plus = torch.sum(train.quad_w[None,None,:]*marg_util_c_plus,dim=-1)
	# 2. euler equation
	FOC_c_ = inverse_marg_util(beta[:-1]*par.R*exp_marg_util_plus + lambda_t[:-1])/c_t[:-1]-1
	FOC_c_terminal = torch.zeros_like(FOC_c_[-1])
	FOC_c = torch.cat((FOC_c_,FOC_c_terminal[None,...]),dim=0)

	# i. labor supply euler equation
	lhs_euler_l = marg_util_ell_t + income_t/ell_t * marg_util_c_t
	term1_rhs = marg_util_ell_plus
	term2_rhs = -income_plus * marg_util_c_plus * (par.alpha/(1+par.alpha*hk_plus) - 1/ell_plus)
	rhs_euler_l_expect = torch.sum(train.quad_w[None,None,:]*(term1_rhs + term2_rhs),dim=-1)
	FOC_ell_ = lhs_euler_l[:-1] - beta[:-1]*rhs_euler_l_expect
	FOC_ell_terminal = lhs_euler_l[-1]
	FOC_ell = torch.cat((FOC_ell_,FOC_ell_terminal[None,...]),dim=0)


	# j. borrowing constraint
	constraint = b*par.R + income_t - c_t
	slackness_ = constraint[:-1] * lambda_t[:-1]
	slackness_terminal = constraint[-1]
	slackness = torch.cat((slackness_,slackness_terminal[None,...]),dim=0)

	# k. combine equations
	eq = torch.stack((FOC_c**2,FOC_ell**2,slackness),dim=-1)

	return eq # shape = (T,N,Neq)

