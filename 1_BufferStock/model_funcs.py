import numpy as np
import torch

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

def util_bequest(model,m_pd):

	""" utility of bequest """

	par = model.par

	if par.bequest == 0:
		return torch.zeros_like(m_pd)
	else:
		return par.bequest*torch.log(m_pd)
	
def marg_util_bequest(model,m_pd):
	""" marginal utility of bequest """

	par = model.par

	if par.bequest == 0:
		return torch.zeros_like(m_pd)
	else:
		return par.bequest/m_pd

###########
# reward #
###########

def outcomes(model,states,actions,t=None,t0=0):
	""" outcomes """

	# Case I: t is None
	#  states.shape = (T,...,Nstates)
	#  actions.shape = (T,...,Nactions)
	# Case II:
	#  states.shape = (...,Nstates)
	#  actions.shape = (...,Nactions)

	par = model.par

	m = states[...,0]
	savings_rate = actions[...,0]
	c = (1-savings_rate)*m

	return torch.stack((c,),dim=-1)
	# Case: I: shape = (T,...,Noutcomes)
	# Case: II: shape = (...,Noutcomes)

def reward(model,states,actions,outcomes,t=None,t0=0):
	""" reward """

	# Case I: t is None
	# 	states.shape = (T,...,Nstates)
	# 	actions.shape = (T,...,Nactions)
	# Case II:
	# 	states.shape = (...,Nstates)
	# 	actions.shape = (...,Nactions)

	par = model.par
	train = model.train

	# a. consumption
	c = outcomes[...,0]

	# b. utility
	u = util(c)

	# c. finalize
	return u 
	# Case I: shape = (T,...)
	# Case II: shape = (...,)

def marginal_reward(model,states,actions,outcomes,discount_factor,q_pd,t=None,t0=0):
	""" marginal reward """

	# Case I: t is None
	#  states.shape = (T,...,Nstates)
	#  actions.shape = (T,...,Nactions)
	#  outcomes.shape = (T,...,Noutcomes)
	#  q_pd.shape = (T,...,NFOC_targets)
	# Case II:
	#  states.shape = (...,Nstates)
	#  actions.shape = (...,Nactions)
	#  outcomes.shape = (...,Noutcomes)
	#  q_pd.shape = (...,NFOC_targets)

	# a. consumption
	c = outcomes[...,0]

	# b. finalize
	return marg_util_c(c)
	# Case I: shape = (T,...)
	# Case II: shape = (...,)

def discount_factor(model,states,t=None,t0=0):
	""" discount factor """

	# Case I: t is None
	#  states_pd.shape = (T,...,Nstates_pd)
	# Case II
	#  states_pd.shape = (...,Nstates_pd)

	par = model.par

	beta = par.beta*torch.ones_like(states[...,0])

	return beta 	
	# Case I: shape (T,...)
	# Case II: shape (...,)
	
############
# terminal #
############

def terminal_actions(model,states):
	""" terminal actions """

	# states.shape = (...,Nstates)
	
	par = model.par
	train = model.train
	dtype = train.dtype
	device = train.device

	m = states[...,0]

	# a. standard actions
	actions = (1-(m/(1+par.beta*par.bequest))/m).reshape((*states.shape[:-1],1))

	# b. multipliers
	if par.KKT:
		multipliers = torch.zeros((*states.shape[:-1],1),dtype=dtype,device=device)
		actions = torch.cat((actions,multipliers),dim=-1)

	return actions 
	# shape = (...,Nactions)

def terminal_reward_pd(model,states_pd):
	""" terminal reward """

	# states_pd.shape = (...,Nstates_pd)

	m_pd = states_pd[...,0]
	value_pd = util_bequest(model,m_pd)	
	
	return value_pd 
	# shapes = (...,)

def terminal_marginal_reward_pd(model,states_pd):
	""" terminal reward """

	# states_pd.shape = (...,Nstates_pd)

	m_pd = states_pd[...,0]
	dvalue_m_pd = marg_util_bequest(model,m_pd)
	
	return torch.stack((dvalue_m_pd,),dim=-1) 
	# shapes = (...,NFOC_targets)

##############
# transition #
##############

def state_trans_pd(model,states,actions,outcomes,t=None,t0=0):
	""" transition to post-decision state """

	# Case I: t is None
	#  states.shape = (T,...,Nstates)
	#  actions.shape = (T,...,Nactions)
	#  outcomes.shape = (T,...,Noutcomes)
	# Case II:
	#  states.shape = (...,Nstates)
	#  actions.shape = (..,Nactions)
	#  outcomes.shape = (...,Noutcomes)

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
	# Case II: shape = (...,Nstates_pd)

def state_trans(model,states_pd,shocks,t=None):
	""" state transition with quadrature """

	# Case I: t is None (solving simultaneously)
	#  states_pd.shape = (T,N,Nnumint,Nstates_pd)
	#  shocks.shape = (T,N,Nnumint,Nshocks)
	# Case IIa: Simulating
	#  states_pd.shape = (N,Nstates_pd)
	#  shocks.shape = (N,Nshocks)
	# Case IIb: Solving backwards
	#  states_pd.shape = (N,Nnumint,Nstates_pd)
	#  shocks.shape = (N,Nnumint,Nshocks)

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

	xi = shocks[...,0]
	psi = shocks[...,1]

	# b. adjust shape and scale quadrature nodes
	xi = torch.exp(sigma_xi*xi-0.5*sigma_xi**2)
	psi = torch.exp(sigma_psi*psi-0.5*sigma_psi**2)

	# c. next period

	# i. persistent income
	p_plus = p_pd**rho_p*xi

	# ii. actual income
	if t is None: # when solving simultaneously

		kappa = par.kappa[:,None,None] # (T,) -> (T,1,1) to match (T,N,Nnumint)

		if par.T_retired == par.T:
			y_plus = kappa[:par.T-1]*p_plus*psi
		else:
			y_plus_before = kappa[:par.T_retired]*p_plus[:par.T_retired]*psi[:par.T_retired]
			y_plus_after = kappa[par.T_retired] * torch.ones_like(p_plus[par.T_retired:])
			y_plus = torch.cat((y_plus_before,y_plus_after),dim=0)
	
	else: # when simulating or solving backwards

		if t < par.T_retired:
			y_plus = par.kappa[t]*p_plus*psi
		else:
			y_plus = par.kappa[t] * torch.ones_like(p_plus)

	# iii. cash-on-hand
	m_plus = par.R*m_pd + y_plus # shape = (T,N,Nnumint) or (T,N)
	
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
	# Case I: shape = (T,...,Nstates)
	# Case II: shape = (...,Nstates)

def exploration(model,states,actions,eps,t=None):

	# Case I: t is None 
	#  states.shape = (T,...,Nstates)
	#  actions.shape = (T,...,Nactions)
	#  eps.shape = (T,...,Nactions)
	# Case II:
	#  states.shape = (...,Nstates)
	#  actions.shape = (...,Nactions)
	#  eps.shape = (...,Nactions)

	return actions + eps
	# Case I: shape = (T,...,Nactions)
	# Case II: shape = (...,Nactions)

###########################
# equations for DeepFOC #
###########################

def eval_equations_FOC(model,states,states_plus,actions,actions_plus,outcomes,outcomes_plus):
	""" evaluate equations for DeepFOC """

	# states.shape = (T,N,Nstates)
	# states_plus.shape = (T,N,Nnumint,Nstates)
	# actions.shape = (T,N,Nactions)
	# actions_plus.shape = (T,N,Nnumint,Nactions)
	# outcomes.shape = (T,N,Noutcomes)
	# outcomes_plus.shape = (T,N,Nnumint,Noutcomes)

	par = model.par

	if par.KKT:
		sq_equations = eval_KKT(model,states,states_plus,actions,actions_plus,outcomes,outcomes_plus)
	else:
		sq_equations = eval_burmeister(model,states,states_plus,actions,actions_plus,outcomes,outcomes_plus)

	return sq_equations

def eval_burmeister(model,states,states_plus,actions,actions_plus,outcomes,outcomes_plus):
	""" evaluate equations for DeepFOC using Fischer Burmeister function """

	par = model.par
	train = model.train

	# a. compute consumption at time t
	c_t = outcomes[...,0]

	# b. compute consumption at time t+1
	c_tplus = outcomes_plus[...,0]

	# c. compute marginal utility at time t+1
	marg_util_tplus = marg_util_c(c_tplus)

	# d. compute expected marginal utility at time t+1
	if marg_util_tplus.ndim == 3:
		numint_weights = train.numint_weights[None,None,:]
	else: # used in special case (uncodumented)
		assert marg_util_tplus.ndim == 2
		numint_weights = train.numint_weights[None,:]

	exp_marg_util_t1 = torch.sum(numint_weights*marg_util_tplus,dim=-1)
	
	# e. euler equation
	beta = discount_factor(model,states)
	FOC = inverse_marg_util(beta*par.R*exp_marg_util_t1)/c_t-1
	
	# f. borrowing constraint
	savings_rate = actions[...,0]
	constraint = savings_rate # >= 0

	# g. combine with fischer burmeister
	eq = fischer_burmeister(FOC,constraint)**2

	return torch.stack((eq,),dim=-1) # shape = (T,N) -> (T,N,1)

def eval_KKT(model,states,states_plus,actions,actions_plus,outcomes,outcomes_plus):
	""" evaluate equations for DeepFOC using KKT conditions """

	par = model.par
	train = model.train

	# a. compute consumption and multiplier at time t
	c_t = outcomes[...,0]
	multiplier_t = actions[...,1]

	# b. compute consumption at time t+1
	c_tplus = outcomes_plus[...,0]

	# c. compute marginal utility at time t+1
	marg_util_tplus = marg_util_c(c_tplus)

	# d. compute expected marginal utility at time t+1
	if marg_util_tplus.ndim == 3:
		numint_weights = train.numint_weights[None,None,:]
	else: # used in special case (uncodumented)
		assert marg_util_tplus.ndim == 2
		numint_weights = train.numint_weights[None,:]

	exp_marg_util_t1 = torch.sum(numint_weights*marg_util_tplus,dim=-1)
	
	# e. compute euler equation
	beta = discount_factor(model,states)
	FOC = inverse_marg_util(beta*par.R*exp_marg_util_t1+multiplier_t)/c_t-1
	
	# f. borrowing constraint (slackness condition)
	savings_rate = actions[...,0]
	slackness = savings_rate*multiplier_t # <= 0

	# g. combine
	return torch.stack((FOC**2,slackness),dim=-1) # shape = (T,N,2)

def eval_equations_FOC_terminal(model,states,actions,outcomes,states_pd):
	""" evaluate equations for DeepFOC with FOC in terminal period """

	# states.shape = (1,N,Nstates)
	# actions.shape = (1,N,Nactions)
	# outcomes.shape = (1,N,Noutcomes)
	# states_pd.shape = (1,N,Nsates_pd)

	par = model.par

	m_pd = states_pd[...,0]

	if par.KKT:

		# a. compute consumption
		c_t = outcomes[...,0]
		multiplier_t = actions[...,1]

		# b. marginal utility of bequest
		marg_util_bequest_ = marg_util_bequest(model,m_pd)

		# c. FOC
		beta = discount_factor(model,states)
		FOC = inverse_marg_util(beta*marg_util_bequest_+multiplier_t)/c_t-1

		# d. borrowing constraint (slackness condition)
		savings_rate = actions[...,0]
		slackness = savings_rate*multiplier_t # <= 0

		# e. combine
		eq = torch.stack((FOC**2,slackness),dim=-1) # shape = (1,N,2)

		return eq

	else:
		
		# a. compute consumption
		c_t = outcomes[...,0]

		# b. marginal utility of bequest
		marg_util_bequest_ = marg_util_bequest(model,m_pd)

		# c. FOC
		beta = discount_factor(model,states)
		FOC = inverse_marg_util(beta*marg_util_bequest_)/c_t-1

		# d. borrowing constraint
		savings_rate = actions[...,0]
		constraint = savings_rate # >= 0

		# e. combine with fischer burmeister
		eq = fischer_burmeister(FOC,constraint)**2

		return torch.stack((eq,),dim=-1) # shape = (1,N) -> (1,N,1)

#########################
# equations for DeepVPD # 
#########################

def eval_equations_VPD(model,states,actions,outcomes,states_pd,q_pd):
	""" evaluate equation for DeepVPD with FOC """

	# states.shape = (T,N,Nstates)
	# actions.shape = (T,N,Nactions)
	# outcomes.shape = (T,N,Noutcomes)
	# states_pd.shape = (T,N,Nsates_pd)

	par = model.par
	
	# a. compute consumption at time t
	c_t = outcomes[...,0]

	# b. compute marginal utility at time t
	marg_util_t = marg_util_c(c_t)

	# c. compute euler equation
	beta = discount_factor(model,states)
	FOC = marg_util_t/(beta*par.R*q_pd[...,0])-1

	# d. constraint
	savings_rate = actions[...,0]
	constraint = savings_rate

	# e. combine with fischer burmeister
	eq = fischer_burmeister(FOC,constraint)**2

	return eq

def eval_equations_VPD_terminal(model,states,actions,outcomes,states_pd):
	""" evaluate equation for DeepVPD with FOC in terminal period """

	# states.shape = (1,N,Nstates)
	# actions.shape = (1,N,Nactions)
	# outcomes.shape = (1,N,Noutcomes)
	# states_pd.shape = (1,N,Nsates_pd)

	par = model.par
	
	# a. compute consumption at time t
	c_t = outcomes[...,0]

	# b. compute marginal utility at time t
	marg_util_t = marg_util_c(c_t)

	# c. marginal utility of bequest
	m_pd = states_pd[...,0]
	marg_util_bequest_ = marg_util_bequest(model,m_pd)

	# c. compute euler equation
	beta = discount_factor(model,states)
	FOC = marg_util_t/(par.beta*marg_util_bequest_)-1

	# d. constraint
	savings_rate = actions[...,0]
	constraint = savings_rate

	# e. combine with fischer burmeister
	eq = fischer_burmeister(FOC,constraint)**2

	return eq

def eval_equations_FOC_t(model,states,states_plus,states_pd,actions,actions_plus,outcomes,outcomes_plus,t):

	sq_equations = eval_equations_FOC(model,states,states_plus,actions,actions_plus,outcomes,outcomes_plus)

	return sq_equations

##############
# auxilliary #
##############

def fischer_burmeister(a,b):
	""" Fischer-Burmeister function """
	 
	return torch.sqrt(a**2 + b**2) - a - b