import numpy as np
import torch

from EconDLSolvers import expand_to_quad, expand_to_states

######################
# budget constraint #
######################

def adj_cost(par,Delta):
	""" Adjustment cost """

	return par.nu * (Delta)**2

def d_adj_cost(par,Delta):
	""" Derivative of adjustment cost """

	return 2*par.nu*Delta

def d_consume_all(tau,m,epsilon):
	""" Computation maximum investment level if the consumer spend all cash-on-hand except for epsilon """

	if tau == 0.0:
	
		Delta_max = m - epsilon # if no adjustment cost problem is linear
	
	else:

		a = (-tau)
		b = -1
		c = m - epsilon
		denominator = 2*a
		numerator = (-b - (b**2-4*a*c)**(1/2))
		Delta_max = numerator/denominator

	return Delta_max

def compute_d_and_c_from_action(par,state,action,eps=None):
	""" Compute c,d,m_pd from state and action"""

	# a. make array for d
	array_shape =  state.shape[:-1] + (par.D,)
	if type(state) is np.ndarray:
		array_shape =  state.shape[:-1] + (par.D,)
		d_array = np.zeros(array_shape)
	else:
		d_array = torch.zeros(array_shape,dtype=state.dtype,device=state.device)

	# b. consumption choice
	
	# i. add noise if relevant
	if eps is None:
		action_c = action[...,0]
	else:
		if type(state) is np.ndarray:
			action_c = np.clip(action[...,0] + eps[...,0],0,0.9999)
		else:
			action_c = torch.clamp(action[...,0] + eps[...,0],0,0.9999)
	# ii. compute consumption
	m = state[...,0]
	c = m*(1-action_c)
	
	# iii. compute wealth after consumption
	mbar = m - c

	# c. loop over durables to get durable choices
	for i_d in range(par.D):

		# i. compute bounds
		d_low = 0.0
		if par.nonnegative_investment:
			d_low = state[...,2+i_d]

		d_high = state[...,2+i_d] + d_consume_all(par.nu,mbar,0.0) # current level + maximum investment level

		# ii. add noise to action if needed
		if eps is None:
			action_d = action[...,1+i_d]
		else:
			if type(state) is np.ndarray:
				action_d = np.clip(action[...,1+i_d]+eps[...,1+i_d],0,0.9999)
			else:
				action_d = torch.clamp(action[...,1+i_d]+eps[...,1+i_d],0,0.9999)
		
		# iii. compute durable consumption
		d = d_low+(d_high-d_low)*action_d

		# iv. store durable consumption
		d_array[...,i_d] = d

		# v. compute wealth after durable consumption
		Delta_d = d - state[...,2+i_d]
		mbar = mbar - Delta_d - adj_cost(par,Delta_d)
	

	# e. set m_pd as mbar
	m_pd = mbar

	return c,d_array,m_pd

##########
# reward #
##########

def outcomes(model,states,actions,t0=0,t=None):

	# a. unpack
	par = model.par
	
	# get c, d and m_pd
	c,d_array,m_pd = compute_d_and_c_from_action(par,states,actions,eps=None)

	# b. combine
	return torch.cat((c[...,None],d_array,m_pd[...,None]),dim=-1)

def utility(c,d,par,train):
	""" utility """

	# a. compute durable utility
	d_util = d + par.d_ubar
	d_power = par.omega
	dexp = d_util**d_power[None,:]
	dprod_util = torch.prod(dexp,dim=-1)

	# b. compute consumption utility
	c_power =(1 - torch.sum(par.omega))
	c_util = c**c_power

	# c. combine
	util_fac = c_util * dprod_util

	return util_fac**(1 - par.rho) / (1 - par.rho)

def marg_util_c(c,d,par,train):
	""" marginal utility of consumption	"""

	# a. compute non-durable utility
	c_power = (1-torch.sum(par.omega)) * (1-par.rho) -1
	c_util = c**c_power

	# b. compute durable utility
	d_power = par.omega * (1-par.rho)
	d_tot = d + par.d_ubar
	d_util = torch.prod(d_tot**d_power,dim=-1)
	factor = 1 - torch.sum(par.omega)

	return factor * d_util * c_util

def inv_marg_util_c(marg_util_c,d,par,train):
	""" inverse of marginal utility of consumption with respect to consumption"""
	
	# a. compute durable utility
	d_power = par.omega * (1-par.rho)
	d_tot = d + par.d_ubar
	d_util = torch.prod(d_tot**d_power,dim=-1)
	factor = 1 - torch.sum(par.omega)

	# b. compute non-durable utility
	c_power = (1-torch.sum(par.omega)) * (1-par.rho) -1
	c_util = marg_util_c / (factor * d_util)

	return c_util**(1/c_power)

def marg_util_d(c,d,par,train):
	""" marginal utility of durable consumption for each durable"""

	choice_shape = d.shape[:-1]

	# a. initialize result_array
	result = torch.zeros_like(d)

	# b. get part derived from consumption
	c_power = (1-torch.sum(par.omega)) * (1-par.rho)
	c_util = (c**c_power).reshape(choice_shape)

	# c. get part derived from durable consumption
	for i in range(par.D):

		# i. durable wrt to which we take the derivative
		d_power_i = par.omega[i] * (1-par.rho) - 1
		dtot_i = d[...,i] + par.d_ubar[i]
		d_util_i = (dtot_i**d_power_i).reshape(choice_shape)
		factor = par.omega[i]

		# ii. remaining durables
		if par.D > 1:

			# o. masks to remove the i-th durable
			mask_i = torch.ones_like(par.omega,dtype=torch.bool)
			mask_i[i] = 0
			
			# oo. compute utility part without d_i
			d__i = d[...,mask_i]
			omega__i = par.omega[mask_i].reshape(1,par.D-1)
			d_ubar__i = par.d_ubar[mask_i].reshape(1,par.D-1)
			d_power__i = omega__i * (1-par.rho)
			dtot__i = d__i + d_ubar__i
			dutil__i = torch.prod(dtot__i**d_power__i,dim=-1).reshape(choice_shape)

		else:

			dutil__i = 1

		# iii. store result
		result[...,i] = factor * c_util * d_util_i * dutil__i

	return result

def reward(model,states,actions,outcomes,t0=0,t=None):
	""" reward """

	par = model.par
	train = model.train

	# a. get c and d
	c = outcomes[...,0]
	d = outcomes[...,1:par.D+1]

	# b. compute reward
	return utility(c,d,par,train) # shape = (T,...)

def marginal_reward(model,states,actions,outcomes,t0=0,t=None):
	""" marginal reward """

	par = model.par
	train = model.train

	# a. get c and d
	c = outcomes[...,0]
	d = outcomes[...,1:par.D+1]
	Delta = d - states[...,2:par.Nstates]
	lambdaa = actions[...,par.D+1]
	mu = actions[...,par.D+2:par.D+2+par.D]

	# b. compute marginal utility of consumption
	marg_util_c_t = marg_util_c(c,d,par,train)
	dvpd_dmpd = par.R * marg_util_c_t

	if train.NFOC_targets == 1: # only use consumption euler
		return dvpd_dmpd
	else: # use all FOCs
		term_1 = (1+d_adj_cost(par,Delta)) * marg_util_c_t[...,None]
		term_2 = - mu
		dvpd_npd = (1-par.delta) * (term_1 + term_2)
		
		return torch.cat((dvpd_dmpd[...,None],dvpd_npd),dim=-1)

def marginal_terminal_reward(model,states_pd):
	""" marginal reward """

	train = model.train
	h = torch.zeros((*states_pd.shape[:-1],train.NFOC_targets),dtype=train.dtype,device=states_pd.device)

	return h

def discount_factor(model,states,t0=0,t=None):
	""" discount factor """

	par = model.par

	beta = par.beta * torch.ones_like(states[...,0])
	return beta # shape (T,...)
	
############
# terminal #
############

def terminal_actions(model,states):
	""" terminal actions """

	raise NotImplementedError("This function is not implemented in the durable model")

def terminal_reward_pd(model,states_pd):
	""" terminal reward """

	train = model.train
	h = torch.zeros((*states_pd.shape[:-1],1),dtype=train.dtype,device=states_pd.device)

	return h

##############
# transition #
##############

def state_trans_pd(model,states,actions,outcomes,t0=0,t=None):
	""" transition to post-decision state """

	par = model.par

	# a. get p
	p = states[...,1]
	
	# b. get d and  a
	d = outcomes[...,1:par.D+1]
	m_pd = outcomes[...,-1]

	# c. post-dec state
	m_pd = m_pd
	p_pd = p
	n_pd_arr = d
	
	# d. combine
	states_pd = torch.cat((m_pd[...,None],p_pd[...,None],n_pd_arr),dim=-1)
	return states_pd

def state_trans(model,states_pd,quad,t=None):
	""" state transition with quadrature """

	# states_pd.shape = (T,N,Nstates_pd)
	# quad.shape = (Nquad,Nshocks)
	par = model.par
	train = model.train

	# a. unpack
	m_pd = states_pd[...,0]
	p_pd = states_pd[...,1]
	n_pd_arr = states_pd[...,2:par.Nstates]
	xi = quad[:,0]
	psi = quad[:,1]
	
	# b. adjust shape
	if t is None:

		T,N = states_pd.shape[:-1]
		m_pd = expand_to_quad(m_pd,train.Nquad)
		p_pd = expand_to_quad(p_pd,train.Nquad)
		n_pd_arr = expand_to_quad(n_pd_arr,train.Nquad)

		xi = expand_to_states(xi,states_pd)
		psi = expand_to_states(psi,states_pd)
		delta = expand_to_states(par.delta,states_pd) # expand from (D,) to (T,N,D)
		delta = expand_to_quad(delta,train.Nquad) # expand from (T,N,D) to (T,N,Nquad,D)
		kappa = par.kappa.reshape(par.T,1,1).repeat(1,N,train.Nquad)

	else:

		delta = par.delta 

	# c. next period
	
	# i. persistent income

	p_plus = p_pd**par.rho_p*xi

	# ii.actual income
	if t is None:
		if par.T_retired == par.T:
			y_plus = kappa[:par.T-1]*p_plus*psi
		else:
			y_plus_before = kappa[:par.T_retired]*p_plus[:par.T_retired]*psi[:par.T_retired]
			y_plus_after = par.kappa[par.T_retired] * torch.ones_like(p_plus[par.T_retired:])
			y_plus = torch.cat((y_plus_before,y_plus_after),dim=0)
	else:
		if t < par.T_retired:
			y_plus = par.kappa[t]*p_plus*psi
		else:
			y_plus = par.kappa[t]

	# iii. cash-on-hand
	m_plus = par.R*m_pd + y_plus # shape = (T,N,Nquad) or (1,N,Nquad)

	# e. durable
	n_plus = n_pd_arr * (1 - delta)
	
	# f. combine
	states_plus = torch.cat((m_plus[...,None],p_plus[...,None],n_plus),dim=-1)
	return states_plus

def exploration(model,states,actions,eps,t=None):
	return actions + eps

#########################
# Equations for DeepFOC #
#########################

def eval_equations_FOC(model,states,states_plus,actions,actions_plus,outcomes,outcomes_plus):
	""" evaluate equations for DeepFOC """

	par = model.par

	if par.KKT:
		sq_equations = eval_KKT(model,states,states_plus,actions,actions_plus,outcomes,outcomes_plus)
	else:
		raise NotImplementedError("KKT must be True in the durable model when using DeepFOC")

	return sq_equations

def eval_KKT(model, states, states_plus, actions, actions_plus,outcomes,outcomes_plus):
	""" evaluate KKT conditions"""

	par = model.par
	train = model.train
	device = states.device

	# a. compute d and c at time t
	c = outcomes[...,0]
	d = outcomes[...,1:par.D+1]
	m_pd = outcomes[...,-1] # post-decision wealth
	Delta = d - states[...,2:par.Nstates]

	# b. get multipliers
	lambda_multiplier_t = actions[...,par.D+1] # borrowing constraint multiplier
	mu_multiplier_t = actions[...,par.D+2:par.D+2+par.D] # durable constraint multiplier
	mu_multiplier_tplus = actions_plus[...,par.D+2:par.D+2+par.D] # future durable constraint multiplier

	# c. compute marginal utility at time t
	marg_util_c_t = marg_util_c(c,d,par,train)
	marg_util_d_t = marg_util_d(c,d,par,train)

	# d. compute d and c at time t+1
	c_plus = outcomes_plus[...,0]
	d_plus = outcomes_plus[...,1:par.D+1]
	Delta_plus = d_plus - states_plus[...,2:par.Nstates]

	# e. compute marginal utility at time t+1
	marg_util_c_plus = marg_util_c(c_plus,d_plus,par,train)

	# f. non-durable consumption euler
	beta = discount_factor(model, states) # get discount factor
	exp_marg_util = torch.sum(train.quad_w[None,None,:]*marg_util_c_plus[:par.T-1], dim=-1) # expected marginal utility at t+1
	euler_c_ = (marg_util_c_t[:par.T-1] - lambda_multiplier_t[:par.T-1]) / (beta[:par.T-1]*par.R*exp_marg_util) - 1 # euler equation for consumption

	# pad terminal period as optimal savings rate in c is known ex-ante
	euler_c_pad = torch.zeros_like(euler_c_[-2:-1])
	euler_c = torch.cat((euler_c_,euler_c_pad),dim=0)

	# g. durable consumption euler
	N_dim = actions.shape[1]
	euler_d = torch.zeros((par.T,N_dim,par.D),dtype=train.dtype,device=device)
	for i_d in range(par.D):
		# a. left hand side of euler equation for durable
		lhs_d =  marg_util_d_t[...,i_d]
		
		# b. right hand side of euler equation for durable
		rhs_d_1 = (1 + d_adj_cost(par,Delta[...,i_d])) * marg_util_c_t # first term in rhs
		rhs_d_2_quad = (1 + d_adj_cost(par,Delta_plus[...,i_d])) * (marg_util_c_plus)  # part of second term in rhs - across quadrature points
		rhs_d_2 = beta[:par.T-1] * (1 - par.delta[i_d]) * torch.sum(train.quad_w[None,None,:]*rhs_d_2_quad[:par.T-1], dim=-1) # second term in rhs
		rhs_d_3 = beta[:par.T-1] * (1 - par.delta[i_d]) * torch.sum(train.quad_w[None,None,:]*mu_multiplier_tplus[:par.T-1,...,i_d], dim=-1) # third term in rhs
		rhs_d_4 = mu_multiplier_t[:par.T-1,...,i_d] # fourth term in rhs
		rhs_d = rhs_d_1[:par.T-1] - rhs_d_2[:par.T-1] + rhs_d_3[:par.T-1] - rhs_d_4[:par.T-1] # combined rhs
		euler_d_b = lhs_d[:par.T-1] - rhs_d[:par.T-1] # euler equation for durable - before terminal period
		euler_d_a = lhs_d[par.T-1:par.T] - ((1 + d_adj_cost(par,Delta[par.T-1:par.T,...,i_d])) * marg_util_c_t[par.T-1:par.T] - mu_multiplier_t[par.T-1:par.T,...,i_d]) # euler equation for durable - terminal period
		euler_d[...,i_d] = torch.cat((euler_d_b,euler_d_a),dim=0) # combine

	# h. slackness constraint non-durable consumption
	slackness_c_ = lambda_multiplier_t[:par.T-1] * m_pd[:par.T-1]
	slacness_c_terminal = m_pd[par.T-1:par.T] # no savings left in terminal period
	slackness_c = torch.cat((slackness_c_,slacness_c_terminal),dim=0)

	# i. slackness constraint durable consumption
	slackness_d_ = mu_multiplier_t * Delta
	slackness_d = slackness_d_

	# k. combine

	eq = torch.cat((euler_c[...,None]**2,euler_d**2,slackness_c[...,None],slackness_d),dim=-1)
	
	return eq

#########################
# Equations for DeepVPD # 
#########################

def eval_equations_VPD(model,states,action,dvalue_pd):
	""" evaluate equation for DeepVPD with FOC """

	par = model.par
	train = model.train
	
	# a. compute consumption at time t
	outcomes = model.outcomes(states,action)
	c_t = outcomes[...,0]
	d_t = outcomes[...,1:par.D+1]
	Delta_t = d_t - states[...,2:par.Nstates]
	m_pd = outcomes[...,-1]
	lambda_t = action[...,par.D+1]
	mu = action[...,par.D+2:par.D+2+par.D]
	beta = par.beta
	dvpd_dmpd = dvalue_pd[...,0].clamp(0.0,100000.0)
	dvpd_dnpd = dvalue_pd[...,1:1+par.D].clamp(0.0,100000.0)


	# b. compute marginal utility at time t
	marg_util_c_t = marg_util_c(c_t,d_t,par,model.train)

	# c. compute non-durable euler equation
	FOC_c_ = (marg_util_c_t-lambda_t)/(beta*dvpd_dmpd)-1
	FOC_c_pad = torch.zeros_like(FOC_c_[-2:-1])
	FOC_c = torch.cat((FOC_c_[:par.T-1],FOC_c_pad),dim=0)

	# d. borrowing constraint
	bor_constraint_ = m_pd[:par.T-1] * lambda_t[:par.T-1]
	bor_constraint_terminal = m_pd[par.T-1:par.T]
	bor_constraint = FOC_c = torch.cat((bor_constraint_,bor_constraint_terminal),dim=0)

	# e. compute durable euler
	# FOC_d = torch.zeros((par.T,par.N,par.D),dtype=model.train.dtype,device=states.device)
	if train.NFOC_targets > 1:
		FOC_d = torch.zeros((states.shape[0],states.shape[1],par.D),dtype=model.train.dtype,device=states.device)
		# i. left hand side of euler equation for durable
		lhs_d = marg_util_d(c_t,d_t,par,train)
		for i_d in range(par.D):
			# ii. right hand side of euler equation for durable
			rhs_term1 = (1 + d_adj_cost(par,Delta_t[...,i_d])) * marg_util_c_t # first term in rhs
			rhs_term2 =  beta * dvpd_dnpd[...,i_d] # second term in rhs
			rhs_term3 = mu[...,i_d] # third term in rhs
			rhs_d = rhs_term1 - rhs_term2 - rhs_term3 # combined rhs

			FOC_d_b = lhs_d[:par.T-1,:,i_d] - rhs_d[:par.T-1] # euler equation for durable - before terminal period
			FOC_d_a = lhs_d[par.T-1:par.T,:,i_d] - (rhs_term1[par.T-1:par.T]-rhs_term3[par.T-1:par.T])
			FOC_d[...,i_d] = torch.cat((FOC_d_b,FOC_d_a),dim=0)
		
		# f. durable constraint
		dur_constraint = Delta_t * mu

	# g. compute equation errors
	if train.NFOC_targets == 1:
		weight = 1 / ( 2 + 1) 
		eq = weight * (2 * FOC_c[...,None]**2 + bor_constraint[...,None])
	else:
		weight = 1 / ( 2 + 1 + 2*par.D + par.D)
		eq = weight * (2 * FOC_c[...,None]**2 + bor_constraint[...,None] + 2*FOC_d**2 + dur_constraint)

	return eq

##############
# Auxilliary #
##############

def fischer_burmeister(a,b):
	""" Fischer-Burmeister function - torch """
	 
	return torch.sqrt(a**2 + b**2) - a - b

##################
# numpy versions #
##################

def utility_np(c,d,par,train):
	""" utility - numpy version """

	# a. compute durable utility
	d_util = d + par.d_ubar
	d_power = par.omega
	dexp = d_util**d_power[None,:]
	dprod_util = np.prod(dexp,axis=-1)

	# b. compute consumption utility
	c_power =(1 - np.sum(par.omega))
	c_util = c**c_power 

	# c. combine
	util_fac = c_util * dprod_util
	return util_fac**(1 - par.rho) / (1 - par.rho)

def inv_marg_util_c_np(marg_util_c,d,par,train):
	""" inverse of marginal utility of consumption with respect to consumption - numpy version"""

	# a. compute durable utility
	d_power = par.omega * (1-par.rho)
	d_tot = d + par.d_ubar
	d_util = np.prod(d_tot**d_power,axis=-1)
	factor = 1 - np.sum(par.omega)

	# b. compute non-durable utility
	c_power = (1-np.sum(par.omega)) * (1-par.rho) -1
	c_util = marg_util_c / (factor * d_util)

	return c_util**(1/c_power)

def marg_util_c_np(c,d,par,train):
	""" marginal utility of consumption - numpy version
		c_shape = (T,N,...)
		d_shape = (T,N,D,...)
	"""

	# a. compute non-durable utility
	c_power = (1-np.sum(par.omega)) * (1-par.rho) -1
	c_util = c**c_power

	# b. compute durable utility
	d_power = par.omega * (1-par.rho)
	d_tot = d + par.d_ubar
	d_util = np.prod(d_tot**d_power,axis=-1)
	factor = 1 - np.sum(par.omega)

	return factor * d_util * c_util