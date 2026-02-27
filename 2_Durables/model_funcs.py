import numpy as np
import torch

######################
# budget constraint #
######################

def adj_cost(par,Delta):
	""" Adjustment cost """

	return par.nu*(Delta)**2

def d_adj_cost(par,Delta):
	""" Derivative of adjustment cost """

	return 2*par.nu*Delta

def d_consume_all(nu,m,epsilon):
	""" Computation maximum investment level if the consumer spend all cash-on-hand except for epsilon """

	if nu == 0.0:
	
		Delta_max = m - epsilon # if no adjustment cost problem is linear
	
	else:

		a = (-nu)
		b = -1
		c = m - epsilon
		denominator = 2*a
		numerator = (-b-(b**2-4*a*c)**(1/2))
		Delta_max = numerator/denominator

	return Delta_max

def compute_d_and_c_from_action(par,state,action):
	""" Compute c,d,m_pd from state and action"""

	# a. make array for d
	array_shape =  state.shape[:-1] + (par.D,)
	if type(state) is np.ndarray:
		array_shape =  state.shape[:-1] + (par.D,)
		d_array = np.zeros(array_shape)
	else:
		d_array = torch.zeros(array_shape,dtype=state.dtype,device=state.device)

	# b. consumption choice
	
	# i. compute consumption
	m = state[...,0]
	c = m*(1-action[...,0])
	
	# ii. compute wealth after consumption
	mbar = m - c

	# c. loop over durables to get durable choices
	for i_d in range(par.D):

		# i. compute bounds
		d_low = 0.0		
		if par.nonnegative_investment:
			d_low = state[...,2+i_d]

		d_high = state[...,2+i_d] + d_consume_all(par.nu,mbar,0.0) # current level + maximum investment level

		# ii. compute durable consumption
		d = d_low+(d_high-d_low)*action[...,1+i_d]

		# iv. store durable consumption
		d_array[...,i_d] = d

		# v. compute wealth after durable consumption
		Delta_d = d - state[...,2+i_d]
		mbar = mbar - Delta_d - adj_cost(par,Delta_d)
	
	# d. set m_pd as mbar
	m_pd = mbar

	return c,d_array,m_pd

def invert_expenditures(par,exp,action,durable_stock):
	""" Invert expenditures to get action """

	Delta = exp = (-1 + torch.sqrt(1 + 4*par.nu*exp)) / (2*par.nu)
	d = durable_stock + Delta

	return d

def compute_d_and_c_from_action_softmax(par,state,action):
	""" Compute c,d,m_pd from state and action"""

	# a. make array for d
	array_shape =  state.shape[:-1] + (par.D,)
	if type(state) is np.ndarray:
		array_shape =  state.shape[:-1] + (par.D,)
		d_array = np.zeros(array_shape)
	else:
		d_array = torch.zeros(array_shape,dtype=state.dtype,device=state.device)

	# b. spending shares
	m = state[...,0]
	if par.KKT:
		spending_shares = action[...,:par.D+2]
	else:
		spending_shares = action

	# c. consumption choice
	c = m*spending_shares[...,0]

	# d. savings_choice
	m_pd = m * spending_shares[...,1]

	# e. loop over durables to get durable choices
	for i_d in range(par.D):
		exp = spending_shares[...,2+i_d] * m
		d_array[...,i_d] = invert_expenditures(par,exp,action[...,2+i_d],state[...,2+i_d])

	return c,d_array,m_pd

##########
# reward #
##########

def outcomes(model,states,actions,t=None,t0=0):

	# a. unpack
	par = model.par
	
	# get c, d and m_pd
	if par.use_softmax:
		c,d_array,m_pd = compute_d_and_c_from_action_softmax(par,states,actions)
	else:
		c,d_array,m_pd = compute_d_and_c_from_action(par,states,actions)

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
	c_power = (1 - torch.sum(par.omega))
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

def reward(model,states,actions,outcomes,t=None,t0=0):
	""" reward """

	par = model.par
	train = model.train

	# a. get c and d
	c = outcomes[...,0]
	d = outcomes[...,1:par.D+1]

	# b. compute reward
	return utility(c,d,par,train) # shape = (T,...)

def marginal_reward(model,states,actions,outcomes,discount_factor,q_pd,t=None,t0=0):
	""" marginal reward """

	par = model.par
	train = model.train

	# a. unpack
	n = states[...,2:par.Nstates]

	mu_mult = actions[...,par.D+2:par.D+2+par.D]
	
	c = outcomes[...,0]
	d = outcomes[...,1:par.D+1]
	Delta = d - n

	# b. marginal utility
	marg_util_c_t = marg_util_c(c,d,par,train)
	
	# c. dmpd
	dvpd_dmpd = par.R*marg_util_c_t

	# d. dnpd
	dvpd_dnpd = torch.zeros((*states.shape[:-1],par.D),dtype=model.train.dtype,device=states.device)

	for i_d in range(par.D):

		d_adj_cost_ = d_adj_cost(par,Delta[...,i_d])

		term_1 = (1+d_adj_cost_)*marg_util_c_t
		term_2 = mu_mult[...,i_d]

		dvpd_dnpd[...,i_d] = (1-par.delta[i_d])*(term_1-term_2)

	return torch.cat((dvpd_dmpd[...,None],dvpd_dnpd),dim=-1)

def discount_factor(model,states,t=None,t0=0):
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
	return torch.zeros((*states_pd.shape[:-1],),dtype=train.dtype,device=states_pd.device)

def terminal_marginal_reward_pd(model,states_pd):
	""" marginal reward """

	train = model.train
	return torch.zeros((*states_pd.shape[:-1],train.NFOC_targets),dtype=train.dtype,device=states_pd.device)

##############
# transition #
##############

def state_trans_pd(model,states,actions,outcomes,t=None,t0=0):
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

def state_trans(model,states_pd,shocks,t=None):
	""" state transition with quadrature """

	par = model.par
	train = model.train

	# a. unpack
	m_pd = states_pd[...,0]
	p_pd = states_pd[...,1]
	n_pd_arr = states_pd[...,2:par.Nstates]
	
	xi = shocks[...,0]
	psi = shocks[...,1]
	
	# b. adjust shape
	if t is None:
		kappa = par.kappa[:,None,None] # (T,) -> (T,1,1) to match (T,N,Nnumint)
		delta = par.delta[None,None,None,:] # (D,) -> (1,1,1,D) to match (T,N,Nnumint,D)

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
			y_plus = par.kappa[t] * torch.ones_like(p_plus)

	# iii. cash-on-hand
	m_plus = par.R*m_pd + y_plus # shape = (T,N,Nnumint) or (1,N,Nnumint)

	# e. durable
	n_plus = n_pd_arr * (1 - delta) * torch.ones_like(p_plus[...,None])
	
	# f. combine
	states_plus = torch.cat((m_plus[...,None],p_plus[...,None],n_plus),dim=-1)
	return states_plus

def exploration(model,states,actions,eps,t=None):
	return actions + eps

#########################
# equations for DeepFOC #
#########################

def eval_equations_FOC(model,states,states_plus,actions,actions_plus,outcomes,outcomes_plus):
	""" evaluate equations for DeepFOC """

	par = model.par

	if par.KKT:
		sq_equations = eval_KKT(model,states,states_plus,actions,actions_plus,outcomes,outcomes_plus)
	else:
		raise NotImplementedError("KKT must be True in the durable model when using DeepFOC")

	return sq_equations

def eval_KKT(model,states,states_plus,actions,actions_plus,outcomes,outcomes_plus):
	""" evaluate KKT conditions"""

	par = model.par
	train = model.train
	device = states.device

	# a. unpack
	n = states[...,2:par.Nstates]

	index_lambda = par.D+2 if par.use_softmax else par.D+1
	lambda_mult = actions[...,index_lambda]
	mu_mult = actions[...,index_lambda+1:index_lambda+1+par.D]

	c = outcomes[...,0]
	d = outcomes[...,1:par.D+1]
	m_pd = outcomes[...,-1]
	Delta = d - n

	n_plus = states_plus[...,2:par.Nstates]

	mu_multplus = actions_plus[...,index_lambda+1:index_lambda+1+par.D]

	c_plus = outcomes_plus[...,0]
	d_plus = outcomes_plus[...,1:par.D+1]
	Delta_plus = d_plus-n_plus

	# b. marginal utility
	marg_util_c_t = marg_util_c(c,d,par,train)
	marg_util_d_t = marg_util_d(c,d,par,train)

	marg_util_c_plus = marg_util_c(c_plus,d_plus,par,train)

	# d. FOC d
	if len(states.shape) == 3:
		FOC_d = torch.zeros((states.shape[0],states.shape[1],par.D),dtype=train.dtype,device=device)
		broadcast_shape = (None, None, slice(None))
	else:
		FOC_d = torch.zeros((states.shape[0],par.D),dtype=train.dtype,device=device)
		broadcast_shape = (None, slice(None))

	# c. euler c
	beta = par.beta
	exp_marg_util = torch.sum(train.numint_weights[broadcast_shape]*marg_util_c_plus,dim=-1)
	euler_c = (marg_util_c_t-lambda_mult)/(beta*par.R*exp_marg_util)-1

	for i_d in range(par.D):

		# i. lhs
		lhs_d =  marg_util_d_t[...,i_d]
		
		# ii. rhs
		d_adj_cost_ = d_adj_cost(par,Delta[...,i_d])
		d_adj_cost_plus = d_adj_cost(par,Delta_plus[...,i_d])

		rhs_d_1 = (1+d_adj_cost_)*marg_util_c_t
		rhs_d_2_quad = (1+d_adj_cost_plus)*marg_util_c_plus
		rhs_d_2 = beta*(1-par.delta[i_d])*torch.sum(train.numint_weights[broadcast_shape]*rhs_d_2_quad,dim=-1)

		if len(states.shape) == 3:
			rhs_d_3 = beta*(1-par.delta[i_d])*torch.sum(train.numint_weights[broadcast_shape]*mu_multplus[:par.T-1,...,i_d],dim=-1)
		else:
			rhs_d_3 = beta*(1-par.delta[i_d])*torch.sum(train.numint_weights[broadcast_shape]*mu_multplus[...,i_d],dim=-1)

		rhs_d_4 = mu_mult[...,i_d]

		rhs_d = rhs_d_1 - rhs_d_2 + rhs_d_3 - rhs_d_4

		# iii. FOC
		FOC_d[...,i_d] = lhs_d - rhs_d		

	# e. slackness constraints
	slackness_c = lambda_mult*m_pd
	slackness_d = mu_mult*Delta
	
	# f. combine
	eq = torch.cat((euler_c[...,None]**2,FOC_d**2,slackness_c[...,None],slackness_d),dim=-1)
	
	return eq

def eval_equations_FOC_terminal(model,states,actions,outcomes,states_pd):

	par = model.par
	train = model.train
	device = states.device

	# a. unpack
	n = states[...,2:par.Nstates]

	index_lambda = par.D+2 if par.use_softmax else par.D+1
	mu_mult = actions[...,index_lambda+1:index_lambda+1+par.D]

	c = outcomes[...,0]
	d = outcomes[...,1:par.D+1]
	m_pd = outcomes[...,-1]
	Delta = d - n

	# b. marginal utility
	marg_util_c_t = marg_util_c(c,d,par,train)
	marg_util_d_t = marg_util_d(c,d,par,train)

	# c. euler c
	euler_c = torch.zeros_like(marg_util_c_t) # marg_util_c_t-lambda_mult

	# d. durable consumption FOCs
	if len(states.shape)  == 3:
		FOC_d = torch.zeros((states.shape[0],states.shape[1],par.D),dtype=train.dtype,device=device)
	else:
		FOC_d = torch.zeros((states.shape[0], par.D),dtype=train.dtype,device=device)

	for i_d in range(par.D):

		# i. lhs
		lhs_d =  marg_util_d_t[...,i_d]

		# ii. rhs
		d_adj_cost_ = d_adj_cost(par,Delta[...,i_d])
		rhs_d = (1+d_adj_cost_)*marg_util_c_t-mu_mult[...,i_d]
		
		# iii. combine
		FOC_d[...,i_d] = lhs_d - rhs_d

	# e. slackness constraints
	slackness_c = m_pd # lambda_mult*m_pd
	slackness_d = mu_mult*Delta

	# f. combine	
	eq = torch.cat((euler_c[...,None]**2,FOC_d**2,slackness_c[...,None],slackness_d),dim=-1)
	return eq

#########################
# equations for DeepVPD # 
#########################

def eval_equations_VPD(model,states,actions,outcomes,states_pd,q_pd):
	""" evaluate equation for DeepVPD with FOC """
	
	par = model.par
	train = model.train
	
	# a. unpack
	n = states[...,2:par.Nstates]

	index_lambda = par.D+2 if par.use_softmax else par.D+1
	lambda_mult = actions[...,index_lambda]
	mu_mult = actions[...,index_lambda+1:index_lambda+1+par.D]

	c = outcomes[...,0]
	d = outcomes[...,1:par.D+1]
	m_pd = outcomes[...,-1]
	Delta = d - n
	
	dvpd_dmpd = q_pd[...,0].clamp(0.0,100000.0)
	dvpd_dnpd = q_pd[...,1:1+par.D].clamp(0.0,100000.0)

	# b. marginal utility
	marg_util_c_t = marg_util_c(c,d,par,model.train)
	marg_util_d_t = marg_util_d(c,d,par,train)

	# c. FOC c
	beta = par.beta
	FOC_c = (marg_util_c_t-lambda_mult)/(beta*dvpd_dmpd)-1

	# e. FOC d
	FOC_d = torch.zeros((states.shape[0],states.shape[1],par.D),dtype=model.train.dtype,device=states.device)

	for i_d in range(par.D):

		# i. lhs
		lhs_d = marg_util_d_t[...,i_d]
		
		# ii. rhs
		d_adj_cost_ = d_adj_cost(par,Delta[...,i_d])
		
		rhs_term1 = (1+d_adj_cost_)*marg_util_c_t
		rhs_term2 = beta*dvpd_dnpd[...,i_d]
		rhs_term3 = mu_mult[...,i_d]
		rhs_d = rhs_term1 - rhs_term2 - rhs_term3

		FOC_d[...,i_d] = lhs_d - rhs_d
	
	# d. borrowing constraint
	slackness_c = lambda_mult*m_pd
	slackness_d = mu_mult*Delta

	# g. compute equation errors
	weight = 1/(2+1+2*par.D+par.D)
	eq = weight*(2*FOC_c[...,None]**2 + slackness_c[...,None] + 2*FOC_d**2 + slackness_d)
	return eq

def eval_equations_VPD_terminal(model,states,actions,outcomes,states_pd):

	par = model.par
	train = model.train

	# a. unpac
	n = states[...,2:par.Nstates]
	
	index_lambda = par.D+2 if par.use_softmax else par.D+1
	mu_mult = actions[...,index_lambda+1:index_lambda+1+par.D]

	c = outcomes[...,0]
	d = outcomes[...,1:par.D+1]
	m_pd = outcomes[...,-1]
	Delta = d - n
	
	# b. marginal utility
	marg_util_c_t = marg_util_c(c,d,par,train)
	marg_util_d_t = marg_util_d(c,d,par,train)

	# c. FOC c
	FOC_c = torch.zeros_like(marg_util_c_t)
	
	# d. FOC d
	FOC_d = torch.zeros((states.shape[0],states.shape[1],par.D),dtype=train.dtype,device=states.device)

	for i_d in range(par.D):

		# i. lhs
		lhs_d = marg_util_d_t[...,i_d]

		# ii. rhs
		d_adj_cost_ = d_adj_cost(par,Delta[...,i_d])

		rhs_term1 = (1+d_adj_cost_)*marg_util_c_t
		rhs_term3 = mu_mult[...,i_d]
		rhs_d = rhs_term1 - rhs_term3

		# iii. combine
		FOC_d[...,i_d] = lhs_d - rhs_d

	# e. slack
	slackness_c = m_pd
	slackness_d = mu_mult*Delta

	# f. compute equation errors
	weight = 1/(2+1+2*par.D+par.D)
	eq = weight*(2*FOC_c[...,None]**2 + slackness_c[...,None] + 2*FOC_d**2 + slackness_d)

	return eq

def eval_equations_FOC_t(model,states,states_plus,states_pd,actions,actions_plus,outcomes,outcomes_plus,t):
	if t == model.par.T-1:
		eq = eval_equations_FOC_terminal(model,states,actions,outcomes,states_pd)
	else:
		eq = eval_KKT(model,states,states_plus,actions,actions_plus,outcomes,outcomes_plus)

	return eq

##############
# auxilliary #
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