import numpy as np
import torch

from EconDLSolvers import expand_to_quad, expand_to_states

###########
# reward #
###########

def u(c,d,par):
	""" utility function """

	cobb_d = c**(par.alpha)*(d+par.d_ubar)**(1-par.alpha)
	u = cobb_d**(1-par.rho)/(1-par.rho)

	return u

def marg_u_c(c,d,par):
	""" marginal utility of consumption """
	
	return par.alpha*c**(par.alpha*(1-par.rho)-1.0)*(d+par.d_ubar)**((1-par.alpha)*(1-par.rho))

def outcomes(model,states,actions,t0=0,t=None):
	""" outcomes """

	par = model.par

	# a. unpack
	m = states[...,0]
	n = states[...,2]

	# b. cash-on-hand after adjustment
	mbar = m + (1-par.kappa)*n

	# c. utility	
	c_keep = (1-actions[...,0])*m
	d_keep = n

	expenditure_adj = (1-actions[...,1])*mbar
	c_adj = actions[...,2]*expenditure_adj
	d_adj = (1-actions[...,2])*expenditure_adj
	
	# d. finalize
	return torch.stack((c_keep,d_keep,c_adj,d_adj),dim=-1)

def reward(model,states,actions,outcomes,t0=0,t=None):
	""" reward """

	par = model.par
	train = model.train

	# a. utility
	c_keep = outcomes[...,0]
	d_keep = outcomes[...,1]
	u_keep = u(c_keep,d_keep,par)

	c_adj = outcomes[...,2]
	d_adj = outcomes[...,3]
	u_adj = u(c_adj,d_adj,par)

	# b. finalize
	return torch.stack((u_keep,u_adj),dim=-1)

def discount_factor(model,states,t0=0,t=None):
	""" discount factor """

	par = model.par

	beta = par.beta*torch.ones_like(states[...,0])
	
	return beta 

def exploration(model,states,actions,eps,t=None):

	return actions + eps

def terminal_reward_pd(model,states_pd):
	""" terminal reward """

	# Case I: states_pd.shape = (1,...,Nstates_pd)
	# Case II: states_pd.shape = (N,Nstates_pd)

	train = model.train
	dtype = train.dtype
	device = train.device

	value_pd = torch.zeros((*states_pd.shape[:-1],1),dtype=dtype,device=device)

	return value_pd 
	# Case I: shapes = (1,...,1)
	# Case II: shapes = (N,1)

##############
# transition #
##############

def state_trans_pd(model,states,actions,outcomes,t0=0,t=None):

	par = model.par

	# a. unpack
	m = states[...,0]
	p = states[...,1]
	n = states[...,2]

	# b. cash-on-hand after adjustment
	mbar = m + (1-par.kappa) * n

	# c. post-decision states
	p_pd_adj = p_pd_keep = p

	m_pd_keep = actions[...,0]*m
	n_pd_keep = n

	m_pd_adj = actions[...,1]*mbar
	expenditure_adj = (1-actions[...,1])*mbar
	n_pd_adj = (1-actions[...,2])*expenditure_adj

	# d. finalize
	pd_keep = torch.stack((m_pd_keep,p_pd_keep,n_pd_keep),dim=-1)
	pd_adj = torch.stack((m_pd_adj,p_pd_adj,n_pd_adj),dim=-1)
	pd = torch.stack((pd_keep,pd_adj),dim=-1)
	return pd

def state_trans(model,states_pd,quad,t0=0,t=None):
	""" state transition with quadrature """

	par = model.par
	train = model.train

	# a. unpack
	a = states_pd[...,0]
	p = states_pd[...,1]
	d = states_pd[...,2]

	xi = quad[:,0]
	psi = quad[:,1]

	# b. expand
	if t is None:

		a = expand_to_quad(a,train.Nquad)
		p = expand_to_quad(p,train.Nquad)
		d = expand_to_quad(d,train.Nquad)

		xi = expand_to_states(xi,states_pd)
		psi = expand_to_states(psi,states_pd)
		
	# c. permanent income state
	p_plus = p**par.eta*xi

	# d. compute income for cash-in-hand
	income = p_plus * psi

	# e. cash-on-hand state
	m_plus = par.R*a+income

	# f. durable state
	n_plus = (1-par.delta)*d*torch.ones_like(m_plus)

	# g. pack
	return torch.stack((m_plus,p_plus,n_plus),dim=-1)

#######
# FOC #
#######

def marginal_reward(model,states,actions,outcomes,t0=0,t=None):
	""" reward """

	par = model.par
	train = model.train

	# a. utility
	c_keep = outcomes[...,0]
	d_keep = outcomes[...,1]
	marg_u_c_keep = marg_u_c(c_keep,d_keep,par)

	c_adj = outcomes[...,2]
	d_adj = outcomes[...,3]
	marg_u_c_adj = marg_u_c(c_adj,d_adj,par)

	# b. finalize
	return torch.stack((marg_u_c_keep,marg_u_c_adj),dim=-1)

def eval_equations_DeepVPDDC(model,states,actions,marginal_reward,q):
	""" evaluate equation for DeepVPDDC with FOC """

	par = model.par

	# a. borrowing constraint
	constraint = actions[:-1,:,:2]

	# b. compute euler equation
	FOC = marginal_reward[:-1] / (par.R*par.beta*q) - 1

	# c. combine with borrowing constraint
	eq_error = fischer_burmeister(FOC,constraint)

	return eq_error**2

def fischer_burmeister(a,b):
	""" Fischer-Burmeister function - torch """
	 
	return torch.sqrt(a**2 + b**2) - a - b