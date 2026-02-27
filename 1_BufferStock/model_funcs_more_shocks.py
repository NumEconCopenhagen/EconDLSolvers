import numpy as np
import torch

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
	
	xi = shocks[...,0] # persistent income shock
	psi = shocks[...,1] # transitory income shock
	
	# uniform shocks
	u_shock = shocks[...,2] # unemployment shock
	mix_xi_shock = shocks[...,3] # xi shock mix
	mix_psi_shock = shocks[...,4] # psi shock mix
	
	#transform into binaries
	if (train.use_quad == False) or (t is not None and (len(shocks.shape)==2)): # using monte carlo or simulating
		u = u_shock < par.p_u # unemployment
		mix_xi = mix_xi_shock < par.p_xi # xi shock mix
		mix_psi = mix_psi_shock < par.p_psi # psi shock mix
	else:
		xi = xi * 2**(1/2) # adjust for quadrature nodes
		psi = psi * 2**(1/2)
		u = u_shock.bool()
		mix_xi = mix_xi_shock.bool()
		mix_psi = mix_psi_shock.bool()

	sigma_xi = torch.where(mix_xi, par.sigma_xi_1, par.sigma_xi_2) # shape = (T,N,Nnumint)
	sigma_psi = torch.where(mix_psi, par.sigma_psi_1, par.sigma_psi_2) # shape = (T,N,Nnumint)

	# b. adjust shape and scale quadrature nodes
	xi = torch.exp(sigma_xi*xi-0.5*sigma_xi**2)
	psi = torch.exp(sigma_psi*psi-0.5*sigma_psi**2)

	# c. next period

	# i. persistent income
	p_plus = p_pd**par.rho_p_base*xi

	# ii. actual income
	if t is None: # when solving simultaneously

		kappa = par.kappa[:,None,None] # (T,) -> (T,1,1) to match (T,N,Nnumint)

		if par.T_retired == par.T:
			y_plus_ = kappa[:par.T-1]*p_plus*psi
			y_plus = torch.where(u, y_plus_ * par.u_replacement, y_plus_)
		else:
			y_plus_before_ = kappa[:par.T_retired]*p_plus[:par.T_retired]*psi[:par.T_retired]
			y_plus_before = torch.where(u[:par.T_retired], y_plus_before_ * par.u_replacement, y_plus_before_)
			y_plus_after = kappa[par.T_retired] * torch.ones_like(p_plus[par.T_retired:])
			y_plus = torch.cat((y_plus_before,y_plus_after),dim=0)
	
	else: # when simulating or solving backwards

		if t < par.T_retired:
			y_plus_base = par.kappa[t]*p_plus*psi
			y_plus =  torch.where(u, y_plus_base * par.u_replacement, y_plus_base) # unemployed get zero income
		else:
			y_plus = par.kappa[t] * torch.ones_like(p_plus)

	# iii. cash-on-hand
	m_plus = par.R*m_pd + y_plus # shape = (T,N,Nnumint) or (T,N)
	
	# d. finalize
	states_plus = torch.stack((m_plus,p_plus), dim=-1) # shape = (T,N,Nnumint,Nstates_pd) or (N,Nnumint,Nstates_pd	)
	return states_plus
	# Case I: shape = (T,...,Nstates)
	# Case II: shape = (...,Nstates)