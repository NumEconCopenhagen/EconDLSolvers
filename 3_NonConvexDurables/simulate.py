import numpy as np
from consav.linear_interp import interp_3d

def simulate(par,vfi,sim):
	""" simulate model """

	# a. unpack
	states = sim.states # shape (T,N,Nstates)
	states_pd = sim.states_pd # shape (T,N,Nstates_pd)
	outcomes = sim.outcomes # shape (T,N,Noutcomes)
	shocks = sim.shocks # shape (T,N,Nshocks)
	reward = sim.reward # shape (T,N,Nstates)
	taste_shocks = sim.taste_shocks # shape (T,N,NDC)
	actions = sim.actions # shape (T,N,Nactions)
	DC = sim.DC # shape (T,N)

	m = states[:,:,0]
	p = states[:,:,1]
	n = states[:,:,2]

	a = states_pd[:,:,0]
	p_pd = states_pd[:,:,1]
	n_pd = states_pd[:,:,2]

	c = sim.c
	d = sim.d
	
	xi = shocks[:,:,0]
	psi = shocks[:,:,1]

	# b. time loop  
	for t in range(par.T):

		# i. discrete choice
		value_adj = np.zeros((sim.N))
		value_keep = np.zeros((sim.N))

		for i in range(sim.N):
			value_adj[i] = interp_3d(vfi.p_grid,vfi.n_grid,vfi.m_grid,vfi.sol_v_adj[t],p[t,i],n[t,i],m[t,i])
			value_keep[i] = interp_3d(vfi.p_grid,vfi.n_grid,vfi.m_grid,vfi.sol_v_keep[t],p[t,i],n[t,i],m[t,i])

		value_keep += taste_shocks[t,:,0]
		value_adj += taste_shocks[t,:,1]

		DC[t] = value_adj > value_keep
		
		# ii. continous choice
		for i in range(sim.N):
			
			# continous choices
			if DC[t,i] == 0: # keep

				sav_share = interp_3d(vfi.p_grid,vfi.n_grid,vfi.m_grid,vfi.sol_sav_share_keep[t],p[t,i],n[t,i],m[t,i])
				sav_share = np.clip(sav_share,0.0,0.9999)
				c[t,i] = (1-sav_share)*m[t,i]
				d[t,i] = n[t,i]

			else: # adjust

				exp_share = interp_3d(vfi.p_grid,vfi.n_grid,vfi.m_grid,vfi.sol_exp_share_adj[t],p[t,i],n[t,i],m[t,i])
				exp_share = np.clip(exp_share,0.0,0.9999)
				c_share = interp_3d(vfi.p_grid,vfi.n_grid,vfi.m_grid,vfi.sol_c_share_adj[t],p[t,i],n[t,i],m[t,i])
				c_share = np.clip(c_share,1e-8,1.0)

				exp_adjustment = (1-exp_share)*(m[t,i]+(1-par.kappa)*n[t,i])
				c[t,i] = c_share*exp_adjustment
				d[t,i] = (1-c_share)*exp_adjustment

		# iii. reward
		c_util = c[t]**(par.alpha)
		d_util = (d[t]+par.d_ubar)**(1-par.alpha)
		cobb_d = c_util*d_util
		reward[t] = (cobb_d)**(1-par.rho)/(1-par.rho) + taste_shocks[t,:,0] * (DC[t]==0) + taste_shocks[t,:,1] * (DC[t]==1)

		# iv. post-decision states
		a[t] = m[t] - c[t] + (DC[t]==1)*((1-par.kappa)*n[t] - d[t])

		p_pd[t] = p[t]
		n_pd[t] = n[t] * (DC[t]==0) + d[t] * (DC[t]==1)

		# v. next period
		if t < par.T-1:
			
			# i. permanent income state
			p[t+1] = (p_pd[t]**par.eta)*xi[t+1]
			
			# ii. income
			y = p[t+1]*psi[t+1]
			
			# iii. cash-on-hand
			m[t+1] = par.R*a[t] + y

			n[t+1] = n_pd[t] * (1-par.delta)