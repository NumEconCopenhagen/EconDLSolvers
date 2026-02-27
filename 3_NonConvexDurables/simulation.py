
import numpy as np
from consav.linear_interp import interp_2d, interp_3d, interp_4d
import numba as nb

def simulate_1d(par,vfi,sim):
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
	d1 = sim.d1
	
	xi = shocks[:,:,0]
	psi = shocks[:,:,1]

	# b. time loop  
	for t in range(par.T):

		# i. discrete choice
		value_adj = np.zeros((sim.N))
		value_keep = np.zeros((sim.N))
		# mbar = np.zeros((sim.N))
		# x = np.zeros((sim.N))
		sav_rate = np.zeros((sim.N))

		for i in range(sim.N):
			x = m[t,i] + (1-par.nu)*n[t,i]
			# value_adj[i] = interp_3d(vfi.p_grid,vfi.n_grid,vfi.m_grid,vfi.sol_v_adj[t],p[t,i],n[t,i],m[t,i])
			value_adj[i] = interp_2d(vfi.p_grid,vfi.x_grid,vfi.sol_v_adj[t],p[t,i],x)
			value_keep[i] = interp_3d(vfi.p_grid,vfi.n_grid,vfi.m_grid,vfi.sol_v_keep[t],p[t,i],n[t,i],m[t,i])

		value_keep += taste_shocks[t,:,0]
		value_adj += taste_shocks[t,:,1]

		DC[t] = value_adj > value_keep
		
		# ii. continous choice
		x = np.zeros((sim.N))
		for i in range(sim.N):
			
			# continous choices
			if DC[t,i] == 0: # keep
				x[i] = m[t,i]

				sav_share = interp_3d(vfi.p_grid,vfi.n_grid,vfi.m_grid,vfi.sol_sav_share_keep[t],p[t,i],n[t,i],m[t,i])
				sav_share = np.clip(sav_share,0.0,0.9999)
				# sav_share = 0.5
				sav_rate[i] = sav_share
				# print(sav_share)
				c[t,i] = (1-sav_share)*x[i]
				d1[t,i] = n[t,i]

			else: # adjust
				x[i] = m[t,i] + (1-par.nu)*n[t,i]

				exp_share = interp_2d(vfi.p_grid,vfi.x_grid,vfi.sol_exp_share_adj[t],p[t,i],x[i])
				exp_share = np.clip(exp_share,1e-8,1.0)
				# exp_share = 0.5
				sav_rate[i] = 1 - exp_share
				c_share = interp_2d(vfi.p_grid,vfi.x_grid,vfi.sol_c_share_adj[t],p[t,i],x[i])
				c_share = np.clip(c_share,1e-8,1.0)
				# c_share = 0.5

				exp_adjustment = exp_share*(x[i])
				c[t,i] = c_share*exp_adjustment
				d1[t,i] = (1-c_share)*exp_adjustment

		# iii. reward
		c_util = c[t]**(1 - par.omega[0])
		d_util = (d1[t]+par.d_ubar[0])**(par.omega[0])
		cobb_d = c_util*d_util
		reward[t,:,0] = (cobb_d)**(1-par.rho)/(1-par.rho) + taste_shocks[t,:,0] * (DC[t]==0) + taste_shocks[t,:,1] * (DC[t]==1)
		reward[t,:,1] = 0.0

		# iv. post-decision states
		a[t] = sav_rate * x

		p_pd[t] = p[t]
		n_pd[t] = d1[t]

		# v. next period
		if t < par.T-1:
			
			# i. permanent income state
			p[t+1] = (p_pd[t]**par.eta)*xi[t+1]
			
			# ii. income
			y = par.kappa[t]*p[t+1]*psi[t+1]
			
			# iii. cash-on-hand
			m[t+1] = par.R*a[t] + y

			n[t+1] = n_pd[t] * (1-par.delta[0])



def simulate_2d(par,vfi,sim):
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
	n1 = states[:,:,2]
	n2 = states[:,:,3]

	a = states_pd[:,:,0]
	p_pd = states_pd[:,:,1]
	n1_pd = states_pd[:,:,2]
	n2_pd = states_pd[:,:,3]

	c = sim.c
	d1 = sim.d1
	d2 = sim.d2
	
	xi = shocks[:,:,0]
	psi = shocks[:,:,1]

	# b. time loop  
	for t in range(par.T):

		# i. discrete choice
		value_adj1 = np.zeros((sim.N))
		value_adj2 = np.zeros((sim.N))
		value_adj12 = np.zeros((sim.N))
		value_keep = np.zeros((sim.N))

		for i in range(sim.N):
			value_keep[i] = interp_4d(vfi.p_grid,vfi.n_grid,vfi.n_grid,vfi.m_grid,vfi.sol_v_keep[t],p[t,i],n1[t,i],n2[t,i],m[t,i])
			x_adj1 = m[t,i] + (1-par.nu)*n1[t,i]
			value_adj1[i] = interp_3d(vfi.p_grid,vfi.n_grid,vfi.x_grid,vfi.sol_v_adj1[t],p[t,i],n2[t,i],x_adj1)
			x_adj2 = m[t,i] + (1-par.nu)*n2[t,i]
			value_adj2[i] = interp_3d(vfi.p_grid,vfi.n_grid,vfi.x_grid,vfi.sol_v_adj2[t],p[t,i],n1[t,i],x_adj2)
			x_adj12 = m[t,i] + (1-par.nu)*(n1[t,i]+n2[t,i])
			value_adj12[i] = interp_2d(vfi.p_grid,vfi.x_grid,vfi.sol_v_adj12[t],p[t,i],x_adj12)


		value_keep += taste_shocks[t,:,0]
		value_adj1 += taste_shocks[t,:,2]
		value_adj2 += taste_shocks[t,:,1]
		value_adj12 += taste_shocks[t,:,3]

		# 2 and 1 are swapped to match ordering in DL
		DC[t] = np.argmax(np.stack((value_keep, value_adj2, value_adj1, value_adj12)), axis=0)

		# ii. continous choice
		x = np.zeros((sim.N))
		exp_share = np.zeros((sim.N))
		for i in range(sim.N):
			
			# continous choices
			if DC[t,i] == 0: # keep
				x[i] = m[t,i]

				sav_share = interp_4d(vfi.p_grid,vfi.n_grid,vfi.n_grid,vfi.m_grid,vfi.sol_sav_share_keep[t],p[t,i],n1[t,i],n2[t,i],m[t,i])
				sav_share = np.clip(sav_share,0.0,0.9999)
				# sav_share = 0.5
				exp_share[i] = 1 - sav_share
				c[t,i] = (1-sav_share)*x[i]
				d1[t,i] = n1[t,i]
				d2[t,i] = n2[t,i]
			elif DC[t,i] == 2: # adjust durable 1
				x[i] = m[t,i] + (1-par.nu)*n1[t,i]

				exp_share[i] = interp_3d(vfi.p_grid,vfi.n_grid,vfi.x_grid,vfi.sol_exp_share_adj1[t],p[t,i],n2[t,i],x[i])
				exp_share[i] = np.clip(exp_share[i],1e-8,1.0)
				c_share = interp_3d(vfi.p_grid,vfi.n_grid,vfi.x_grid,vfi.sol_c_share_adj1[t],p[t,i],n2[t,i],x[i])
				c_share = np.clip(c_share,1e-8,1.0)
				# c_share = 0.5
				# exp_share[i] = 0.5
				exp_adjustment = exp_share[i]*(x[i])
				c[t,i] = c_share*exp_adjustment
				d1[t,i] = (1-c_share)*exp_adjustment
				d2[t,i] = n2[t,i]

			elif DC[t,i] == 1: # adjust durable 2
				x[i] = m[t,i] + (1-par.nu)*n2[t,i]

				exp_share[i] = interp_3d(vfi.p_grid,vfi.n_grid,vfi.x_grid,vfi.sol_exp_share_adj2[t],p[t,i],n1[t,i],x[i])
				exp_share[i] = np.clip(exp_share[i],1e-8,1.0)
				# exp_share[i] = 0.5
				c_share = interp_3d(vfi.p_grid,vfi.n_grid,vfi.x_grid,vfi.sol_c_share_adj2[t],p[t,i],n1[t,i],x[i])
				c_share = np.clip(c_share,1e-8,1.0)
				# c_share = 0.5
				exp_adjustment = exp_share[i]*(x[i])
				c[t,i] = c_share*exp_adjustment
				d1[t,i] = n1[t,i]
				d2[t,i] = (1-c_share)*exp_adjustment
			
			else: # adjust both durable
				x[i] = m[t,i] + (1-par.nu)*(n1[t,i]+n2[t,i])

				exp_share[i] = interp_2d(vfi.p_grid,vfi.x_grid,vfi.sol_exp_share_adj12[t],p[t,i],x[i])
				exp_share[i] = np.clip(exp_share[i],1e-8,1.0)
				# exp_share[i] = 0.5
				c_share = interp_2d(vfi.p_grid,vfi.x_grid,vfi.sol_c_share_adj12[t],p[t,i],x[i])
				c_share = np.clip(c_share,1e-8,1.0)
				# c_share = 0.5
				d1_share = interp_2d(vfi.p_grid, vfi.x_grid,vfi.sol_d1_share_adj12[t],p[t,i],x[i])
				d1_share = np.clip(d1_share, 0.0, 1.0)
				# d1_share = 0.5
				exp_adjustment = exp_share[i]*(x[i])
				c[t,i] = c_share*exp_adjustment
				durable_exp = (1-c_share)*exp_adjustment
				d1[t,i] = d1_share*durable_exp
				d2[t,i] = (1-d1_share)*durable_exp
				
		# iii. reward
		c_util = c[t]**(1 - par.omega[0] - par.omega[1])
		d1_util = (d1[t]+par.d_ubar[0])**(par.omega[0])
		d2_util = (d2[t]+par.d_ubar[1])**(par.omega[1])
		cobb_d = c_util*d1_util*d2_util
		selected_shocks = taste_shocks[t, np.arange(sim.N), DC[t].astype(int)]

		reward[t,:,0] = (cobb_d)**(1-par.rho)/(1-par.rho) + selected_shocks

		# iv. post-decision states
		a[t] = x * (1 - exp_share)

		p_pd[t] = p[t]
		n1_pd[t] = d1[t]
		n2_pd[t] = d2[t]

		# v. next period
		if t < par.T-1:
			
			# i. permanent income state
			p[t+1] = (p_pd[t]**par.eta)*xi[t+1]
			
			# ii. income
			y = par.kappa[t]*p[t+1]*psi[t+1]
			
			# iii. cash-on-hand
			m[t+1] = par.R*a[t] + y

			n1[t+1] = n1_pd[t] * (1-par.delta[0])
			n2[t+1] = n2_pd[t] * (1-par.delta[1])
