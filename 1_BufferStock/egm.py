import numpy as np
import numba as nb

from EconModel import jit
from consav.linear_interp import interp_2d,interp_3d, interp_4d, interp_5d, interp_1d_vec_mon_noprep

#########
# solve #
#########

@nb.njit
def inverse_marg_util(par, u):
	""" Inverse function of marginal utility of consumption """

	return 1/u

@nb.njit
def marg_util_con(par, c):
	""" marginal utility of consumption """

	return 1/c

@nb.njit
def utility(par, c):
	""" utility function """

	return np.log(c)

@nb.njit
def utility_terminal(par, c, b):
	""" utility function in terminal period """
	if par.bequest == 0.0:
		u = np.log(c)
	else:
		u = np.log(c) + par.bequest*np.log(b)
	return u



@nb.njit(parallel=False)
def EGM(t,par,egm):
	""" EGM for policy functions at time t"""

	sol_con = egm.sol_con
	m_pd_grid = egm.m_pd_grid
	m_grid = egm.m_grid
	p_grid = egm.p_grid
	sigma_xi_grid = egm.sigma_xi_grid
	sigma_psi_grid = egm.sigma_psi_grid
	rho_p_grid = egm.rho_p_grid

	# a. last period
	if t == par.T-1:

		for i_p in nb.prange(egm.Np):
			for i_sigma_xi in nb.prange(egm.Nsigma_xi):
				for i_sigma_psi in nb.prange(egm.Nsigma_psi):
					for i_rho_p in nb.prange(egm.Nrho_p):
						sol_con[t,i_p,i_sigma_xi,i_sigma_psi,i_rho_p,:] = (m_grid)/(1+par.bequest)

	# b. other periods
	else:

		shape = sol_con[0].shape
		q_grid = np.zeros(shape)

		# i. next-period marginal value of cash
		for i_p in nb.prange(egm.Np):
			p = p_grid[i_p]
			for i_m_pd in nb.prange(egm.Nm_pd):
				m_pd = m_pd_grid[i_m_pd]
				for i_sigma_xi in nb.prange(egm.Nsigma_xi):
					sigma_xi = sigma_xi_grid[i_sigma_xi]
					for i_sigma_psi in nb.prange(egm.Nsigma_psi):
						sigma_psi = sigma_psi_grid[i_sigma_psi]
						for i_rho_p in nb.prange(egm.Nrho_p):
							rho_p = rho_p_grid[i_rho_p]

							if par.Nstates_fixed == 0:
								sigma_xi = par.sigma_xi_base
								sigma_psi = par.sigma_psi_base
								rho_p = par.rho_p_base
							elif par.Nstates_fixed == 1:
								sigma_psi = par.sigma_psi_base
								rho_p = par.rho_p_base
							elif par.Nstates_fixed == 2:
								rho_p = par.rho_p_base
							
							q = compute_q(par,egm,t,sigma_xi,sigma_psi,m_pd,p,rho_p)
							q_grid[i_p,i_sigma_xi,i_sigma_psi,i_rho_p,i_m_pd] = q

		# ii. endogenous grid and interpolation to common grid
		for i_p in nb.prange(egm.Np):
			for i_sigma_xi in nb.prange(egm.Nsigma_xi):
				for i_sigma_psi in nb.prange(egm.Nsigma_psi):
					for i_rho_p in nb.prange(egm.Nrho_p):
						interp_to_common_grid(par,egm,t,q_grid,(i_p,i_sigma_xi,i_sigma_psi,i_rho_p))

@nb.njit
def compute_q(par,egm,t,sigma_xi,sigma_psi,m_pd,p,rho_p):
	""" compute post-decision marginal value of cash """

	# unpack
	sol_con = egm.sol_con
	m_grid = egm.m_grid
	p_grid = egm.p_grid
	sigma_xi_grid = egm.sigma_xi_grid
	sigma_psi_grid = egm.sigma_psi_grid
	rho_p_grid = egm.rho_p_grid

	# a. initialize q
	q = 0.0

	# b. loop over psi and xi
	for i_psi in range(par.Npsi):
		for i_xi in range(par.Nxi):
			
			# o. adjust nodes
			xi = par.xi[i_xi]
			xi = xi*sigma_xi
			xi = np.exp(xi-0.5*sigma_xi**2)
			psi = par.psi[i_psi]
			psi = psi*sigma_psi
			psi = np.exp(psi-0.5*sigma_psi**2)

			# oo. next-period states
			p_plus = p**(rho_p) * xi
			if t < par.T_retired:
				m_plus = par.R*m_pd + psi*p_plus * par.kappa[t]
			else:
				m_plus = par.R*m_pd  + par.kappa[t]

			# ooo. next-period consumption and marginal utility
			if par.Nstates_fixed == 0:
				c_plus = interp_2d(p_grid,m_grid,sol_con[t+1,:,0,0,0],p_plus,m_plus) # slice to get rid of sigma_xi and sigma_psi
			elif par.Nstates_fixed == 1:
				c_plus = interp_3d(p_grid,sigma_xi_grid,m_grid,sol_con[t+1,:,:,0,0],p_plus,sigma_xi, m_plus) # slice to get rid of sigma_psi
			elif par.Nstates_fixed == 2:
				c_plus = interp_4d(p_grid,sigma_xi_grid,sigma_psi_grid,m_grid,sol_con[t+1,:,:,:,0],p_plus,sigma_xi,sigma_psi,m_plus)
			elif par.Nstates_fixed == 3:
				c_plus = interp_5d(p_grid,sigma_xi_grid,sigma_psi_grid,rho_p_grid,m_grid,sol_con[t+1],p_plus,sigma_xi, sigma_psi, rho_p, m_plus)
			else:
				raise ValueError('Nstates_fixed must be 0, 1, 2 or 3')
			
			mu_plus = marg_util_con(par, c_plus)
			
			# oooo. add to q
			q += par.psi_w[i_psi]*par.xi_w[i_xi]*mu_plus

	return q
				
@nb.njit
def interp_to_common_grid(par,egm,t,q_grid,q_index_tup):
	""" endogenous grid method """
	
	# o. temp
	m_grid = egm.m_grid
	m_pd_grid = egm.m_pd_grid
	sol_con = egm.sol_con

	m_temp = np.zeros(egm.Nm_pd+1)
	c_temp = np.zeros(egm.Nm_pd+1)
	
	# o. endogenous grid
	for i_m_pd in range(egm.Nm_pd):
		m_pd = m_pd_grid[i_m_pd]
		q_index = q_index_tup + (i_m_pd,)
		c_temp[i_m_pd+1] = inverse_marg_util(par, par.beta*par.R*q_grid[q_index])
		m_temp[i_m_pd+1] = m_pd + c_temp[i_m_pd+1]

	# oo. interpolation to common grid
	# add index together
	sol_con_index = (t,) + q_index_tup
	interp_1d_vec_mon_noprep(m_temp,c_temp,m_grid,sol_con[sol_con_index])

###############
# euler error #
###############

@nb.njit
def compute_euler_errors(par,egm,sim):
	""" compute euler error for EGM"""

	euler_error = sim.euler_error
	sigma_xi = par.sigma_xi_base
	sigma_psi = par.sigma_psi_base
	rho_p = par.rho_p_base

	c = sim.outcomes[:,:,0]

	for t in range(par.T-1):
		for i in range(sim.N):

			c_ti = c[t,i]
			m_ti = sim.states[t,i,0]
			a_ti = m_ti - c_ti
			p_ti = sim.states[t,i,1]

			if par.Nstates_fixed > 0:
				sigma_xi = sim.states[t,i,2]
			if par.Nstates_fixed > 1:
				sigma_psi = sim.states[t,i,3]
			if par.Nstates_fixed > 2:
				rho_p = sim.states[t,i,4]

			q = compute_q(par,egm,t,sigma_xi,sigma_psi,a_ti,p_ti,rho_p)
			euler_error[t,i] = inverse_marg_util(par,par.beta*par.R*q)/c_ti-1

##############
# simulation #
##############

@nb.njit(parallel=True)
def policy_interp(par,egm,N,t,states,con):
	""" interpolate policy function to get consumption given current states """

	for i in nb.prange(N):
		if par.Nstates_fixed == 0:
			con[i]  = interp_2d(egm.p_grid, egm.m_grid, egm.sol_con[t,:,0,0,0,:],states[i,1],states[i,0])
		elif par.Nstates_fixed == 1:
			con[i] = interp_3d(egm.p_grid, egm.sigma_xi_grid, egm.m_grid, egm.sol_con[t,:,:,0,0,:],states[i,1],states[i,2],states[i,0])
		elif par.Nstates_fixed == 2:
			con[i] = interp_4d(egm.p_grid, egm.sigma_xi_grid, egm.sigma_psi_grid, egm.m_grid, egm.sol_con[t,:,:,:,0,:],states[i,1],states[i,2],states[i,3],states[i,0])
		elif par.Nstates_fixed == 3:
			con[i] = interp_5d(egm.p_grid, egm.sigma_xi_grid, egm.sigma_psi_grid, egm.rho_p_grid, egm.m_grid, egm.sol_con[t,:,:,:,:,:],states[i,1],states[i,2],states[i,3],states[i,4],states[i,0])	

def simulate(par,egm,sim,final=False):
	""" simulate model """

	# final=True: extra output is produced

	# a. unpack
	states = sim.states # shape (T,N,Nstates)
	states_pd = sim.states_pd # shape (T,N,Nstates_pd)
	outcomes = sim.outcomes # shape (T,N,Noutcomes)
	shocks = sim.shocks # shape (T,N,Nshocks)
	reward = sim.reward # shape (T,N,Nstates)
	MPC = sim.MPC # shape (T,N,Nstates)

	m = states[:,:,0]
	p = states[:,:,1]

	m_pd = states_pd[:,:,0]
	p_pd = states_pd[:,:,1]
	
	xi = shocks[:,:,0]
	psi = shocks[:,:,1]

	c = outcomes[:,:,0]

	if par.Nstates_fixed > 0:
		fixed_states = states[:,:,2:]
		fixed_states_pd = states_pd[:,:,2:]

	if par.Nstates_fixed >= 1:
		sigma_xi = states[0,:,2]
	else:
		sigma_xi = par.sigma_xi_base

	if par.Nstates_fixed >= 2:
		sigma_psi = states[0,:,3]
	else:
		sigma_psi = par.sigma_psi_base

	if par.Nstates_fixed >= 3:
		rho_p = states[0,:,4]
	else:
		rho_p = par.rho_p_base
	
	# b. scale shocks
	xi = np.exp(sigma_xi*xi-0.5*sigma_xi**2)
	psi = np.exp(sigma_psi*psi-0.5*sigma_psi**2)

	# c. time loop  
	for t in range(par.T):

		if par.Nstates_fixed > 0: fixed_states_pd[t] = fixed_states[t]

		# a. final period consumption
		if t == par.T-1:

			c[t] = (m[t])/(1+par.bequest)
			if final: MPC[t] = 1.0/(1+par.bequest)
		# b. consumption all other periods
		else:
				
				policy_interp(par,egm,sim.N,t,states[t],c[t])
				
				if final:

					states_MPC = np.zeros(states[t].shape)	
					states_MPC[:] = states[t]
					states_MPC[:,0] += par.Delta_MPC	

					c_MPC = np.zeros(c[t].shape)	
					c_MPC[:] = c[t]

					policy_interp(par,egm,sim.N,t,states_MPC,c_MPC)
				
					MPC[t] = (c_MPC-c[t])/par.Delta_MPC

		# c. reward
		if t == par.T-1:
			reward[t] = utility_terminal(par,c[t],m[t]-c[t])
		else:
			reward[t] = utility(par,c[t])

		# d. post-decision states
		m_pd[t] = m[t]-c[t]
		p_pd[t] = p[t]

		# e. next period
		if t < par.T-1:

			# i. permanent income state
			p[t+1] = (p_pd[t]**rho_p)*xi[t+1]
			
			# ii. income
			if t < par.T_retired:
				y = par.kappa[t]*p[t+1]*psi[t+1]
			else:
				y = par.kappa[t]
			
			# iii. cash-on-hand
			m[t+1] = par.R*m_pd[t] + y

			# iv. fixed states
			if par.Nstates_fixed > 0:
				fixed_states[t+1] = fixed_states_pd[t]