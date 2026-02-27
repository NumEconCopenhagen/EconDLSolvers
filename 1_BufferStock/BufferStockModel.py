import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE' # without this python may crash when plotting from matplotlib
os.environ['MKL_THREADING_LAYER'] = 'GNU'
from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F
torch.set_warn_always(True)
import math
from torch.quasirandom import SobolEngine

# consav
from consav.quadrature import log_normal_gauss_hermite, gauss_hermite
from consav.grids import nonlinspace

from EconDLSolvers import DLSolverClass, torch_uniform

# local
import model_funcs
import model_funcs_more_shocks

# class
class BufferStockModelClass(DLSolverClass):
	
	#########
	# setup #
	#########

	def setup(self,full=None):
		""" choose parameters """

		# a. unpack
		par = self.par
		sim = self.sim
		
		par.full = full if not full is None else torch.cuda.is_available()
		par.seed = 1 # seed for random number generator in torch

		# b. model
		par.T = 55 # number of periods
		par.T_retired = 45 # number of periods retired

		# c. preferences
		par.beta = 1/1.01 # discount factor
		par.bequest = 0.0 # bequest motive

		# d. income
		par.kappa_base = 1.0 # base 
		par.kappa_growth = 0.02 # income growth
		par.kappa_growth_decay = 0.1 # income growth decay
		par.kappa_retired = 0.7 # replacement rate

		par.rho_p_base = 0.95 # shock, persistence
		par.rho_p_low = 0.70   
		par.rho_p_high = 0.99

		par.sigma_xi_base = 0.1 # shock, permanent , std
		par.sigma_xi_low = 0.05
		par.sigma_xi_high = 0.15
		par.Nxi = 4 # number of qudrature nodes

		par.sigma_psi_base = 0.1 # shock, transitory std
		par.sigma_psi_low = 0.05
		par.sigma_psi_high = 0.15
		par.Npsi = 4 # number of qudrature nodes

		# version with more shocks
		par.MoreShocks = False # whether to use more complex shock structure
		par.sigma_xi_1 = np.nan # shock, permanent , std, case 1
		par.sigma_xi_2 = np.nan # shock, permanent , std, case 2
		par.p_xi = np.nan # mixture probability for xi1 and xi2
		par.sigma_psi_1 = np.nan # shock, transitory std, case 1
		par.sigma_psi_2 = np.nan # shock, transitory std, case 2
		par.p_psi = np.nan # mixture probability for psi1 and psi2
		par.p_u = np.nan # probability of unemployment
		par.u_replacement = np.nan # replacement rate in case of unemployment

		# monte carlo
		par.Nmc = 100 # number of monte carlo draws for psi, xi
		par.do_antithetic_mc = True # use antithetic sampling

		# f. return
		par.R = 1.01 # gross return

		# g. states and shocks
		par.Nstates_fixed = 0 # number of fixed states
		par.Nstates_dynamic = 2 # number of dynamic states
		par.Nstates_dynamic_pd = 2 # number of dynamic post-decision states
		par.Nshocks = 2 # number of shocks

		# h. outcomes and actions
		par.Noutcomes = 1 # number of outcomes (here just c)
		par.KKT = False # use KKT conditions (for DeepFOC)

		# i. Euler
		par.Euler_error_min_savings = 1e-3 # minimum savings rate for computing Euler error
		par.Delta_MPC = 1e-4 # windfall used in MPC calculation

		# j. simulation 
		sim.N = 100_000 # number of agents
		sim.reps = 5 # number of repetitions for R simulation
		
		# k. initial states
		par.mu_m0 = 1.0 # initial cash-on-hand, mean
		par.sigma_m0 = 0.1 # initial cash-on-hand, std

		# l. initial permanent income
		par.mu_p0 = 1.0 # initial permanent income, mean
		par.sigma_p0 = 0.1 # initial income, std

		# m. exploration of initial states (currently not used)
		par.explore_states_endo_fac = 1.0
		par.explore_states_fixed_fac_low = 1.0
		par.explore_states_fixed_fac_high = 1.0

		# n. manual initialization
		par.init_bias_logodds = False
		par.w_p = np.nan
		par.init_sigmoid_xavier_uniform = False
		par.init_sigmoid_xavier_normal = False
		par.init_relu_he_uniform = False
		par.init_relu_he_normal = False

	def allocate(self):
		""" allocate arrays  """

		# a. unpack
		par = self.par
		sim = self.sim
		train = self.train

		dtype = train.dtype
		device = train.device

		# b. for solving without GPU
		if not par.full: 

			par.T = 5
			par.T_retired = 3
			sim.N = 1_000
		
		# c. life cycle income
		par.kappa = torch.zeros(par.T-1,dtype=dtype,device=device)
		par.kappa[0] = par.kappa_base
	
		for t in range(1,par.T_retired):
			par.kappa[t] = par.kappa[t-1]*(1+par.kappa_growth*(1-par.kappa_growth_decay)**(t-1))

		par.kappa[par.T_retired:] = par.kappa_retired*par.kappa_base

		# d. states, shocks and actions
		par.Nstates_fixed_pd = par.Nstates_fixed # number of fixed post-decision states
		par.Nstates = par.Nstates_dynamic + par.Nstates_fixed # number of states
		par.Nstates_pd = par.Nstates_dynamic_pd + par.Nstates_fixed_pd # number of post-decision states

		if par.KKT: 
			par.Nactions = 2 # savings rate and multiplier
		else:
			par.Nactions = 1 # savings rate

		# e. quadrature
		_, par.psi_w = log_normal_gauss_hermite(1.0,par.Npsi)
		par.psi, _ = gauss_hermite(par.Npsi)
		_, par.xi_w = log_normal_gauss_hermite(1.0,par.Nxi)
		par.xi, _ = gauss_hermite(par.Nxi)

		par.psi_w = torch.tensor(par.psi_w,dtype=dtype,device=device)
		par.psi = torch.tensor(par.psi,dtype=dtype,device=device)
		par.xi_w = torch.tensor(par.xi_w,dtype=dtype,device=device)
		par.xi = torch.tensor(par.xi,dtype=dtype,device=device)

		# version with more shocks
		par.u = torch.tensor([],dtype=dtype,device=device)
		par.mix_psi = torch.tensor([],dtype=dtype,device=device)
		par.mix_xi = torch.tensor([],dtype=dtype,device=device)
		par.u_w = torch.tensor([],dtype=dtype,device=device)
		par.mix_psi_w = torch.tensor([],dtype=dtype,device=device)
		par.mix_xi_w = torch.tensor([],dtype=dtype,device=device)

		# d. simulation		
		sim.states = torch.zeros((par.T,sim.N,par.Nstates),dtype=dtype,device=device)
		sim.states_pd = torch.zeros((par.T,sim.N,par.Nstates_pd),dtype=dtype,device=device)
		sim.shocks = torch.zeros((par.T,sim.N,par.Nshocks),dtype=dtype,device=device)
		sim.outcomes = torch.zeros((par.T,sim.N,par.Noutcomes),dtype=dtype,device=device)
		sim.actions = torch.zeros((par.T,sim.N,par.Nactions),dtype=dtype,device=device)
		sim.reward = torch.zeros((par.T,sim.N),dtype=dtype,device=device)		
		
		sim.euler_error = torch.zeros((par.T,sim.N),dtype=dtype,device=device)
		sim.MPC = torch.zeros((par.T,sim.N),dtype=dtype,device=device)
		sim.R = np.nan # average expected discounted utility

	#########
	# train #
	#########

	def setup_train(self):
		""" default parameters for training """
		
		# a. unpack
		par = self.par
		train = self.train
		dtype = train.dtype
		device = train.device

		# b. neural network
		if not par.full:

			train.Nneurons_policy = np.array([100,100])
			train.Nneurons_value = np.array([100,100])
		
		train.manual_init_policy = False

		# c. policy activation functions and clipping
		if par.KKT:
			train.policy_activation_final = ['sigmoid','softplus'] # policy activation functions
			train.min_actions = torch.tensor([0.0,0.0],dtype=dtype,device=device) # minimum action value
			train.max_actions = torch.tensor([0.9999,np.inf],dtype=dtype,device=device) # maximum action value
		else:
			train.policy_activation_final = ['sigmoid']
			train.min_actions = torch.tensor([0.0],dtype=dtype,device=device)
			train.max_actions = torch.tensor([0.9999],dtype=dtype,device=device)
	
		# d. numerical integration
		train.use_quad = True # whether to use quadrature or monte carlo for numerical integration

		# e. misc
		train.terminal_actions_known = True # use terminal actions

		if par.KKT:
			train.epsilon_sigma = torch.tensor([0.1,0.0],dtype=dtype,device=device) # std of exploration shocks
			train.epsilon_sigma_min = torch.tensor([0.0,0.0],dtype=dtype,device=device)
		else:
			train.epsilon_sigma = torch.tensor([0.1],dtype=dtype,device=device)
			train.epsilon_sigma_min = torch.tensor([0.0],dtype=dtype,device=device)

		# f. algorithm specific
	
		# DeepSimulate - fraction of agents that explore
		train.explore_frac = torch.tensor([0.5*(1-t/(par.T-1)) for t in range(par.T)],dtype=dtype,device=device)
		# starts with 50% and goes down to 0% at the end of the life cycle
						   
		# DeepFOC
		if self.train.algoname in ['DeepFOC', 'DeepFOCBackward']:

			if par.KKT:
				train.eq_w = torch.tensor([10.0,1.0],dtype=dtype,device=device) # weight on FOC and on budget constraint
				if par.Nstates_fixed > 0:
					train.eq_w = torch.tensor([40.0,1.0],dtype=dtype,device=device) # weight on FOC and on budget constraint
			else:
				train.eq_w = torch.tensor([1.0],dtype=dtype,device=device)

			pass
			
	def allocate_train(self):
		""" allocate memory training """

		# a. unpack
		par = self.par
		train = self.train
		dtype = train.dtype
		device = train.device

		# a. dependent settings
		if not train.eq_w is None:
			train.eq_w = train.eq_w / torch.sum(train.eq_w) # normalize

		if train.time_input_type == 'manual':
			if train.Ninputs_time is None: train.Ninputs_time = par.T # if not used specified, default to time dummies setup

		if train.input_transformation:
			train.Ninputs_aux = 1

		# c. simulation
		train.states = torch.zeros((par.T,train.N,par.Nstates),dtype=dtype,device=device)
		train.states_pd = torch.zeros((par.T,train.N,par.Nstates_pd),dtype=dtype,device=device)
		train.shocks = torch.zeros((par.T,train.N,par.Nshocks),dtype=dtype,device=device)
		train.outcomes = torch.zeros((par.T,train.N,par.Noutcomes),dtype=dtype,device=device)
		train.actions = torch.zeros((par.T,train.N,par.Nactions),dtype=dtype,device=device)
		train.reward = torch.zeros((par.T,train.N),dtype=dtype,device=device)
				
	##############
	# quadrature #
	##############

	def numerical_integration(self):
		""" quadrature nodes and weights """

		# a. unpack
		par = self.par
		train = self.train
		dtype = train.dtype
		device = train.device

		# b. if quadrature is used
		if train.use_quad:

			# i. separate
			xi,psi = torch.meshgrid(par.xi,par.psi,indexing='ij')
			xi_w,psi_w = torch.meshgrid(par.xi_w,par.psi_w,indexing='ij')

			# ii. combined
			nodes = torch.stack((xi.flatten(),psi.flatten()),dim=-1) # shape (par.Nnumint,par.Nshocks)
			weigths = xi_w.flatten()*psi_w.flatten() # shape (par.Nnumint,)

			return 2.0**(1/2)*nodes,weigths

		# c. if monte carlo is used
		else:

			# i. antithetic sampling
			if par.do_antithetic_mc:
				
				assert par.Nmc % 2 == 0, 'number of mc draws must be even for antithetic sampling'
				N_draws_antithetic = par.Nmc//2 # number of antithetic draws

				# containers
				xi_antithetic = torch.zeros((par.T-1,train.N,par.Nmc),dtype=dtype,device=device) + torch.nan
				psi_antithetic = torch.zeros((par.T-1,train.N,par.Nmc),dtype=dtype,device=device) + torch.nan

				# mc draws for xi and psi
				xi_antithetic_draws = torch.normal(0.0,1.0,size=(par.T-1,train.N,N_draws_antithetic),dtype=dtype,device=device)
				psi_antithetic_draws = torch.normal(0.0,1.0,size=(par.T-1,train.N,N_draws_antithetic),dtype=dtype,device=device)

				# fill up containers by using that normal pdf is symmetric around 0
				xi_antithetic[..., :N_draws_antithetic] = xi_antithetic_draws
				xi_antithetic[..., N_draws_antithetic:] = -xi_antithetic_draws 

				psi_antithetic[..., :N_draws_antithetic] = psi_antithetic_draws
				psi_antithetic[..., N_draws_antithetic:] = -psi_antithetic_draws

				xi = xi_antithetic
				psi = psi_antithetic

			# ii. standard monte carlo sampling
			else:
				
				xi = torch.normal(0.0,1.0,size=(par.T-1,train.N,par.Nmc),dtype=dtype,device=device)
				psi = torch.normal(0.0,1.0,size=(par.T-1,train.N,par.Nmc),dtype=dtype,device=device)    
			
			# iii. combined
			nodes = torch.stack((xi,psi), dim=-1) # stack two vector with random draws -> shape (par.T-1,train.N,par.Nmc,par.Nshocks)
			weights = torch.ones((par.Nmc,)) / par.Nmc # shape (par.Nmc,)

			return nodes, weights            
	
	#########
	# draw #
	#########

	def draw_initial_states(self,N,training=False):
		""" draw initial state (m,p,t) """

		# a. unpack
		par = self.par

		# b. when not training
		fac = 1.0 if not training else par.explore_states_endo_fac
		fac_low = 1.0 if not training else par.explore_states_fixed_fac_low
		fac_high = 1.0 if not training else par.explore_states_fixed_fac_high

		# a. draw cash-on-hand
		sigma_m0 = par.sigma_m0*fac
		m0 = par.mu_m0*torch.exp(torch.normal(-0.5*sigma_m0**2,sigma_m0,size=(N,)))
		
		# d. draw permanent income
		sigma_p0 = par.sigma_p0*fac
		p0 = par.mu_p0*torch.exp(torch.normal(-0.5*sigma_p0**2,sigma_p0,size=(N,)))

		# e. draw sigma_xi
		if par.Nstates_fixed > 0: sigma_xi = torch_uniform(par.sigma_xi_low*fac_low,par.sigma_xi_high*fac_high,size=(N,))

		# f. draw sigma_psi
		if par.Nstates_fixed > 1: sigma_psi = torch_uniform(par.sigma_psi_low*fac_low,par.sigma_psi_high*fac_high,size=(N,))

		# g. draw rho_p  
		if par.Nstates_fixed > 2: rho_p = torch_uniform(par.rho_p_low*fac_low,np.fmin(par.rho_p_high*fac_high,1.0),size=(N,))
		
		# h. stack
		if par.Nstates_fixed == 0:
			return torch.stack((m0,p0),dim=-1)
		elif par.Nstates_fixed == 1:
			return torch.stack((m0,p0,sigma_xi),dim=-1)
		elif par.Nstates_fixed == 2:
			return torch.stack((m0,p0,sigma_xi,sigma_psi),dim=-1)
		elif par.Nstates_fixed == 3:
			return torch.stack((m0,p0,sigma_xi,sigma_psi,rho_p),dim=-1)
		
	def draw_shocks(self,N):
		""" draw shocks """

		# a. unpack
		par = self.par

		# b. draw shocks
		xi = torch.normal(0.0,1.0,size=(par.T,N))
		psi = torch.normal(0.0,1.0,size=(par.T,N))

		return torch.stack((xi,psi),dim=-1)

	def draw_exploration_shocks(self,epsilon_sigma,N):
		""" draw exploration shockss """

		# a. unpack
		par = self.par

		# b. draw exploration shocks
		eps = torch.zeros((par.T,N,par.Nactions))
		for i_a in range(par.Nactions):
			eps[:,:,i_a] = torch.normal(0,epsilon_sigma[i_a],(par.T,N))
	
		return eps

	def draw_endogenous_states(self):
		""" Draw endo states for pure backward """

		# a. unpack
		par = self.par
		train = self.train
		
		# b. draw endogenous states
		m_grid = torch.tensor(nonlinspace(0.4,6.0,400,1.3),device=train.device)
		for t in range(par.T):
			samples = sample_uniform_between_grid_points(m_grid,N=train.N,device=train.device)
			train.states[t,:,0] = samples

	###################
	# model functions #
	###################

	outcomes = model_funcs.outcomes
	reward = model_funcs.reward
	discount_factor = model_funcs.discount_factor
	
	terminal_actions = model_funcs.terminal_actions
	terminal_reward_pd = model_funcs.terminal_reward_pd
	terminal_marginal_reward_pd = model_funcs.terminal_marginal_reward_pd
		
	state_trans_pd = model_funcs.state_trans_pd
	state_trans = model_funcs.state_trans
	exploration = model_funcs.exploration

	marginal_reward = model_funcs.marginal_reward
	eval_equations_FOC = model_funcs.eval_equations_FOC
	eval_equations_FOC_terminal = model_funcs.eval_equations_FOC_terminal
	eval_equations_VPD = model_funcs.eval_equations_VPD
	eval_equations_VPD_terminal = model_funcs.eval_equations_VPD_terminal
	eval_equations_FOC_t = model_funcs.eval_equations_FOC_t

	def add_transfer(self,transfer):
		""" add transfer to initial states """

		# a. unpack
		par = self.par
		sim = self.sim

		# b. add transfer
		sim.states[0,:,0] += transfer
	
	############
	# simulate #
	############

	def compute_MPC(self,Nbatch_share=0.01):
		""" compute MPC """

		# a. unpack
		par = self.par
		sim = self.sim
		
		with torch.no_grad():

			# i. baseline
			c = sim.outcomes[...,0]

			# ii. add windfall
			states = deepcopy(sim.states)
			states[:,:,0] += par.Delta_MPC

			# iii. alternative
			c_alt = torch.ones_like(c)
			c_alt[-1] = c[-1] + par.Delta_MPC
			Nbatch = int(Nbatch_share*sim.N)

			for i in range(0,sim.N,Nbatch):

				index_start = i
				index_end = i + Nbatch
						
				states_ = states[:par.T-1,index_start:index_end,:]
				actions = self.eval_policy(self.policy_NN,states_)
				outcomes = self.outcomes(states_,actions)
				c_alt[:par.T-1,index_start:index_end] = outcomes[:,:,0]

			# iv. MPC
			sim.MPC[:,:] = (c_alt-c)/par.Delta_MPC	

	def compute_euler_errors(self,Nbatch_share=0.01):
		""" compute euler error"""

		# a. unpack
		par = self.par
		sim = self.sim
		train = self.train

		# b. ?
		reset = False
		if not train.use_quad:

			reset = True
			Nnumint = train.Nnumint
			numint_nodes = train.numint_nodes
			numint_weights = train.numint_weights
			train.use_quad = True
			self._numerical_integration()

		Nbatch = int(Nbatch_share*sim.N)

		# c. compute euler error over batches
		for i in range(0,sim.N,Nbatch):

			index_start = i
			index_end = i + Nbatch

			with torch.no_grad():
				
				# i. get consumption and states today
				c = sim.outcomes[:par.T-1,index_start:index_end,0]
				states = sim.states[:par.T-1,index_start:index_end]
				actions = sim.actions[:par.T-1,index_start:index_end]
				outcomes = sim.outcomes[:par.T-1,index_start:index_end]

				# ii. post-decision states
				states_pd = self.state_trans_pd(states,actions,outcomes)

				# iii. next-period states
				states_next = self._state_trans(states_pd)

				# iv. next-period action
				actions_next_before = self.eval_policy(self.policy_NN,states_next[:par.T-2],t0=1)
				actions_next_after = self.terminal_actions(states_next[par.T-2])
				actions_next = torch.cat((actions_next_before,actions_next_after[None,...]),dim=0)

				# v. next-period consumption
				c_next = self.outcomes(states_next,actions_next).squeeze(dim=-1)

				# vi. marginal utility next period
				marg_util_next = model_funcs.marg_util_c(c_next)

				# vii. expected marginal utility next period
				exp_marg_util_next = torch.sum(train.numint_weights[None,None,:]*marg_util_next, dim=-1)

				# viii. euler error
				euler_error_Nbatch = model_funcs.inverse_marg_util(par.R*par.beta*exp_marg_util_next) / c - 1
				euler_error_Nbatch = torch.abs(euler_error_Nbatch)
				sim.euler_error[:par.T-1,index_start:index_end] = euler_error_Nbatch

		# d. ?
		if reset:

			train.use_quad = False
			train.Nnumint = Nnumint
			train.numint_nodes = numint_nodes
			train.numint_weights = numint_weights

	########
	# misc #
	########

	def compute_policy_on_grids(self,egm,transformation_function):
		""" 
		compute state-network on EGM grids

		NN: state-dependent Neural net that you want to get computed in grids
		transformation_function: function that takes state and action and yields transformation that you want to put on grid	
		
		"""

		# a. unpack
		par = self.par
		train = self.train

		# b. get egm grids
		m_grid = egm.m_grid
		p_grid = egm.p_grid
		sigma_xi_grid = egm.sigma_xi_grid
		sigma_psi_grid = egm.sigma_psi_grid
		rho_p_grid = egm.rho_p_grid

		# c. create array for storing values
		egm_sol_shape = egm.sol_con.shape
		sol_con_grid = np.zeros(egm_sol_shape)

		# d. create tensor product data
		if par.Nstates_fixed == 0:
			p, m = np.meshgrid(p_grid,m_grid,indexing='ij')
			states_grid = np.concatenate((m.reshape(-1,1),p.reshape(-1,1)),axis=1)
		elif par.Nstates_fixed == 1:
			p,sigma_xi,m = np.meshgrid(p_grid,sigma_xi_grid,m_grid,indexing='ij')
			states_grid = np.concatenate((m.reshape(-1,1),p.reshape(-1,1),sigma_xi.reshape(-1,1)),axis=1)
		elif par.Nstates_fixed == 2:
			p,sigma_xi,sigma_psi,m = np.meshgrid(p_grid,sigma_xi_grid, sigma_psi_grid,m_grid,indexing='ij')
			states_grid = np.concatenate((m.reshape(-1,1),p.reshape(-1,1),sigma_xi.reshape(-1,1),sigma_psi.reshape(-1,1)),axis=1)
		elif par.Nstates_fixed == 3:
			p,sigma_xi,sigma_psi,rho_p,m = np.meshgrid(p_grid,sigma_xi_grid, sigma_psi_grid,rho_p_grid, m_grid,indexing='ij')
			states_grid = np.concatenate((m.reshape(-1,1),p.reshape(-1,1),sigma_xi.reshape(-1,1),sigma_psi.reshape(-1,1), rho_p.reshape(-1,1)),axis=1)			

		# e. compute
		states_grid = torch.tensor(states_grid,dtype=train.dtype,device=train.device)
		for t in range(par.T-1):
			
			# i. evaluate
			with torch.no_grad(): 
				output = self.eval_policy(self.policy_NN,states_grid,t=t)

			# ii. transformation
			output = transformation_function(states_grid,output).cpu().numpy()

			# iii. store
			for i_p in range(egm.Np):
				for i_sigma_psi in range(egm.Nsigma_psi):
					for i_sigma_xi in range(egm.Nsigma_xi):
						for i_rho_p in range(egm.Nrho_p):
							m_index = i_rho_p + egm.Nrho_p*i_sigma_psi + egm.Nrho_p*egm.Nsigma_psi*i_sigma_xi+egm.Nrho_p*egm.Nsigma_psi*egm.Nsigma_xi*i_p
							sol_con_grid[t,i_p,i_sigma_xi,i_sigma_psi,i_rho_p,:] = output[egm.Nm*m_index:egm.Nm*(m_index+1)]

		return sol_con_grid	
	
	def compute_model_moments(self):
		
		train = self.train
		sim = self.sim
		info = self.info
		
		info[('mean_savings',train.k)] = sim.states_pd[...,0].mean(axis=-1)
		
	##########
	# inputs #
	##########

	def time_inputs(self,time_indices):
		""" create time input for neural network """

		# a. unpack
		par = self.par
		train = self.train

		if train.time_input_type == 'manual': # for continuous time input with retirement dummy, must set train.Ninputs_time = 2
			if train.Ninputs_time != 2:
				raise ValueError('for manual time input with retirement dummy, train.Ninputs_time must be set to 2')

			# i. initialize dummy tensor and fill out for retired agents
			retirement_dummy = torch.zeros_like(time_indices,dtype=torch.float32).to(train.device)
			retirement_dummy[time_indices >= par.T_retired] = 1.0

			# ii. normalize time indices to [0,1]
			time_indices = time_indices.to(torch.float32) / (par.T - 1)
		
			# iii. stack time indices and retirement dummy
			time_inputs = torch.stack([time_indices, retirement_dummy],dim=-1)

		else: # default is time dummies
			
			# i. create dummies
			time_inputs = F.one_hot(time_indices,par.T)

		return time_inputs

	def input_transformation(self,x,t=None,t0=0):
		""" input transformation for neural network """

		new_x = torch.zeros_like(x[...,0])        

		return torch.concat((x,new_x[...,None]),dim=-1)

	def manual_init_policy(self, layers):
		""" manual initialization of neural networks in policy """

		# a. unpack
		par = self.par
		train = self.train
		dtype = train.dtype
		device = train.device

		# b. bias initialization for sigmoid layers 
		if par.init_bias_logodds:
			b = math.log(par.w_p/(1-par.w_p)) # output layer will output wp in the beginning of training
			layers[-1].bias.data = torch.ones_like(layers[-1].bias.data, dtype=dtype, device=device)*b
		
		# c. xavier initialization for sigmoid layers
		if par.init_sigmoid_xavier_uniform:
			torch.nn.init.xavier_uniform_(layers[-1].weight.data)
		
		if par.init_sigmoid_xavier_normal:
			torch.nn.init.xavier_normal_(layers[-1].weight.data)

		# d. he initialization for ReLU layers
		if par.init_relu_he_uniform:
			for layer in layers[:-1]:
				torch.nn.init.kaiming_uniform_(layer.weight.data, nonlinearity='relu')
		
		if par.init_relu_he_normal:
			for layer in layers[:-1]:
				torch.nn.init.kaiming_normal_(layer.weight.data, nonlinearity='relu')

def select_euler_errors(model):
	""" compute mean euler error """

	# a. unpack
	par = model.par
	sim = model.sim

	# b. compute booleans for households being unconstrainted
	savings_indicator = (sim.actions[:par.T_retired,:,0] > par.Euler_error_min_savings)

	# c. return euler errors for unconstrained households
	return sim.euler_error[:par.T_retired,:][savings_indicator]

def mean_log10_euler_error_working(model):

	# a. extract euler errors
	euler_errors = select_euler_errors(model).cpu().numpy()
	
	# b. compute booleans for households have near zero euler error
	I = np.isclose(np.abs(euler_errors),0.0)

	# c. compute mean of logged euler errors different from zero
	return np.mean(np.log10(np.abs(euler_errors[~I])))

def sample_uniform_between_grid_points(m_grid,N,device):

	# a. Uniformly choose one of the M-1 intervals
	M = len(m_grid)
	interval_idx = torch.randint(low=0,high=M-1,size=(N,),device=device)

	# b. Draw uniform value for interpolation
	u = torch.rand(N,device=device)

	# c. Get left and right grid points
	left = m_grid[interval_idx]
	right = m_grid[interval_idx+1]

	# d. Interpolate within the interval
	samples = (1-u)*left+u*right

	return samples

##############
# MoreShocks #
##############

class BufferStockMoreShocksModelClass(BufferStockModelClass):
	""" Buffer stock model with more shocks in income process """

	def setup(self,full=None):
		""" choose parameters """

		# a. unpack
		par = self.par

		# b. call regular model
		super().setup(full=full)
		assert par.Nstates_fixed == 0, 'BufferStockMoreShocksModel does not support fixed states'

		# income
		par.MoreShocks = True # whether to use more complex shock structure
		
		par.sigma_xi_1 = 0.07 # shock, permanent , std, case 1
		par.sigma_xi_2 = 0.10 # shock, permanent , std, case 2
		par.p_xi = 0.20 # mixture probability for xi1 and xi2
		par.Nxi = 4 # number of qudrature nodes

		par.sigma_psi_1 = 0.07 # shock, transitory std, case 1
		par.sigma_psi_2 = 0.1 # shock, transitory std, case 2
		par.p_psi = 0.20 # mixture probability for psi1 and psi2
		par.Npsi = 4 # number of qudrature nodes

		# d. unemployment
		par.p_u = 0.1 # probability of unemployment
		par.u_replacement = 0.5 # replacement rate in case of unemployment

		par.Nshocks = 5 # number of shocks

		par.use_sobol = False # whether to use Sobol QMC for numerical integration

	def allocate(self):
		""" allocate arrays  """

		# a. call regular model
		super().allocate()

		# b. unpack
		par = self.par
		train = self.train
		dtype = train.dtype
		device = train.device

		# c. allocate
		par.u = torch.tensor([1.0,0.0],dtype=dtype,device=device) # unemployment states
		par.mix_psi = torch.tensor([1.0,0.0],dtype=dtype,device=device) # mixture for psi
		par.mix_xi = torch.tensor([1.0,0.0],dtype=dtype,device=device) # mixture for xi
		par.u_w = torch.tensor([par.p_u,1.0-par.p_u],dtype=dtype,device=device) # unemployment weights
		par.mix_psi_w = torch.tensor([par.p_psi,1.0-par.p_psi],dtype=dtype,device=device) # mixture for psi weights
		par.mix_xi_w = torch.tensor([par.p_xi,1.0-par.p_xi],dtype=dtype,device=device) # mixture for xi weights

	def draw_shocks(self,N):
		""" draw shocks """

		# a. unpack
		par = self.par
		train = self.train

		# b. persistent income shocks
		xi = torch.normal(0,1.0,size=(par.T,N))

		# c. transitory income shocks
		psi = torch.normal(0,1.0,size=(par.T,N))

		# c. unemployment
		u = torch.rand((par.T,N),dtype=train.dtype) # uniform random number for unemployment

		# d. mixture for xi
		mix_xi = torch.rand((par.T,N),dtype=train.dtype) # mixture for xi

		# e. mixture for psi
		mix_psi = torch.rand((par.T,N),dtype=train.dtype) # mixture for psi

		return torch.stack((xi,psi,u,mix_xi,mix_psi),dim=-1)

	def numerical_integration(self):
		""" quadrature nodes and weights """

		# a. unpack
		par = self.par
		train = self.train
		dtype = train.dtype
		device = train.device

		# b. if quadrature is used
		if train.use_quad:

			# i. separate
			xi, psi, u, mix_xi, mix_psi = torch.meshgrid(
				par.xi, par.psi, par.u, par.mix_xi, par.mix_psi, indexing='ij'
			)
			xi_w, psi_w, u_w, mix_xi_w, mix_psi_w = torch.meshgrid(
				par.xi_w, par.psi_w, par.u_w, par.mix_xi_w, par.mix_psi_w, indexing='ij'
			)

			# ii. combined
			nodes = torch.stack(
				(xi.flatten(), psi.flatten(), u.flatten(), mix_xi.flatten(), mix_psi.flatten()),
				dim=-1
			)
			weights = (
				xi_w.flatten()
				* psi_w.flatten()
				* u_w.flatten()
				* mix_xi_w.flatten()
				* mix_psi_w.flatten()
			)

			nodes = nodes.to(device=device, dtype=dtype)
			weights = weights.to(device=device, dtype=dtype)

			return nodes, weights

		# c. if Sobol QMC is used
		elif getattr(par, "use_sobol", False):

			# dimension of random vector: (xi, psi, u, mix_xi, mix_psi)
			dim = 5

			# we’ll do Sobol + antithetic, so Nmc must be even
			assert par.Nmc % 2 == 0, "Nmc must be even for Sobol antithetic sampling"
			n_base = par.Nmc // 2

			# i. Sobol points in [0,1]^5
			sobol = SobolEngine(dimension=dim, scramble=True)
			u_base = sobol.draw(n_base)  # (n_base, 5), float32 on CPU
			u_base = u_base.to(device=device, dtype=dtype)

			# ii. Antithetic pairs in [0,1]^5: u and 1-u
			u_full = torch.cat([u_base, 1.0 - u_base], dim=0)  # (Nmc, 5)

			# iii. Map first two dims to N(0,1), last three stay uniform
			normal = torch.distributions.Normal(
				torch.tensor(0.0, device=device, dtype=dtype),
				torch.tensor(1.0, device=device, dtype=dtype),
			)

			xi_1d      = normal.icdf(u_full[:, 0])  # (Nmc,)
			psi_1d     = normal.icdf(u_full[:, 1])  # (Nmc,)
			u_1d       = u_full[:, 2]               # (Nmc,)
			mix_xi_1d  = u_full[:, 3]               # (Nmc,)
			mix_psi_1d = u_full[:, 4]               # (Nmc,)

			# iv. Broadcast to (T-1, N, Nmc)
			shape = (par.T - 1, train.N, par.Nmc)

			def expand_1d_to_3d(x):
				return x.view(1, 1, -1).expand(shape)

			xi      = expand_1d_to_3d(xi_1d)
			psi     = expand_1d_to_3d(psi_1d)
			u       = expand_1d_to_3d(u_1d)
			mix_xi  = expand_1d_to_3d(mix_xi_1d)
			mix_psi = expand_1d_to_3d(mix_psi_1d)

			# v. Combined nodes and equal weights (QMC)
			nodes = torch.stack((xi, psi, u, mix_xi, mix_psi), dim=-1)  # (T-1, N, Nmc, 5)
			weights = torch.ones((par.Nmc,), device=device, dtype=dtype) / par.Nmc

			return nodes, weights

		# d. if (pseudo-)Monte Carlo is used
		else:

			# ia. antithetic sampling
			if par.do_antithetic_mc:

				assert par.Nmc % 2 == 0, 'number of mc draws must be even for antithetic sampling'
				N_draws_antithetic = par.Nmc // 2  # number of antithetic draws

				# containers
				xi_antithetic = torch.zeros(
					(par.T - 1, train.N, par.Nmc), dtype=dtype, device=device
				)
				psi_antithetic = torch.zeros(
					(par.T - 1, train.N, par.Nmc), dtype=dtype, device=device
				)

				# mc draws for xi and psi
				xi_antithetic_draws = torch.normal(
					0.0, 1.0, size=(par.T - 1, train.N, N_draws_antithetic),
					dtype=dtype, device=device
				)
				psi_antithetic_draws = torch.normal(
					0.0, 1.0, size=(par.T - 1, train.N, N_draws_antithetic),
					dtype=dtype, device=device
				)

				# fill up containers using symmetry
				xi_antithetic[..., :N_draws_antithetic] = xi_antithetic_draws
				xi_antithetic[..., N_draws_antithetic:] = -xi_antithetic_draws

				psi_antithetic[..., :N_draws_antithetic] = psi_antithetic_draws
				psi_antithetic[..., N_draws_antithetic:] = -psi_antithetic_draws

				xi = xi_antithetic
				psi = psi_antithetic

			# ib. standard monte carlo sampling
			else:

				xi = torch.normal(
					0.0, 1.0, size=(par.T - 1, train.N, par.Nmc),
					dtype=dtype, device=device
				)
				psi = torch.normal(
					0.0, 1.0, size=(par.T - 1, train.N, par.Nmc),
					dtype=dtype, device=device
				)

			# ii. mc draws for u, mix_xi and mix_psi
			u = torch.rand((par.T - 1, train.N, par.Nmc), dtype=dtype, device=device)
			mix_xi = torch.rand((par.T - 1, train.N, par.Nmc), dtype=dtype, device=device)
			mix_psi = torch.rand((par.T - 1, train.N, par.Nmc), dtype=dtype, device=device)

			# iii. combined
			nodes = torch.stack((xi, psi, u, mix_xi, mix_psi), dim=-1)  # (T-1, N, Nmc, 5)
			weights = torch.ones((par.Nmc,), device=device, dtype=dtype) / par.Nmc

			return nodes, weights        

	state_trans = model_funcs_more_shocks.state_trans