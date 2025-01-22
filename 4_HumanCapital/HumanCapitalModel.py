import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE' # without this python may crash when plotting from matplotlib
from copy import deepcopy
import numpy as np
import torch
torch.set_warn_always(True)

from consav.quadrature import log_normal_gauss_hermite, gauss_hermite

# local
import model_funcs
from EconDLSolvers import DLSolverClass, torch_uniform

# class
class HumanCapitalModelClass(DLSolverClass):
	
	#########
	# setup #
	#########

	def setup(self,full=None):
		""" choose parameters """

		par = self.par
		sim = self.sim
		
		par.full = full if not full is None else torch.cuda.is_available()
		par.seed = 1 # seed for random number generator in torch

		# a. model
		par.T = 5 # number of periods

		# preferences
		par.beta = 1/1.01 # discount factor

		# labor disutility
		par.nu = 2.0
		par.vphi = 0.8

		# income
		par.y_base = 1.0 # base
		par.y_growth = 0.02 # income growth
		par.y_growth_decay = 0.1 # income growth decay
		par.alpha = 0.1 # human capital accumulation

		par.rho_xi = 0.95 # shock, persistence
		par.sigma_xi = 0.1 # shock, permanent , std
		par.Nxi = 4 # number of qudrature nodes

		par.sigma_psi = 0.1 # shock, transitory std
		par.Npsi = 4 # number of qudrature nodes

		# return
		par.R = 1.00 # gross return

		# b. solver settings

		# states and shocks
		par.Nstates = 4
		par.Nstates_pd = 3
		par.Nshocks = 2 # number of shocks

		# outcomes and actions
		par.Noutcomes = 3 # number of outcomes (here just c)
		par.KKT = False # use KKT conditions (for DeepFOC)
		par.NDC = 0 # number of discrete choices

		# scaling
		par.m_scaler = 1/10.0
		par.p_scaler = 1/5.0

		# Euler
		par.Euler_error_min_savings = 1e-3 # minimum savings rate for computing Euler error
		par.Delta_MPC = 1e-4 # windfall used in MPC calculation

		# c. simulation 
		sim.N = 100_000 # number of agents
		sim.reps = 10 # number of repetitions

		# initial states
		par.mu_m0 = 0.1 # initial cash-on-hand, mean
		par.sigma_m0 = 0.1 # initial cash-on-hand, std

		# initial permanent income
		par.mu_p0 = 1.0 # initial durable, mean
		par.sigma_p0 = 0.1 # initial durable, std

	def allocate(self):
		""" allocate arrays  """

		# a. unpack
		par = self.par
		sim = self.sim
		train = self.train

		dtype = train.dtype
		device = train.device

		if not par.full: # for solving without GPU
			par.T = 3
			sim.N = 10_000
		
		# a. life cycle income
		par.y = torch.zeros(par.T,dtype=dtype,device=device)	
		par.y[0] = par.y_base
	
		for t in range(1,par.T):
			par.y[t] = par.y[t-1]*(1+par.y_growth*(1-par.y_growth_decay)**(t-1))

		# b. states, shocks and actions
		if par.KKT: 
			par.Nactions = 3
		else:
			par.Nactions = 2

		# c. quadrature
		par.psi,par.psi_w = log_normal_gauss_hermite(par.sigma_psi,par.Npsi)
		par.xi,par.xi_w = log_normal_gauss_hermite(par.sigma_xi,par.Nxi)

		par.psi_w = torch.tensor(par.psi_w,dtype=dtype,device=device)
		par.psi = torch.tensor(par.psi,dtype=dtype,device=device)
		par.xi_w = torch.tensor(par.xi_w,dtype=dtype,device=device)
		par.xi = torch.tensor(par.xi,dtype=dtype,device=device)

		# d. scaling vector - add ones for time dummies
		par.scale_vec_states = torch.tensor([par.m_scaler,par.p_scaler],dtype=dtype,device=device)
		par.scale_vec_states_pd = torch.tensor([par.m_scaler,par.p_scaler],dtype=dtype,device=device)

		# e. simulation		
		sim.states = torch.zeros((par.T,sim.N,par.Nstates),dtype=dtype,device=device)
		sim.states_pd = torch.zeros((par.T,sim.N,par.Nstates_pd),dtype=dtype,device=device)
		sim.shocks = torch.zeros((par.T,sim.N,par.Nshocks),dtype=dtype,device=device)
		sim.outcomes = torch.zeros((par.T,sim.N,par.Noutcomes),dtype=dtype,device=device)
		sim.actions = torch.zeros((par.T,sim.N,par.Nactions),dtype=dtype,device=device)
		sim.reward = torch.zeros((par.T,sim.N),dtype=dtype,device=device)		
		
		sim.euler_error = torch.zeros((par.T,sim.N),dtype=dtype,device=device)
		sim.MPC = torch.zeros((par.T,sim.N),dtype=dtype,device=device)
		sim.R = np.nan

	#########
	# train #
	#########

	def setup_train(self):
		""" default parameters for training """
		
		par = self.par
		train = self.train
		dtype = train.dtype
		device = train.device

		# a. neural network
		if not par.full:

			train.Nneurons_policy = np.array([100,100])
			train.Nneurons_value = np.array([100,100])
		# b. policy activation functions and clipping
		if par.KKT:
			train.policy_activation_final = ['sigmoid','sigmoid','softplus'] # policy activation functions
			train.min_actions = torch.tensor([0.0,0.0,0.0],dtype=dtype,device=device) # minimum action value
			train.max_actions = torch.tensor([0.9999,1.0-1e-6,np.inf],dtype=dtype,device=device) # maximum action value
		else:
			train.policy_activation_final = ['sigmoid','sigmoid']
			train.min_actions = torch.tensor([0.0,0.0],dtype=dtype,device=device)
			train.max_actions = torch.tensor([0.9999,1.0-1e-6],dtype=dtype,device=device)
		
		# c. misc
		train.terminal_actions_known = False # use terminal actions
		train.use_input_scaling = False # use input scaling
		train.do_exo_actions = -1 

		# d. algorithm specific
	
		# DeepSimulate
		if self.train.algoname == 'DeepSimulate':
			pass

		else:


			if par.KKT:
				train.epsilon_sigma = np.array([0.1,0.1,0.0]) # std of exploration shocks
				train.epsilon_sigma_min = np.array([0.0,0.0,0.0])
			else:
				train.epsilon_sigma = np.array([0.1,0.1])
				train.epsilon_sigma_min = np.array([0.0,0.0])

		# DeepFOC
		if self.train.algoname == 'DeepFOC':
			train.eq_w = torch.tensor([3.0, 3.0, 5.0],dtype=dtype,device=device)

			pass
			
		# DeepVPD
		if train.algoname == 'DeepVPD':
			train.learning_rate_value_decay = 0.999
			train.learning_rate_policy_decay = 0.999
			train.tau = 0.4
			# train.N_value_NN = 3
			train.start_train_policy = 50


			pass

		### DeepV
		if train.algoname == 'DeepV':

			pass

		### DeepQ
		if train.algoname == 'DeepQ':

			pass

	def allocate_train(self):
		""" allocate memory training """

		par = self.par
		train = self.train
		dtype = train.dtype
		device = train.device

		# a. dependent settings
		pass

		# b. simulation
		train.states = torch.zeros((par.T,train.N,par.Nstates),dtype=dtype,device=device)
		train.states_pd = torch.zeros((par.T,train.N,par.Nstates_pd),dtype=dtype,device=device)
		train.shocks = torch.zeros((par.T,train.N,par.Nshocks),dtype=dtype,device=device)
		train.outcomes = torch.zeros((par.T,train.N,par.Noutcomes),dtype=dtype,device=device)
		train.actions = torch.zeros((par.T,train.N,par.Nactions),dtype=dtype,device=device)
		train.reward = torch.zeros((par.T,train.N),dtype=dtype,device=device)
				
	##############
	# quadrature #
	##############

	def quad(self):
		""" quadrature nodes and weights """

		par = self.par

		xi,psi = torch.meshgrid(par.xi,par.psi, indexing='ij')
		xi_w,psi_w = torch.meshgrid(par.xi_w,par.psi_w, indexing='ij')
		
		quad = torch.stack((xi.flatten(),psi.flatten()),dim=1)
		quad_w = xi_w.flatten()*psi_w.flatten() 

		return quad,quad_w
		
	#########
	# draw #
	#########

	def draw_initial_states(self,N,training=False):
		""" draw initial state (m,p,t) """

		par = self.par

		# a. draw financial wealth
		b0 = par.mu_m0*np.exp(torch.normal(-0.5*par.sigma_m0**2,par.sigma_m0,size=(N,)))
		
		# b. draw permanent income
		p0 = par.mu_p0*np.exp(torch.normal(-0.5*par.sigma_p0**2,par.sigma_p0,size=(N,)))

		# c. draw human capital 
		hk0 = torch.zeros((N,))

		# d. transitory shocks
		psi0 = torch.normal(0.0,1.0,size=(N,))
		psi0 = torch.exp(par.sigma_psi*psi0-0.5*par.sigma_psi**2)

		# f. store
		return torch.stack((b0,p0,hk0,psi0),dim=-1)
		
	def draw_shocks(self,N):
		""" draw shocks """

		par = self.par

		# xi 
		xi_loc = -0.5*par.sigma_xi**2
		xi = np.exp(torch.normal(xi_loc,par.sigma_xi,size=(par.T,N,)))

		# psi
		psi_loc = -0.5*par.sigma_psi**2
		psi = np.exp(torch.normal(psi_loc,par.sigma_psi,size=(par.T,N,)))

		return torch.stack((xi,psi),dim=-1)

	def draw_exploration_shocks(self,epsilon_sigma,N):
		""" draw exploration shockss """

		par = self.par

		eps = torch.zeros((par.T,N,par.Nactions))
		for i_a in range(par.Nactions):
			eps[:,:,i_a] = torch.normal(0,epsilon_sigma[i_a],(par.T,N))
	
		return eps

	def draw_exo_actions(self,N):
		""" draw exogenous actions """

		par = self.par

		exo_actions = torch_uniform(0.01,0.8,size=(par.T,N,par.Nactions))
	
		return exo_actions

	###################
	# model functions #
	###################

	outcomes = model_funcs.outcomes
	reward = model_funcs.reward
	discount_factor = model_funcs.discount_factor
	
	terminal_reward_pd = model_funcs.terminal_reward_pd
		
	state_trans_pd = model_funcs.state_trans_pd
	state_trans = model_funcs.state_trans
	exploration = model_funcs.exploration

	#marginal_reward = model_funcs.marginal_reward
	eval_equations_FOC = model_funcs.eval_equations_FOC
		