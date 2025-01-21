import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE' # without this python may crash when plotting from matplotlib
import numpy as np
import torch
torch.set_warn_always(False)

from consav.quadrature import log_normal_gauss_hermite

from EconDLSolvers import DLSolverClass

# local
import model_funcs

class NonConvexDurablesModelClass(DLSolverClass):

	def setup(self,full=None):
		""" choose parameters """

		par = self.par
		sim = self.sim

		par.full = full if full is not None else torch.cuda.is_available()

		# a. model
		par.seed = 1 # seed for random number generator in torch

		# horizon
		if par.full:
			par.T = 15 # number of periods
		else:
			par.T = 15 # number of periods

		# preferences
		par.beta = 0.965 # discount factor
		par.alpha = 0.9 # weight on non-durable consumption
		par.d_ubar = 1e-2 # minimum consumption
		# par.d_ubar = 0.0 # minimum consumption
		par.rho = 2.0 # risk aversion

		# return, durable good and ince income
		par.R = 1.03 # gross return
		par.kappa = 0.1 # adjustment cost parameter
		par.delta = 0.15 # depreciation rate
		
		par.sigma_xi = 0.1 # std of persistent income shock
		par.sigma_psi = 0.1 # std of transitory income shock
		par.eta = 0.95 # persistence of permanent income

		par.Nxi = 4 # number of persistent income shocks - quadrature
		par.Npsi = 4 # number of transitory income shocks - quadrature

		# taste shocks
		# par.sigma_eps = 0.000001
		par.sigma_eps = 0.1
		# par.sigma_eps = 0.2

		# number of states, shocks and actions
		par.Nstates = 3 # number of states
		par.Nstates_pd = 3 # number of post-decision states
		par.Nshocks = 2 # number of shocks
		par.Nactions = 3 # number of continuous choices
		par.Noutcomes = 4 # number of outcomes
		par.NDC = 2 # number of discrete choices

		# b. simulation of life-time-reward
		sim.N = 50_000 # number of agents

		# initial states
		par.mu_m0 = 1.0 
		par.sigma_m0 = 0.1 
		par.mu_p0 = 1.0 
		par.sigma_p0 = 0.1
		par.mu_n0 = 0.0
		par.sigma_n0 = 0.01		

	def allocate(self):
		""" allocate arrays  """

		# a. unpack
		par = self.par
		sim = self.sim
		train = self.train
		dtype = train.dtype	
		device = train.device

		# b. quad
		par.xi, par.xi_w = log_normal_gauss_hermite(par.sigma_xi, par.Nxi)
		par.psi, par.psi_w = log_normal_gauss_hermite(par.sigma_psi, par.Npsi)

		par.psi_w = torch.tensor(par.psi_w,dtype=dtype,device=device)
		par.psi = torch.tensor(par.psi,dtype=dtype,device=device)
		par.xi_w = torch.tensor(par.xi_w,dtype=dtype,device=device)
		par.xi = torch.tensor(par.xi,dtype=dtype,device=device)		

		# c. simulation
		sim.states = torch.zeros((par.T,sim.N,par.Nstates),dtype=dtype,device=device)
		sim.states_pd = torch.zeros((par.T,sim.N,par.Nstates_pd),dtype=dtype,device=device)
		sim.shocks = torch.zeros((par.T,sim.N,par.Nshocks),dtype=dtype,device=device)
		sim.outcomes = torch.zeros((par.T,sim.N,par.Noutcomes),dtype=dtype,device=device) 
		sim.actions = torch.zeros((par.T,sim.N,par.Nactions),dtype=dtype,device=device) 
		sim.reward = torch.zeros((par.T,sim.N,par.NDC),dtype=dtype,device=device)
		
		sim.taste_shocks = torch.zeros((par.T,sim.N,par.NDC),dtype=dtype,device=device)
		par.gumbell_param = get_scale_from_variance(par.sigma_eps)
		sim.DC = torch.zeros((par.T,sim.N),dtype=dtype,device=device)
		
		sim.adj = torch.zeros((par.T,sim.N),dtype=dtype,device=device)
		sim.c = torch.zeros((par.T,sim.N),dtype=dtype,device=device)
		sim.d = torch.zeros((par.T,sim.N),dtype=dtype,device=device)


		

	#########
	# train #
	#########

	def setup_train(self):
		""" default parameters for training """
		
		train = self.train
		dtype = train.dtype
		device = train.device

		# a. neural net settings
		train.Nneurons_value = np.array([350,350]) # number of neurons in each layer
		train.Nneurons_policy = np.array([350,350]) # number of neurons in each layer
		train.policy_activation_final = ['sigmoid','sigmoid','sigmoid'] # activations in policy net
		
		# b. exploration
		train.epsilon_sigma = np.array([0.10,0.10,0.10]) # std of exploration shocks
		train.epsilon_sigma_decay = 1.0 # decay of epsilon_sigma
		train.epsilon_sigma_min = np.array([0.0,0.0,0.0]) # minimum epsilon_sigma
		
		# c. clipping
		train.min_actions = torch.tensor([0.0,0.0,0.0],dtype=dtype,device=device) # minimum action value
		train.max_actions = torch.tensor([0.9999,0.9999,0.9999],dtype=dtype,device=device) # maximum action value

		# d. misc
		train.start_train_policy = 50

		train.N_value_NN = 3

		train.use_FOC=False
		train.NFOC_targets = 1
		
	def allocate_train(self):
		""" allocate memory training """

		train = self.train
		par = self.par
		train = self.train
		detype = train.dtype
		device = train.device

		# a. numpy - data
		train.states = torch.zeros((par.T,train.N,par.Nstates),dtype=detype,device=device)
		train.states_pd = torch.zeros((par.T,train.N,par.Nstates_pd),dtype=detype,device=device)
		train.shocks = torch.zeros((par.T,train.N,par.Nshocks),dtype=detype,device=device)
		train.outcomes = torch.zeros((par.T,train.N,par.Noutcomes),dtype=detype,device=device)
		train.actions = torch.zeros((par.T,train.N,par.Nactions),dtype=detype,device=device) 
		train.reward = torch.zeros((par.T,train.N,par.NDC),dtype=detype,device=device)

		train.taste_shocks = torch.zeros((par.T,train.N,par.NDC),dtype=detype,device=device)
		train.DC = torch.zeros((par.T,train.N),dtype=detype,device=device)
		
	#########
	# draw #
	#########

	def draw_initial_states(self,N,training=False):
		""" draw initial state (m,p,t) """

		par = self.par

		# a. draw cash-on-hand
		m0 = par.mu_m0*torch.exp(torch.normal(-0.5*par.sigma_m0**2,par.sigma_m0,size=(N,)))

		# b. draw persistent income shock
		p0 = par.mu_p0*torch.exp(torch.normal(-0.5*par.sigma_p0**2,par.sigma_p0,size=(N,)))

		# c. draw durable
		n0 = par.mu_n0*torch.exp(torch.normal(-0.5*par.sigma_n0**2,par.sigma_n0,size=(N,)))

		return torch.stack((m0,p0,n0),dim=-1)


	def draw_shocks(self,N):
		""" draw shocks """

		par = self.par

		# a. taste shocks
		dist = torch.distributions.Gumbel(0, par.gumbell_param)
		taste_shocks = dist.sample((par.T,N,par.NDC))

		# b. persistent income shocks
		sigma_xi = par.sigma_xi
		xi_loc = -0.5*sigma_xi**2
		xi = torch.exp(torch.normal(xi_loc,sigma_xi,size=(par.T,N)))

		# c. transitory income shocks
		sigma_psi = par.sigma_psi
		psi_loc = -0.5*sigma_psi**2
		psi = torch.exp(torch.normal(psi_loc,sigma_psi,size=(par.T,N)))

		return torch.stack((xi,psi),dim=-1), taste_shocks

	def draw_exploration_shocks(self,epsilon_sigma,N):
		""" draw exploration shockss """

		par = self.par

		eps = torch.zeros((par.T,N,par.Nactions))
		for i_a in range(par.Nactions):
			eps[:,:,i_a] = torch.normal(0,epsilon_sigma[i_a],(par.T,N))
	
		return eps

	##############
	# quadrature #
	##############

	def quad(self):
		""" quadrature nodes and weights """

		par = self.par

		xi,psi = torch.meshgrid(par.xi,par.psi, indexing='ij')
		xi_w,psi_w = torch.meshgrid(par.xi_w,par.psi_w, indexing='ij')
		
		quad = torch.stack((xi.flatten(),psi.flatten()),dim=-1)
		quad_w = xi_w.flatten()*psi_w.flatten() 

		return quad,quad_w

	###################
	# model functions #
	###################

	outcomes = model_funcs.outcomes
	reward = model_funcs.reward
	discount_factor = model_funcs.discount_factor	
	exploration = model_funcs.exploration
	terminal_reward_pd = model_funcs.terminal_reward_pd
		
	state_trans_pd = model_funcs.state_trans_pd
	state_trans = model_funcs.state_trans

	marginal_reward = model_funcs.marginal_reward
	eval_equations_DeepVPDDC = model_funcs.eval_equations_DeepVPDDC

	def add_transfer(self,transfer):
		""" add transfer to initial states """

		par = self.par
		sim = self.sim

		sim.states[0,:,0] += transfer

	############
	# simulate #
	############

	def more_simulation_outcomes(self):
		""" compute more simulation outcomes """

		sim = self.sim
		
		sim.adj = torch.isclose(sim.DC,torch.ones_like(sim.DC))
		
		sim.c[~sim.adj] = sim.outcomes[...,0][~sim.adj]
		sim.c[sim.adj] = sim.outcomes[...,2][sim.adj]
		
		sim.d[~sim.adj] = sim.outcomes[...,1][~sim.adj]
		sim.d[sim.adj] = sim.outcomes[...,3][sim.adj]


def get_scale_from_variance(sigma):
	""" get scale from variane """
	return sigma * np.sqrt(6) / np.pi