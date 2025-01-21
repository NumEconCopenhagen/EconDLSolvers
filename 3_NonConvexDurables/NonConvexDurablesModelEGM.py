import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE' # without this python may crash when plotting from matplotlib
import pickle
import time
from types import SimpleNamespace
import numpy as np
import torch
from copy import deepcopy
torch.set_warn_always(False)

from EconModel import EconModelClass, jit
from consav.grids import nonlinspace
from simulate import simulate
# local
from NonConvexDurablesModel import NonConvexDurablesModelClass
# from DP import solve_timestep, simulate

class NonConvexDurablesModelEGMClass(EconModelClass,NonConvexDurablesModelClass):

	#########
	# setup #
	#########

	def settings(self):
		""" basic settings """
		
		self.namespaces = ['par','sim','egm'] # must be numba-able
		self.other_attrs = ['train','info']

		# info
		self.info = {}

		# train
		self.train = SimpleNamespace()
		self.train.dtype = torch.float32
		self.train.device = 'cpu'

		# cpp
		self.cpp_filename = 'cppfuncs/main.cpp'
		self.cpp_options = {'compiler':'vs'}	

	def setup(self):
		""" choose parameters """

		egm = self.egm
		
		# a. from BufferStockModelClass
		super().setup(full=True)

		# b. egm	
		egm.Nm = 150 # number of grid points
		egm.Na = 150 # number of grid points
		egm.Np = 150 # number of grid points
		egm.Nn = 150 # number of grid points

		egm.m_max = 5.0 # max cash-on-hand
		egm.m_min = 1e-8
		egm.a_max = egm.m_max + 1.0
		egm.n_min = 0.0 # min durable
		egm.n_max = 5.0 # max durable
		egm.p_min = 1e-4 # min persistent income
		egm.p_max = 12.0 # max persistent income

		egm.solver = 2

		self.sim.reps = 0

		self.par.cppthreads = min(egm.Np,os.cpu_count())


	def allocate(self):
		""" allocate arrays  """

		# unpack
		par = self.par
		sim = self.sim

		# a. from BufferStockModelClass
		super().allocate()
		self.prepare_simulate_R()

		# convert par and sim to numpy
		for k,v in par.__dict__.items():
			if isinstance(v,torch.Tensor): par.__dict__[k] = v.to(torch.float64).cpu().numpy()

		for k,v in sim.__dict__.items():
			if isinstance(v,torch.Tensor): sim.__dict__[k] = v.to(torch.float64).cpu().numpy()

		sim.reward = np.zeros((par.T,sim.N))

		# b. egm
		self.create_EGM_grids()
	
	def create_EGM_grids(self):
		""" create grids for EGM and dependent variables"""

		par = self.par
		egm = self.egm

		# a. grids for dynamic states
		m_min = x_min = egm.p_min*par.psi.min()
		egm.m_grid = nonlinspace(m_min,egm.m_max,egm.Nm,1.1)
		egm.a_grid = nonlinspace(0.0,egm.a_max,egm.Na,1.1)
		egm.p_grid = np.linspace(egm.p_min,egm.p_max,egm.Np)
		egm.n_grid = np.linspace(egm.n_min,egm.n_max,egm.Nn)

		# b. solution objects
		shape = (par.T,egm.Np,egm.Nn,egm.Nm)

		# keeper
		egm.sol_sav_share_keep = np.zeros(shape)
		egm.sol_v_keep = np.zeros(shape)

		# adjuster
		egm.sol_exp_share_adj = np.zeros(shape)
		egm.sol_c_share_adj = np.zeros(shape)
		egm.sol_v_adj = np.zeros(shape)

		egm.sol_func_evals_keep = np.zeros(shape)
		egm.sol_flag_keep = np.zeros(shape)
		egm.sol_func_evals_adj = np.zeros(shape)
		egm.sol_flag_adj = np.zeros(shape)

		# post-decision
		post_shape = (par.T-1,egm.Np,egm.Nn,egm.Na)
		egm.sol_w = np.zeros(post_shape)


		pos = np.array([1,5,10,25,50,100,500,1000])
		neg = np.array([1,5,10,25,50,100,500,1000,1500,2000,2500,3000,3500,4000,4500,5000])
		egm.transfer_grid = np.concatenate((-np.flip(neg),np.zeros(1),pos))/10_000
		egm.Ntransfer = egm.transfer_grid.size

	#########
	# solve #
	#########



	############
	# simulate #
	############
	
	def simulate_R(self):
		""" simulate life time reward"""

		par = self.par
		sim = self.sim
		egm = self.egm

		simulate(par,egm,sim)
		beta = par.beta 
		beta_t = np.zeros((par.T,sim.N))
		for t in range(par.T):
			beta_t[t] = beta**t

		sim.R = np.sum(beta_t*sim.reward)/sim.N

	

	def compute_transfer_func(self):
		""" compute EGM utility for different transfer levels"""

		sim = self.sim
		egm = self.egm

		R_transfer = np.zeros(egm.Ntransfer)
		for i, transfer in enumerate(self.egm.transfer_grid):
			
			sim_ = deepcopy(sim) # save
			
			sim.states[0,:,0] += transfer

			self.simulate_R()
			R_transfer[i] = sim.R
		
			sim = self.sim = sim_ # reset

		sim.R_transfer = R_transfer



	########
	# save #
	########
	
	def save(self,filename):
		""" save the model """

		# a. create model dict
		model_dict = self.as_dict()

		# b. save to disc
		with open(f'{filename}', 'wb') as f:
			pickle.dump(model_dict, f)

	def solve(self):
		""" solve the model """

		par = self.par
		egm = self.egm

		self.cpp.solve_all(par,egm)