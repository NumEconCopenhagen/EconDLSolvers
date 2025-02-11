import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE' # without this python may crash when plotting from matplotlib
import pickle
from types import SimpleNamespace
import numpy as np
import torch
from copy import deepcopy
torch.set_warn_always(False)

from EconModel import EconModelClass
from consav.grids import nonlinspace
from simulate import simulate

# local
from NonConvexDurablesModel import NonConvexDurablesModelClass

class NonConvexDurablesModelVFIClass(EconModelClass,NonConvexDurablesModelClass):

	#########
	# setup #
	#########

	def settings(self):
		""" basic settings """
		
		self.namespaces = ['par','sim','vfi'] # must be numba-able
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

		vfi = self.vfi
		
		# a. from BufferStockModelClass
		super().setup(full=True)

		# b. vfi	
		vfi.Nm = 150 # number of grid points
		vfi.Na = 150 # number of grid points
		vfi.Np = 150 # number of grid points
		vfi.Nn = 150 # number of grid points

		vfi.m_max = 5.0 # max cash-on-hand
		vfi.m_min = 1e-8
		vfi.a_max = vfi.m_max + 1.0
		vfi.n_min = 0.0 # min durable
		vfi.n_max = 5.0 # max durable
		vfi.p_min = 1e-4 # min persistent income
		vfi.p_max = 12.0 # max persistent income

		vfi.solver = 2 # 0: Nelder-Mead, 1: SLSQP, 2: MMA

		# c. number of threads
		self.par.cppthreads = min(vfi.Np,os.cpu_count())

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

		# b. vfi
		self.create_VFI_grids()
	
	def create_VFI_grids(self):
		""" create grids for VFI and dependent variables"""

		par = self.par
		vfi = self.vfi

		# a. grids for dynamic states
		m_min = x_min = vfi.p_min*par.psi.min()
		vfi.m_grid = nonlinspace(m_min,vfi.m_max,vfi.Nm,1.1)
		vfi.a_grid = nonlinspace(0.0,vfi.a_max,vfi.Na,1.1)
		vfi.p_grid = np.linspace(vfi.p_min,vfi.p_max,vfi.Np)
		vfi.n_grid = np.linspace(vfi.n_min,vfi.n_max,vfi.Nn)

		# b. solution objects
		shape = (par.T,vfi.Np,vfi.Nn,vfi.Nm)

		# keeper
		vfi.sol_sav_share_keep = np.zeros(shape)
		vfi.sol_v_keep = np.zeros(shape)

		# adjuster
		vfi.sol_exp_share_adj = np.zeros(shape)
		vfi.sol_c_share_adj = np.zeros(shape)
		vfi.sol_v_adj = np.zeros(shape)

		vfi.sol_func_evals_keep = np.zeros(shape)
		vfi.sol_flag_keep = np.zeros(shape)
		vfi.sol_func_evals_adj = np.zeros(shape)
		vfi.sol_flag_adj = np.zeros(shape)

		# post-decision
		post_shape = (par.T-1,vfi.Np,vfi.Nn,vfi.Na)
		vfi.sol_w = np.zeros(post_shape)

		pos = np.array([1,5,10,25,50,100,500,1000])
		neg = np.array([1,5,10,25,50,100,500,1000,1500,2000,2500,3000,3500,4000,4500,5000])
		vfi.transfer_grid = np.concatenate((-np.flip(neg),np.zeros(1),pos))/10_000
		vfi.Ntransfer = vfi.transfer_grid.size

	############
	# simulate #
	############
	
	def simulate_R(self):
		""" simulate life time reward"""

		par = self.par
		sim = self.sim
		vfi = self.vfi

		simulate(par,vfi,sim)

		beta = par.beta 
		beta_t = np.zeros((par.T,sim.N))
		for t in range(par.T):
			beta_t[t] = beta**t

		sim.R = np.sum(beta_t*sim.reward)/sim.N

	def compute_transfer_func(self):
		""" compute VFI utility for different transfer levels"""

		sim = self.sim
		vfi = self.vfi

		R_transfer = np.zeros(vfi.Ntransfer)
		for i, transfer in enumerate(self.vfi.transfer_grid):
			
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
		vfi = self.vfi

		self.cpp.solve_all(par,vfi)