import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE' # without this python may crash when plotting from matplotlib
import pickle
import time
from types import SimpleNamespace
import numpy as np
import torch
torch.set_warn_always(False)

from EconModel import EconModelClass, jit
from consav.grids import nonlinspace

# local
from NonConvexDurablesModel import NonConvexDurablesModelClass
from DP import solve_timestep, simulate

class NonConvexDurablesModelEGMClass2(EconModelClass,NonConvexDurablesModelClass):

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
		self.cpp_filename = 'cppfuncs/egm.cpp'
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
		egm.Nx = 150 # number of grid points

		egm.m_max = 5.0 # max cash-on-hand
		egm.a_max = egm.m_max + 1.0
		egm.n_min = 0.0 # min durable
		egm.n_max = 5.0 # max durable
		egm.p_min = 1e-4 # min persistent income
		egm.p_max = 12.0 # max persistent income
		egm.x_max = egm.m_max + egm.n_max




		self.sim.reps = 0

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
		egm.x_grid = nonlinspace(x_min,egm.x_max,egm.Nx,1.1)

		# b. solution objects

		# keeper
		keep_shape = (par.T,egm.Np,egm.Nn,egm.Nm)
		egm.sol_c_keep = np.zeros(keep_shape)
		egm.sol_v_keep = np.zeros(keep_shape)
		egm.sol_marg_u_c_keep = np.zeros(keep_shape)

		# adjuster
		adj_shape = (par.T,egm.Np,egm.Nx)
		egm.sol_c_adj = np.zeros(adj_shape)
		egm.sol_d_adj = np.zeros(adj_shape)
		egm.sol_v_adj = np.zeros(adj_shape)
		egm.sol_marg_u_c_adj = np.zeros(adj_shape)

		# post-decision
		post_shape = (par.T-1,egm.Np,egm.Nn,egm.Na)
		egm.sol_w = np.zeros(post_shape)
		egm.sol_q = np.zeros(post_shape)
		egm.sol_q_c = np.zeros(post_shape)
		egm.sol_q_m = np.zeros(post_shape)

	#########
	# solve #
	#########

	def solve_DP(self):
		""" solve with DP """

		t0 = time.perf_counter()
		with jit(self) as model:

			par = model.par
			egm = model.egm

			for t in reversed(range(par.T)):

				t0 = time.perf_counter()
				print(f'Solving period {t}',end='')
				solve_timestep(par,egm,t)
				print(f' - {time.perf_counter()-t0:.1f} secs')

		self.info['time'] = time.perf_counter()-t0
		print(f'Total time: {time.perf_counter()-t0:.1f} secs')

	############
	# simulate #
	############
	
	def simulate_R(self):
		""" simulate life time reward"""

		par = self.par
		sim = self.sim

		simulate(self,sim)
		beta = par.beta 
		beta_t = np.zeros((par.T,sim.N,2))
		for t in range(par.T):
			beta_t[t] = beta**t

		sim.R = np.sum(beta_t*sim.reward)/sim.N

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