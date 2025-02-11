import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE' # without this python may crash when plotting from matplotlib
from copy import deepcopy
from types import SimpleNamespace
import pickle
import numpy as np
import torch
torch.set_warn_always(True)

from EconModel import EconModelClass, jit
from consav.grids import nonlinspace # grids

from HumanCapitalModel import HumanCapitalModelClass

# local
from egm_human_capital import egm_retired, egm_work, simulate

class HumanCapitalModelDPClass(EconModelClass,HumanCapitalModelClass):

	def settings(self):
		""" basic settings """
		
		self.namespaces = ['par','sim','dp'] # must be numba-able
		
		# save
		self.other_attrs = ['info','train'] # other attributes to save
		self.savefolder = 'saved' # folder for saved data

		# info
		self.info = {}

		# train
		self.train = SimpleNamespace()
		self.train.dtype = torch.float32
		self.train.device = 'cpu'


	def setup(self):
		""" choose parameters """

		par = self.par
		dp = self.dp

		# a. from BufferStockModelClass
		super().setup(full=True)

		# b. egm
		dp.Na = 225 # number of grid points
		dp.Np = 150 # number of grid points
		dp.Nhk = 30 # number of grid point
		dp.Nell = 20 # number of grid points

		dp.hk_max = par.T_retired * 4# max human capital
		dp.m_max = 20.0 # max cash-on-hand
		dp.p_max = 10.0 # max permanent income
		dp.p_min = 0.1 # min permanent income
		dp.ell_min = 0.0 
		dp.ell_max = 0.99

		dp.Ntransfer = 50 # grid points for transfer interpolation
		dp.transfer_min = -0.50 # min transfer
		dp.transfer_max = 0.10 # max transfer

		# b. get number of cores
		par.cppthreads = min(dp.Np,os.cpu_count())

	def allocate(self):

		par = self.par
		sim = self.sim
		dp = self.dp

		# a. from BufferStockModelClass
		super().allocate()
		self.prepare_simulate_R()

		for k,v in par.__dict__.items():
			if isinstance(v,torch.Tensor): par.__dict__[k] = v.to(torch.float64).cpu().numpy()

		for k,v in sim.__dict__.items():
			if isinstance(v,torch.Tensor): sim.__dict__[k] = v.to(torch.float64).cpu().numpy()

		self.create_EGM_grids()

	def create_EGM_grids(self):
		""" create grids for EGM and dependent variables"""

		par = self.par
		dp = self.dp

		# a. grids for dynamic states
		dp.a_grid = nonlinspace(0.0,dp.m_max,dp.Na,1.1)
		dp.p_grid = nonlinspace(0.1,dp.p_max,dp.Np,1.1)
		dp.hk_grid = nonlinspace(0.0,dp.hk_max,dp.Nhk,1.1)
		dp.ell_grid = np.linspace(dp.ell_min,dp.ell_max,dp.Nell)


		# b. solution objects
		dp.sol_con = np.zeros((par.T, dp.Np, dp.Nhk, par.Npsi, dp.Na))
		dp.sol_ell = np.zeros((par.T, dp.Np, dp.Nhk, par.Npsi, dp.Na))
		dp.sol_q   = np.zeros((par.T, dp.Np, dp.Nhk, dp.Na))
		dp.sol_w   = np.zeros((par.T, dp.Np, dp.Nhk, dp.Na))

		# c. misc
		dp.transfer_grid = np.linspace(dp.transfer_min,dp.transfer_max,dp.Ntransfer)
		dp.R_transfer = np.zeros(dp.Ntransfer)

	def solve_DP(self):
		""" solve with DP """

		with jit(self) as model:

			par = self.par
			dp = self.dp

			for t in reversed(range(par.T)):
				print(t)
				if t < par.T_retired:
					egm_work(t,par,dp)
				else:
					egm_retired(t,par,dp)


	############
	# simulate #
	############

	def simulate_R(self,final=False):	
		""" simulate life time reward """

		par = self.par
		sim = self.sim

		# a. simulate
		with jit(self) as model:
			simulate(model.par,model.dp,model.sim)

		# b. compute R
		beta = par.beta 
		beta_t = np.zeros((par.T,sim.N))
		for t in range(par.T):
			beta_t[t] = beta**t

		c = sim.outcomes[:,:,0]
		ell = sim.outcomes[:,:,1]
		u = beta_t*(np.log(c) + par.vphi*((1-ell)**(1-par.nu))/(1-par.nu))
		sim.R = np.sum(u) / sim.N

	def compute_transfer_func(self):
		""" compute EGM utility for different transfer levels"""

		sim = self.sim
		dp = self.dp

		R_transfer = np.zeros(dp.Ntransfer)
		for i, transfer in enumerate(self.dp.transfer_grid):
			
			sim_ = deepcopy(sim) # save
			
			sim.states[0,:,0] += transfer

			self.simulate_R()
			R_transfer[i] = sim.R
		
			sim = self.sim = sim_ # reset

		dp.R_transfer = R_transfer
	
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

def select_euler_errors_EGM(model):
	""" compute mean euler error """

	par = model.par
	sim = model.sim

	c = sim.outcomes[:par.T_retired,:,0]
	states = sim.states
	actions = 1-c/states[:par.T_retired,:,0]  

	savings_indicator = (actions > par.Euler_error_min_savings)
	
	return sim.euler_error[:par.T_retired,:][savings_indicator]