import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE' # without this python may crash when plotting from matplotlib
import pickle
from types import SimpleNamespace
import numpy as np
import numba as nb 
import torch
from copy import deepcopy
torch.set_warn_always(True)

# EconModel and consav
from EconModel import EconModelClass, jit
from consav.grids import nonlinspace
from consav.linear_interp import interp_2d, interp_3d, interp_4d
from simulation import simulate_1d, simulate_2d

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
        self.savefolder = '../output/'
        
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
        par = self.par
        sim = self.sim

        sim.R = 0.0
        
        # a. from BufferStockModelClass
        super().setup(full=True)

        # b. vfi	
        if par.full:
            vfi.Nm = 150 # number of grid points
            vfi.Na = 200 # number of grid points
            vfi.Np = 100 # number of grid points
            vfi.Nn = 150 # number of grid points
            vfi.Nx = 150 # number of grid points
        else:
            vfi.Nm = 10
            vfi.Na = 20 
            vfi.Np = 10 
            vfi.Nn = 10 
            vfi.Nx = 10

        vfi.m_min = 1e-8
        vfi.m_max = 5.0 
        vfi.a_max = vfi.m_max + 1.0

        vfi.n_min = 0.0 
        vfi.n_max = 7.0 
        
        vfi.x_min = 1e-8
        vfi.x_max = 5.0 + (1-par.nu) * vfi.n_max * par.D
        
        vfi.p_min = 1e-4
        vfi.p_max = 5.0

        vfi.solver = 2 # 0: Nelder-Mead, 1: SLSQP, 2: MMA

        # c. number of threads
        self.par.cppthreads = min(vfi.Np,os.cpu_count())

    def allocate(self):
        """ allocate arrays  """

        # unpack
        par = self.par
        sim = self.sim
        vfi = self.vfi
        dtype = self.train.dtype
        device = self.train.device

        # a. from BufferStockModelClass
        super().allocate()
        self.prepare_simulate_R()

        # convert par and sim to numpy
        for k,v in par.__dict__.items():
            if isinstance(v,torch.Tensor): par.__dict__[k] = v.to(torch.float64).cpu().numpy()

        for k,v in sim.__dict__.items():
            if isinstance(v,torch.Tensor): sim.__dict__[k] = v.to(torch.float64).cpu().numpy()
        
        del par.choices_from_iDC, par.iDC_from_choices, par.actions_from_iDC, par.actions_from_i_actions

        # b. vfi
        self.create_VFI_grids()
    
    def create_VFI_grids(self):
        """ create grids for VFI and dependent variables"""

        par = self.par
        vfi = self.vfi
        sim = self.sim

        # a. grids for dynamic states
        m_min = x_min = vfi.p_min*par.psi.min()
        vfi.m_grid = nonlinspace(m_min,vfi.m_max,vfi.Nm,1.1)
        vfi.a_grid = nonlinspace(0.0,vfi.a_max,vfi.Na,1.1)
        vfi.p_grid = np.linspace(vfi.p_min,vfi.p_max,vfi.Np)
        vfi.n_grid = np.linspace(vfi.n_min,vfi.n_max,vfi.Nn)
        vfi.x_grid = nonlinspace(x_min,vfi.x_max,vfi.Nx,1.1)

        # b. solution objects
        
        # D = 1
        if par.D == 1:

            shape_keep = (par.T,vfi.Np,vfi.Nn,vfi.Nm)

            # keeper
            vfi.sol_sav_share_keep = np.zeros(shape_keep)
            vfi.sol_v_keep = np.zeros(shape_keep)
            vfi.sol_func_evals_keep = np.zeros(shape_keep)
            vfi.sol_flag_keep = np.zeros(shape_keep)

            # adjuster
            shape_adjust = (par.T,vfi.Np,vfi.Nx)
            vfi.sol_exp_share_adj = np.zeros(shape_adjust)
            vfi.sol_c_share_adj = np.zeros(shape_adjust)
            vfi.sol_v_adj = np.zeros(shape_adjust)
            vfi.sol_func_evals_adj = np.zeros(shape_adjust)
            vfi.sol_flag_adj = np.zeros(shape_adjust)

            # adjust durable 1
            vfi.sol_exp_share_adj1 = np.array([0.0])
            vfi.sol_c_share_adj1 = np.array([0.0])
            vfi.sol_v_adj1 = np.array([0.0])
            vfi.sol_func_evals_adj1 = np.array([0.0])
            vfi.sol_flag_adj1 = np.array([0.0])

            # adjust durable 2
            vfi.sol_exp_share_adj2 = np.array([0.0])
            vfi.sol_c_share_adj2 = np.array([0.0])
            vfi.sol_v_adj2 = np.array([0.0])
            vfi.sol_func_evals_adj2 = np.array([0.0])
            vfi.sol_flag_adj2 = np.array([0.0])
            
            # adjust both durables
            vfi.sol_exp_share_adj12 = np.array([0.0])
            vfi.sol_c_share_adj12 = np.array([0.0])
            vfi.sol_d1_share_adj12 = np.array([0.0])
            vfi.sol_v_adj12 = np.array([0.0])
            vfi.sol_func_evals_adj12 = np.array([0.0])
            vfi.sol_flag_adj12 = np.array([0.0])

            # post-decision
            post_shape = (par.T-1,vfi.Np,vfi.Nn,vfi.Na)
            vfi.sol_w = np.zeros(post_shape)

        # D=2
        else:

            shape_keep = (par.T,vfi.Np,vfi.Nn,vfi.Nn,vfi.Nm)

            # keeper
            vfi.sol_sav_share_keep = np.zeros(shape_keep)
            vfi.sol_v_keep = np.zeros(shape_keep)
            vfi.sol_func_evals_keep = np.zeros(shape_keep)
            vfi.sol_flag_keep = np.zeros(shape_keep)

            # adjuster
            vfi.sol_exp_share_adj = np.array([0.0])
            vfi.sol_c_share_adj = np.array([0.0])
            vfi.sol_v_adj = np.array([0.0])
            vfi.sol_func_evals_adj = np.array([0.0])
            vfi.sol_flag_adj = np.array([0.0])

            shape_adjust_single = (par.T,vfi.Np,vfi.Nn,vfi.Nx)

            # adjust durable 1
            vfi.sol_exp_share_adj1 = np.zeros(shape_adjust_single)
            vfi.sol_c_share_adj1 = np.zeros(shape_adjust_single)
            vfi.sol_v_adj1 = np.zeros(shape_adjust_single)
            vfi.sol_func_evals_adj1 = np.zeros(shape_adjust_single)
            vfi.sol_flag_adj1 = np.zeros(shape_adjust_single)

            # adjust durable 2
            vfi.sol_exp_share_adj2 = np.zeros(shape_adjust_single)
            vfi.sol_c_share_adj2 = np.zeros(shape_adjust_single)
            vfi.sol_v_adj2 = np.zeros(shape_adjust_single)
            vfi.sol_func_evals_adj2 = np.zeros(shape_adjust_single)
            vfi.sol_flag_adj2 = np.zeros(shape_adjust_single)

            # adjust both durables
            shape_adjust_both = (par.T,vfi.Np,vfi.Nx)
            vfi.sol_exp_share_adj12 = np.zeros(shape_adjust_both)
            vfi.sol_c_share_adj12 = np.zeros(shape_adjust_both)
            vfi.sol_d1_share_adj12 = np.zeros(shape_adjust_both)
            vfi.sol_v_adj12 = np.zeros(shape_adjust_both)
            vfi.sol_func_evals_adj12 = np.zeros(shape_adjust_both)
            vfi.sol_flag_adj12 = np.zeros(shape_adjust_both)

            # post-decision
            post_shape = (par.T-1,vfi.Np,vfi.Nn,vfi.Nn,vfi.Na)
            vfi.sol_w = np.zeros(post_shape)
        vfi.transfer_grid = sim.transfer_grid 


    ############
    # simulate #
    ############
    
    def simulate_R(self):
        """ simulate life time reward"""

        par = self.par
        sim = self.sim
        vfi = self.vfi

        self.cpp.simulate_cpp(par,vfi,sim)

            
        beta = par.beta 
        beta_t = np.zeros((par.T,sim.N))
        for t in range(par.T):
            beta_t[t] = beta**t

        sim.R = np.sum(beta_t[...,None]*sim.reward)/sim.N

    def compute_euler_error(self):

        with jit(self) as model:

            par = model.par 
            sim = model.sim 
            vfi = model.vfi 

            if par.D == 1: 
                self.cpp.compute_euler_errors_cpp(par,sim,vfi)
            elif par.D == 2:
                self.cpp.compute_euler_errors_cpp(par,sim,vfi)
            else:
                raise('Ooly 1D and 2D implemented')

    def compute_transfer_func(self):
        """ compute VFI utility for different transfer levels"""

        sim = self.sim
        par = self.par

        R_transfer = np.zeros(sim.Ntransfer)
        individual_R_transfer = np.zeros((sim.Ntransfer,sim.N))
        
        discount = par.beta ** np.arange(par.T,dtype=float)
        for i,transfer in enumerate(sim.transfer_grid):
            
            sim_ = deepcopy(sim) # save
            
            sim.states[0,:,0] += transfer

            self.simulate_R()
            R_transfer[i] = sim.R

            rewards = self.sim.reward
            individual_R_transfer[i] = np.sum(rewards*discount[:,None,None],axis=(0,2))
        
            sim = self.sim = sim_ # reset

        sim.R_transfer[:] = R_transfer
        sim.individual_R_transfer[:] = individual_R_transfer

    ########
    # save #
    ########
    
    def save(self,filename):
        """ save the model """

        # a. create model dict
        model_dict = self.as_dict()

        # b. save to disc
        with open(f'{filename}','wb') as f:
            pickle.dump(model_dict,f)

    def solve(self):
        """ solve the model """

        par = self.par
        vfi = self.vfi

        self.cpp.solve_all(par,vfi)        