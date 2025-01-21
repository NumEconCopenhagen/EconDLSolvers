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

from BufferStockModel import BufferStockModelClass

# local
from egm import EGM, simulate, compute_euler_errors

def select_euler_errors_EGM(model):
    """ compute mean euler error """

    par = model.par
    sim = model.sim

    c = sim.outcomes[:par.T_retired,:,0]
    states = sim.states
    actions = 1-c/states[:par.T_retired,:,0]  

    savings_indicator = (actions > par.Euler_error_min_savings)
    
    return sim.euler_error[:par.T_retired,:][savings_indicator]

class BufferStockModelEGMClass(EconModelClass,BufferStockModelClass):

    def settings(self):
        """ basic settings """
        
        self.namespaces = ['par','sim','egm'] # must be numba-able
        
        # save
        self.other_attrs = ['info','train'] # other attributes to save
        self.savefolder = 'saved' # folder for saved data

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

        par = self.par
        egm = self.egm
        sim = self.sim

        # a. from BufferStockModelClass
        super().setup(full=True)
        self._setup_default()

        # b. egm
        egm.Nm_pd = 200 # number of grid points
        egm.Nm = 400 # number of grid points
        egm.Np = 150 # number of grid points
        egm.Nsigma_psi = 15 # number of grid points
        egm.Nsigma_xi = 15 # number of grid points
        egm.Nrho_p = 15 # number of grid points

        egm.m_max = 20.0 # max cash-on-hand
        egm.p_max = 10.0 # max permanent income
        egm.p_min = 0.1 # min permanent income

        # b. get number of cores
        par.cppthreads = min(egm.Np,os.cpu_count())

    def allocate(self):
        """ allocate arrays """

        par = self.par
        sim = self.sim
        egm = self.egm

        # a. from BufferStockModelClass
        super().allocate()
        self.prepare_simulate_R()

        for k,v in par.__dict__.items():
            if isinstance(v,torch.Tensor): par.__dict__[k] = v.to(torch.float64).cpu().numpy()

        for k,v in sim.__dict__.items():
            if isinstance(v,torch.Tensor): sim.__dict__[k] = v.to(torch.float64).cpu().numpy()

        # b. grids
        if par.Nstates_fixed < 1: egm.Nsigma_xi = 1
        if par.Nstates_fixed < 2: egm.Nsigma_psi = 1
        if par.Nstates_fixed < 3: egm.Nrho_p = 1

        self.create_EGM_grids()

        # c. R transfer
        sim.R_transfer = np.zeros(egm.Ntransfer)
        sim.R_transfers = np.zeros((sim.reps,egm.Ntransfer))
                
    def create_EGM_grids(self):
        """ create grids for EGM and dependent variables """

        par = self.par
        egm = self.egm

        # a. grids for dynamic states
        egm.m_pd_grid = nonlinspace(0.0,egm.m_max,egm.Nm_pd,1.4)
        egm.m_grid = nonlinspace(0.0,egm.m_max,egm.Nm,1.4)
        egm.p_grid = nonlinspace(0.1,egm.p_max,egm.Np,1.1)

        egm.sigma_xi_grid = np.linspace(par.sigma_xi_low,par.sigma_xi_high,egm.Nsigma_xi)
        egm.sigma_psi_grid = np.linspace(par.sigma_psi_low,par.sigma_psi_high,egm.Nsigma_psi)
        egm.rho_p_grid = np.linspace(par.rho_p_low,par.rho_p_high,egm.Nrho_p)
    
        # b. solution objects
        egm.sol_con = np.zeros((par.T, egm.Np, egm.Nsigma_xi, egm.Nsigma_psi, egm.Nrho_p, egm.Nm))
        egm.sol_w = np.zeros((par.T, egm.Np, egm.Nsigma_xi, egm.Nsigma_psi, egm.Nrho_p, egm.Nm_pd))

        # c. misc
        pos = np.array([1,5,10,25,50,100,500,1000])
        neg = np.array([1,5,10,25,50,100,500,1000,1500,2000,2500,3000,3500,4000,4500,5000])
        egm.transfer_grid = np.concatenate((-np.flip(neg),np.zeros(1),pos))/10_000
        egm.Ntransfer = egm.transfer_grid.size

    def solve_EGM(self):
        """ solve with EGM """

        with jit(self) as model:

            par = model.par
            egm = model.egm

            for t in reversed(range(par.T)):
                EGM(t,par,egm)
            
    ############
    # simulate #
    ############

    def simulate_R(self,final=False):	
        """ simulate life time reward """

        par = self.par
        sim = self.sim

        # a. simulate
        with jit(self) as model:
            simulate(model.par,model.egm,model.sim,final=final)

        # b. compute R
        beta = par.beta 
        beta_t = np.zeros((par.T,sim.N))
        for t in range(par.T):
            beta_t[t] = beta**t

        c = sim.outcomes[:,:,0]
        sim.R = np.sum(beta_t*np.log(c))/sim.N

    def simulate_Rs(self):
        """ simulate life time reward for different transfer levels """

        # a. remember
        old_sim = self.sim
        Rs = self.info['Rs'] = np.zeros(self.sim.reps)

        # b. loop
        for rep in range(self.sim.reps):

            # i. set rng
            torch.set_rng_state(self.torch_rng_state[('sim',rep)])

            # ii. draw
            self.sim = deepcopy(self.sim)
            self.sim.states[0] = self.draw_initial_states(self.sim.N).numpy()
            self.sim.shocks[:,:] = self.draw_shocks(self.sim.N).numpy()

            # iii. simulate
            self.simulate_R()

            # iv. compute R_transfer
            self.compute_transfer_func()

            Rs[rep] = self.sim.R
            old_sim.R_transfers[rep] = self.sim.R_transfer

        # c. reset
        self.sim = old_sim
           
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
    
    def compute_euler_errors(self):
        """ compute euler error """

        with jit(self) as model:
            compute_euler_errors(model.par,model.egm,model.sim)

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

def mean_log10_euler_error_working_EGM(model):

    euler_errors = select_euler_errors_EGM(model)
    I = np.isclose(np.abs(euler_errors),0.0)

    return np.mean(np.log10(np.abs(euler_errors[~I])))