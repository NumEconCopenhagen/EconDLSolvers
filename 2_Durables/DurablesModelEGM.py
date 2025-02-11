import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE' # without this python may crash when plotting from matplotlib
import pickle
from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import torch
torch.set_warn_always(True)

from EconModel import EconModelClass 
from consav.grids import nonlinspace # grids
from consav.linear_interp import interp_2d,interp_3d, interp_4d, interp_5d, interp_1d_vec_mon_noprep

# local
from DurablesModel import DurablesModelClass
from model_funcs import compute_d_and_c_from_action, marg_util_c_np, inv_marg_util_c_np
class DurablesModelEGMClass(EconModelClass,DurablesModelClass):

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
        self.cpp_filename = 'cppfuncs/main.cpp'
        self.cpp_options = {'compiler':'intel'}	

    def setup(self):
        """ choose parameters """

        par = self.par
        egm = self.egm
        sim = self.sim
        
        # a. from BufferStockModelClass
        super().setup(full=True)
        self._setup_default()

        # b. egm	
        egm.Np = 100 # number of grid points
        egm.Nm_keep = 300 # number of grid points
        egm.Nm_pd = 300 # number of grid points
        egm.Nm = 100 # number of grid points
        egm.Nn = 100 # number of grid points

        egm.m_pd_max = 5.0 # max end-of-period assets
        egm.m_max = 5.0 # max cash-on-hand
        egm.m_min = 1e-4 # min cash-on-hand

        egm.p_max = 4.0 # max permanent income
        egm.p_min = 0.1 # min permanent income
        
        egm.n_max = 4.0 # max durables

        egm.solver = 0 # 0: Nelder-Mead, 1: SLSQP, 2: MMA
        egm.pure_vfi = False # use pure VFI
        egm.min_action = 0.0 # min action
        egm.max_action = 0.9999 # max action
        
    def allocate(self):
        """ allocate arrays  """

        # unpack
        par = self.par
        sim = self.sim
        egm = self.egm

        # a. from BufferStockModelClass
        super().allocate()
        self.prepare_simulate_R()

        # convert par and sim to numpy
        for k,v in par.__dict__.items():
            if isinstance(v,torch.Tensor): par.__dict__[k] = v.to(torch.float64).cpu().numpy()

        for k,v in sim.__dict__.items():
            if isinstance(v,torch.Tensor): sim.__dict__[k] = v.to(torch.float64).cpu().numpy()

        # b. grids

        # i. grids for dynamic states
        assert egm.m_min > 0
        assert egm.p_min > 0

        egm.p_grid = np.linspace(egm.p_min,egm.p_max,egm.Np,1.1)
        egm.n_grid = nonlinspace(0.0,egm.n_max,egm.Nn,1.1)
        egm.m_pd_grid = nonlinspace(0.0,egm.m_pd_max,egm.Nm_pd,1.1)
        egm.m_keep_grid = nonlinspace(egm.m_min,egm.m_max,egm.Nm_keep,1.1)
        egm.m_grid = nonlinspace(egm.m_min,egm.m_max,egm.Nm,1.1)
                
        # ii. policy functions
        D_tuple = tuple([egm.Nn for _ in range(par.D)])
        sol_tuple = (par.T,egm.Np,) + D_tuple + (egm.Nm,)
        
        egm.sol_v = np.zeros(sol_tuple)
        egm.sol_vm = np.zeros(sol_tuple)

        egm.sol_d1_fac = np.zeros(sol_tuple)
        egm.sol_d2_fac = np.zeros(sol_tuple) if par.D > 1 else np.zeros(1)
        egm.sol_d3_fac = np.zeros(sol_tuple) if par.D > 2 else np.zeros(1)
        egm.sol_m_pd_fac = np.zeros(sol_tuple)

        egm.sol_func_evals = np.zeros(sol_tuple,dtype=np.int_)
        egm.sol_flag = np.zeros(sol_tuple,dtype=np.int_)		
        
        # iii. other solution objects
        keep_decision_tuple = (egm.Np,) + D_tuple + (egm.Nm_keep,)
        egm.sol_c_keep = np.zeros(keep_decision_tuple)

        post_decision_tuple = (egm.Np,) + D_tuple + (egm.Nm_pd,)
        egm.sol_w = np.zeros(post_decision_tuple)
        egm.sol_q = np.zeros(post_decision_tuple)
        
        # e. misc
        pos = np.array([1,5,10,25,50,100,500,1000])
        neg = np.array([1,5,10,25,50,100,500,1000,1500,2000,2500,3000,3500,4000,4500,5000])
        egm.transfer_grid = np.concatenate((-np.flip(neg),np.zeros(1),pos))/10_000
        egm.Ntransfer = egm.transfer_grid.size

        # f. get number of threads
        par.cppthreads = os.cpu_count()

        # g. simulation
        sim.R_transfer = np.zeros(egm.Ntransfer)
        sim.R_transfers = np.zeros((sim.reps,egm.Ntransfer))

    #########
    # solve #
    #########

    def solve(self):
        """ solve the model """

        par = self.par
        egm = self.egm

        self.cpp.solve_all(par,egm)

    ############
    # simulate #
    ############

    def simulate_R(self,final=False):
        """ simulate life time reward"""

        par = self.par
        egm = self.egm
        sim = self.sim

        # a. simulate
        self.cpp.simulate(par,egm,sim)

        # b. compute R
        beta = par.beta 
        beta_t = np.zeros((par.T,sim.N,))
        for t in range(par.T):
            beta_t[t] = beta**t
        
        sim.R = np.sum(beta_t*sim.reward)/sim.N

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

        return Rs

    ########
    # EGM #
    ########

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

    def interp_actions(self, states, t):
        """ interpolate actions """

        par = self.par
        egm = self.egm

        # a. prepare states
        action_shape = (states.shape[0],1+par.D)
        actions = np.zeros(action_shape)

        # b. interp actions at each element
        for i in range(states.shape[0]):
            
            # i. unpack
            m = states[i,0]
            p = states[i,1]
            n1 = states[i,2]
            if par.D > 1: n2 = states[i,3]
            if par.D > 2: n3 = states[i,4]

            # ii. interp
            if par.D == 1:
                actions[i,0] = np.clip(interp_3d(egm.p_grid, egm.n_grid, egm.m_grid, egm.sol_m_pd_fac[t,:,:,:],p,n1,m), egm.min_action, egm.max_action)
                actions[i,1] = np.clip(interp_3d(egm.p_grid, egm.n_grid, egm.m_grid, egm.sol_d1_fac[t,:,:,:],p,n1,m), egm.min_action, egm.max_action)
            elif par.D == 2:
                actions[i,0] = np.clip(interp_4d(egm.p_grid, egm.n_grid, egm.n_grid, egm.m_grid, egm.sol_m_pd_fac[t,:,:,:,:],p,n1,n2,m), egm.min_action, egm.max_action)
                actions[i,1] = np.clip(interp_4d(egm.p_grid, egm.n_grid, egm.n_grid, egm.m_grid, egm.sol_d1_fac[t,:,:,:,:],p,n1,n2,m), egm.min_action, egm.max_action)
                actions[i,2] = np.clip(interp_4d(egm.p_grid, egm.n_grid, egm.n_grid, egm.m_grid, egm.sol_d2_fac[t,:,:,:,:],p,n1,n2,m), egm.min_action, egm.max_action)
            elif par.D == 3:
                actions[i,0] = np.clip(interp_5d(egm.p_grid, egm.n_grid, egm.n_grid, egm.n_grid, egm.m_grid, egm.sol_m_pd_fac[t,:,:,:,:,:],p,n1,n2,n3,m), egm.min_action, egm.max_action)
                actions[i,1] = np.clip(interp_5d(egm.p_grid, egm.n_grid, egm.n_grid, egm.n_grid, egm.m_grid, egm.sol_d1_fac[t,:,:,:,:,:],p,n1,n2,n3,m), egm.min_action, egm.max_action)
                actions[i,2] = np.clip(interp_5d(egm.p_grid, egm.n_grid, egm.n_grid, egm.n_grid, egm.m_grid, egm.sol_d2_fac[t,:,:,:,:,:],p,n1,n2,n3,m), egm.min_action, egm.max_action)
                actions[i,3] = np.clip(interp_5d(egm.p_grid, egm.n_grid, egm.n_grid, egm.n_grid, egm.m_grid, egm.sol_d3_fac[t,:,:,:,:,:],p,n1,n2,n3,m), egm.min_action, egm.max_action)
            else:
                raise ValueError('D > 3 not supported')

        return actions

    def compute_euler_errors_DP(self):
        """ compute euler error"""

        par = self.par
        sim = self.sim
        train = self.train

        # a. get consumption and states today
        c = sim.outcomes[:par.T-1,:,0]
        d = sim.outcomes[:par.T-1,:,1:1+par.D]
        states = sim.states[:par.T-1,]
        states_pd = sim.states_pd[:par.T-1,]
        actions = sim.actions[:par.T-1]
        outcomes = sim.outcomes[:par.T-1]

        # c. compute next-period expected marginal utility
        exp_marg_util_next = np.zeros_like(c)
        for t in range(par.T-1):
            for i_xi, xi in enumerate(par.xi):
                for i_psi, psi in enumerate(par.psi):

                    # i. state transition
                    p_next = states_pd[t,:,1]**par.rho_p*xi
                    n_next = states_pd[t,:,2:2+par.D] * (1-par.delta)
                    y = p_next * par.kappa[t] * psi
                    m_next = par.R*states_pd[t,:,0] + y
                    states_next = np.concatenate((m_next[:,None],p_next[:,None],n_next),axis=1)

                    # ii. next period actions
                    actions_next = self.interp_actions(states_next,t)

                    # iii. next period marginal utility of consumption
                    c_next,d_next,a_next = compute_d_and_c_from_action(par,states_next,actions_next)
                    marg_util_next = marg_util_c_np(c_next,d_next,par,train)
                    exp_marg_util_next[t] += par.xi_w[i_xi]*par.psi_w[i_psi]*marg_util_next
        
        # d. euler error
        sim.euler_error[:par.T-1] = inv_marg_util_c_np(par.R*par.beta*exp_marg_util_next,d,par,train) / c - 1