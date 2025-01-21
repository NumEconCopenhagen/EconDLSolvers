import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE' # without this python may crash when plotting from matplotlib
import numpy as np
import torch
torch.set_warn_always(True)
from copy import deepcopy
from consav.quadrature import log_normal_gauss_hermite

from EconDLSolvers import DLSolverClass

# local
import model_funcs

# auxilliary functions
def torch_uniform(a,b,size):
    """ uniform random numbers in [a,b] """

    return a + (b-a)*torch.rand(size)

def get_omega_delta_d_ubar(D):
    """ Get omega, delta, and d_ubar for a given number of durables """

    omega = torch.tensor([0.2/D for i in range(D)])

    if D == 1:
        delta = torch.tensor([0.2])
    elif D == 2:
        delta = torch.tensor(sorted([0.15, 0.2]))
    elif D == 3:
        delta = torch.tensor(sorted([0.15, 0.2, 0.1]))
    elif D == 4:
        delta = torch.tensor(sorted([0.15, 0.2, 0.1, 0.25]))
    elif D == 5:
        delta = torch.tensor(sorted([0.15, 0.2, 0.1, 0.25, 0.3]))
    elif D == 6:
        delta = torch.tensor(sorted([0.15, 0.2, 0.1, 0.25, 0.3, 0.27]))
    elif D == 7:
        delta = torch.tensor(sorted([0.15, 0.2, 0.1, 0.25, 0.3, 0.27, 0.22]))
    elif D == 8:
        delta = torch.tensor(sorted([0.15, 0.2, 0.1, 0.25, 0.3, 0.27, 0.22, 0.19]))
    elif D == 9:
        delta = torch.tensor(sorted([0.15, 0.2, 0.1, 0.25, 0.3, 0.27, 0.22, 0.19, 0.17]))
    elif D == 10:
        delta = torch.tensor(sorted([0.15, 0.2, 0.1, 0.25, 0.3, 0.27, 0.22, 0.19, 0.17, 0.16]))
    
    d_ubar = torch.tensor([0.01 for i in range(D)])

    return omega, delta, d_ubar

# class
class DurablesModelClass(DLSolverClass):

    #####################
    # setup par and sim #
    #####################

    def setup(self,full=None):
        """ choose parameters """

        par = self.par
        sim = self.sim

        par.full = full if not full is None else torch.cuda.is_available()
        par.seed = 1 # seed for random number generator

        # a. model
        par.T = 30 # number of periods
        par.T_retired = 25 # number of periods retired

        # preferences
        par.beta = 1.0/1.01 # discount factor
        par.rho = 2.0 # CRRA

        # income process  
        par.kappa_base = 1.0 # base income
        par.kappa_retired = 0.7 # replacement rate
        par.kappa_growth = 0.02 # income growth
        par.kappa_growth_decay = 0.1 # income growth decay

        par.rho_p = 0.95  # shock, persistence
        par.sigma_xi = 0.1 # shock, std
        par.sigma_psi = 0.1 # shock, std

        par.Nxi = 4 # shock, number of qudrature nodes, permanent
        par.Npsi = 4 # shock, number of qudrature nodes, transitory

        # saving and durables
        par.R = 1.01 # gross return
        par.D = 1 # number of durables
        par.tau = 0.1 # adjustment cost
        par.nonnegative_investment = True # nonnegative investment

        # b. misc
        par.KKT = False # use KKT conditions (multipliers must be included in policy_NN)

        par.m_scaler = 1 / 10.0
        par.p_scaler = 1 / 5.0
        par.n_scaler = 1 / 1.0

        # c. simulation

        # initial states
        par.mu_m0 = 1.0 # initial cash-on-hand, mean
        par.sigma_m0 = 0.1 # initial cash-on-hand, std
        par.mu_p0 = 1.0 # initial permanent income, mean
        par.sigma_p0 = 0.1 # initial permanent income, std
        par.mu_n0 = 0.1 # initial durable, mean
        par.sigma_d0 = 0.1 # initial durable, std

        # life-time reward
        sim.N = 100_000 # number of agents

        # misc
        par.Delta_MPC = 1e-4 # windfall used in MPC calculation

    def allocate(self):
        """ allocate arrays  """

        # unpack
        par = self.par
        sim = self.sim
        train = self.train
        dtype = train.dtype
        device = train.device

        if not par.full: # for solving without GPU
            par.T = 5
            par.T_retired = 3
            sim.N = 1_000 

        # a. life-cycle income
        par.kappa = torch.zeros(par.T,dtype=dtype,device=device)	
        par.kappa[0] = par.kappa_base
    
        for t in range(1,par.T_retired):
            par.kappa[t] = par.kappa[t-1]*(1+par.kappa_growth*(1-par.kappa_growth_decay)**(t-1))

        par.kappa[par.T_retired:] = par.kappa_retired * par.kappa_base

        # b. durables
        par.omega,par.delta,par.d_ubar = get_omega_delta_d_ubar(par.D)
        
        par.omega = par.omega.to(dtype).to(device)
        par.delta = par.delta.to(dtype).to(device)
        par.d_ubar = par.d_ubar.to(dtype).to(device)

        # c. states, actions and shocks
        
        # states
        par.Nstates_fixed = 0 # number of fixed states
        par.Nstates_fixed_pd = 0 # number of fixed post-decision states		

        par.Nstates_dynamic = 2+par.D # number of dynamic states
        par.Nstates_dynamic_pd = 2+par.D # number of dynamic post-decision states

        par.Nstates = par.Nstates_dynamic + par.Nstates_fixed # number of states
        par.Nstates_pd = par.Nstates_dynamic_pd + par.Nstates_fixed_pd # number of post-decision states

        # number of actions and outcomes
        if par.KKT:
            par.Nactions = 1 + 1 * par.D + 1 + 1 * par.D
        else:
            par.Nactions = 1 + 1 * par.D
        
        par.Noutcomes = 2 + par.D

        # number of shocks
        par.Nshocks = 2

        # d. quadrature
        par.psi,par.psi_w = log_normal_gauss_hermite(par.sigma_psi,par.Npsi)
        par.xi,par.xi_w = log_normal_gauss_hermite(par.sigma_xi,par.Nxi)

        par.psi_w = torch.tensor(par.psi_w,dtype=dtype,device=device)
        par.psi = torch.tensor(par.psi,dtype=dtype,device=device)
        par.xi_w = torch.tensor(par.xi_w,dtype=dtype,device=device)
        par.xi = torch.tensor(par.xi,dtype=dtype,device=device)		

        # e. simulation
        sim.states = torch.zeros((par.T,sim.N,par.Nstates),dtype=dtype,device=device)
        sim.states_pd = torch.zeros((par.T,sim.N,par.Nstates_pd),dtype=dtype,device=device)
        sim.shocks = torch.zeros((par.T,sim.N,par.Nshocks),dtype=dtype,device=device)
        sim.outcomes = torch.zeros((par.T,sim.N,par.Noutcomes),dtype=dtype,device=device)
        sim.actions = torch.zeros((par.T,sim.N,par.Nactions),dtype=dtype,device=device)
        sim.reward = torch.zeros((par.T,sim.N),dtype=dtype,device=device)	

        sim.R = np.nan
        sim.euler_error = torch.zeros((par.T,sim.N),dtype=dtype,device=device)
        sim.MPC_c = torch.zeros((par.T,sim.N),dtype=dtype,device=device)
        sim.MPC_d = torch.zeros((par.T,sim.N,par.D),dtype=dtype,device=device)

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
            train.Nneurons_policy = np.array([50,50])
            train.Nneurons_value = np.array([50,50])
        
        # b. policy activation functions and clipping
        train.policy_activation_final = ['sigmoid']
        
        train.min_actions = torch.tensor([0.0 for _ in range(par.Nactions)],dtype=dtype,device=device)

        if not par.KKT:
        
            train.max_actions = torch.tensor([0.9999 for _ in range(par.Nactions)],dtype=dtype,device=device) # maximum action value		
        
        else:

            max_action_val_sr = torch.tensor([0.9999 for i in range(par.D + 1)],dtype=dtype,device=device)
            max_action_val_mult = torch.tensor([np.inf for i in range(par.D + 1)],dtype=dtype,device=device)
            train.max_actions = torch.cat((max_action_val_sr,max_action_val_mult))

            final_actvation_list_sr = ['sigmoid' for i in range(par.D+1)]
            policy_activation_final_list_mult = ['softplus' for i in range(par.D+1)]
            train.policy_activation_final = final_actvation_list_sr + policy_activation_final_list_mult	
            
        # c. misc
        train.terminal_actions_known = False

        # b. algorithm specific

        # DeepSimulate 
        if self.train.algoname == 'DeepSimulate':
            pass
        else:
            train.epsilon_sigma = 0.1*np.ones(par.Nactions)
            if par.D == 3:
                train.epsilon_sigma = 0.05*np.ones(par.Nactions)
            if par.D == 8:
                train.epsilon_sigma = 0.03*np.ones(par.Nactions)

            train.epsilon_sigma_min = np.zeros(par.Nactions)

        # DeepFOC
        if self.train.algoname == 'DeepFOC':
            train.Nneurons_policy = np.array([700,700])
            if par.D == 8:
                train.Nneurons_policy = np.array([1000,1000])

            train.eq_w = torch.tensor([5.0] + [3.0 for i_d in range(par.D)] + [1.0 for i in range(1+par.D)],dtype=dtype,device=device) # weight 5 on consumption euler, weight 3 on all durable eulers, weight 1 on all slackness
            
            
        if self.train.algoname == 'DeepVPD':
            
            if par.D < 4:
                train.N_value_NN = 3
                train.learning_rate_value_decay = 0.999
                train.learning_rate_policy_decay = 0.999
                train.tau = 1.0

            if par.D == 8:
                train.N_value_NN = 4
                train.epsilon_sigma = 0.06*np.ones(par.Nactions)
                train.tau = 0.8
                train.Nneurons_value = np.array([1000,1000])
                
                if train.use_FOC:
                    train.Nneurons_policy = np.array([1000,1000])
                    train.NFOC_targets = 1+par.D # number of targets in FOCs
                    assert self.par.KKT == True, 'KKT must be True for FOC in Durables model'
            
    
    def allocate_train(self):
        """ allocate memory training """

        par = self.par
        train = self.train
        dtype = train.dtype
        device = train.device

        # a. dependent settings
        scale_vec = [par.m_scaler,par.p_scaler] + [par.n_scaler for _ in range(par.D)]
        train.scale_vec_states = torch.tensor(scale_vec,dtype=train.dtype,device=train.device)
        train.scale_vec_states_pd = torch.tensor(scale_vec,dtype=train.dtype,device=train.device)

        if train.algoname == 'DeepFOC':
            train.eq_w = train.eq_w / torch.sum(train.eq_w)

        # b. training samples
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
        train = self.train

        sigma_m0 = par.sigma_m0
        sigma_p0 = par.sigma_p0
        sigma_d0 = par.sigma_d0

        # a. draw cash-on-hand		
        m0 = par.mu_m0*np.exp(torch.normal(-0.5*sigma_m0**2,sigma_m0,size=(N,)))
        
        # b. draw permanent income
        p0 = par.mu_p0*np.exp(torch.normal(-0.5*sigma_p0**2,sigma_p0,size=(N,)))

        # c. draw initial durables
        n0s = (par.mu_n0*np.exp(torch.normal(-0.5*sigma_d0**2,sigma_d0,size=(N,))) for _ in range(par.D))

        # d. store
        return torch.stack((m0,p0,*n0s),dim=1)

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
        """ draw exploration shocks - need to be changed """

        par = self.par

        eps = torch.zeros((par.T,N,par.Nactions))

        for i_action in range(par.Nactions):
            epsilon_sigma_i = epsilon_sigma[i_action]
            eps[:,:,i_action] = torch.normal(0,epsilon_sigma_i,(par.T,N))
    
        return eps

    def draw_exo_actions(self,N):
        """ draw exogenous actions - need to be changed """

        par = self.par

        exo_actions = np.zeros((par.T,N,par.Nactions))

        for i_action in range(par.Nactions):
            exo_actions[:,:,i_action] = torch_uniform(0.1,0.5,(par.T,N))
    
        return exo_actions

    ###################
    # model functions #
    ###################

    outcomes = model_funcs.outcomes
    reward = model_funcs.reward
    discount_factor = model_funcs.discount_factor
    
    terminal_actions = model_funcs.terminal_actions
    terminal_reward_pd = model_funcs.terminal_reward_pd
        
    state_trans_pd = model_funcs.state_trans_pd
    state_trans = model_funcs.state_trans
    exploration = model_funcs.exploration

    marginal_reward = model_funcs.marginal_reward
    marginal_terminal_reward = model_funcs.marginal_terminal_reward
    eval_equations_FOC = model_funcs.eval_equations_FOC
    eval_equations_VPD = model_funcs.eval_equations_VPD

    def add_transfer(self,transfer):
        """ add transfer to initial states """

        par = self.par
        sim = self.sim

        sim.states[0,:,0] += transfer
        
    ############ 
    # simulate #
    ############
     
    def compute_euler_errors(self,Nbatch_share=0.01):
        """ compute euler error"""

        par = self.par
        sim = self.sim
        train = self.train

        Nbatch = int(Nbatch_share*sim.N)

        for i in range(0,sim.N,Nbatch):

            index_start = i
            index_end = i + Nbatch

            with torch.no_grad():
                
                # a. get consumption and states today
                c = sim.outcomes[:par.T-1,index_start:index_end,0]
                d = sim.outcomes[:par.T-1,index_start:index_end,1:1+par.D]
                states = sim.states[:par.T-1,index_start:index_end]
                actions = sim.actions[:par.T-1,index_start:index_end]
                outcomes = sim.outcomes[:par.T-1,index_start:index_end]

                # b. post-decision states
                states_pd = self.state_trans_pd(states,actions,outcomes)

                # c. next-period states
                states_next = self.state_trans(states_pd,train.quad)

                # d. next-period action
                actions_next = self.eval_policy(self.policy_NN,states_next,t0=1)

                # e. next-period consumption
                outcomes_next = self.outcomes(states_next,actions_next,t0=1)
                c_next = outcomes_next[...,0]
                d_next = outcomes_next[...,1:1+par.D]

                # f. marginal utility next period
                marg_util_next = model_funcs.marg_util_c(c_next,d_next,par,train)

                # g. expected marginal utility next period
                exp_marg_util_next = torch.sum(train.quad_w[None,None,:]*marg_util_next, dim=-1)

                # h. euler error
                euler_error_Nbatch = model_funcs.inv_marg_util_c(par.R*par.beta*exp_marg_util_next,d,par,train) / c - 1
                euler_error_Nbatch = torch.abs(euler_error_Nbatch)
                sim.euler_error[:par.T-1,index_start:index_end] = euler_error_Nbatch

    def compute_MPC(self,Nbatch_share=0.01):
        """ compute MPC """

        par = self.par
        sim = self.sim
        
        with torch.no_grad():

            # a. baseline
            c = sim.outcomes[...,0]
            d = sim.outcomes[...,1:1+par.D]

            # b. add windfall
            states = deepcopy(sim.states)
            states[:,:,0] += par.Delta_MPC

            # b. alternative
            c_alt = torch.ones_like(c)
            d_alt = torch.ones_like(d)
            Nbatch = int(Nbatch_share*sim.N)
            for i in range(0,sim.N,Nbatch):

                index_start = i
                index_end = i + Nbatch
                        
                states_ = states[:par.T-1,index_start:index_end,:]
                actions = self.eval_policy(self.policy_NN,states_)
                outcomes = self.outcomes(states_,actions)
                c_alt[:par.T-1,index_start:index_end] = outcomes[:,:,0]
                d_alt[:par.T-1,index_start:index_end] = outcomes[:,:,1:1+par.D]

            # c. MPC
            sim.MPC_c[:,:] = (c_alt-c)/par.Delta_MPC
            sim.MPC_d[:,:,:] = (d_alt-d)/par.Delta_MPC

    #######
    # DDP #
    #######

    def transfer_to_device(self,device_send='cpu'):
        """ transfer tensors and networks to device"""
        
        par = self.par
        train = self.train

        # a. par
        for k,v in par.__dict__.items():
            if isinstance(v,torch.tensor): par.__dict__[k] = v.to(device_send)

        # b. train
        train.min_actions = train.min_actions.to(device_send)
        train.max_actions = train.max_actions.to(device_send)
        train.quad = train.quad.to(device_send)
        train.quad_w = train.quad_w.to(device_send)
        train.scale_vec = train.scale_vec.to(device_send)