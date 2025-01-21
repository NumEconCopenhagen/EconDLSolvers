import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE' # without this python may crash when plotting from matplotlib
from copy import deepcopy
import numpy as np
import torch
torch.set_warn_always(True)

from consav.quadrature import log_normal_gauss_hermite, gauss_hermite

# local
import model_funcs
from EconDLSolvers import DLSolverClass, torch_uniform, compute_transfer

# class
class BufferStockModelClass(DLSolverClass):
    
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
        par.T = 55 # number of periods
        par.T_retired = 45 # number of periods retired

        # preferences
        par.beta = 1/1.01 # discount factor

        # income
        par.kappa_base = 1.0 # base  # Martin - Life cycle is kappa in paper
        par.kappa_growth = 0.02 # income growth
        par.kappa_growth_decay = 0.1 # income growth decay
        par.kappa_retired = 0.7 # replacement rate

        par.rho_p_base = 0.95 # shock, persistence
        par.rho_p_low = 0.70   
        par.rho_p_high = 0.99

        par.sigma_xi_base = 0.1 # shock, permanent , std
        par.sigma_xi_low = 0.05
        par.sigma_xi_high = 0.15
        par.Nxi = 4 # number of qudrature nodes

        par.sigma_psi_base = 0.1 # shock, transitory std
        par.sigma_psi_low = 0.05
        par.sigma_psi_high = 0.15
        par.Npsi = 4 # number of qudrature nodes

        # return
        par.R = 1.01 # gross return

        # b. solver settings

        # states and shocks
        par.Nstates_fixed = 0 # number of fixed states
        par.Nstates_dynamic = 2 # number of dynamic states
        par.Nstates_dynamic_pd = 2 # number of dynamic post-decision states
        par.Nshocks = 2 # number of shocks

        # outcomes and actions
        par.Noutcomes = 1 # number of outcomes (here just c)
        par.KKT = False # use KKT conditions (for DeepFOC)
        par.NDC = 0 # number of discrete choices

        # scaling
        par.m_scaler = 1/10.0
        par.p_scaler = 1/5.0

        # policy prediction
        par.policy_predict = 'savings_rate' # 'savings_rate' or 'consumption'

        # Euler
        par.Euler_error_min_savings = 1e-3 # minimum savings rate for computing Euler error
        par.Delta_MPC = 1e-4 # windfall used in MPC calculation

        # c. simulation 
        sim.N = 100_000 # number of agents

        # initial states
        par.mu_m0 = 1.0 # initial cash-on-hand, mean
        par.sigma_m0 = 0.1 # initial cash-on-hand, std

        # initial permanent income
        par.mu_p0 = 1.0 # initial durable, mean
        par.sigma_p0 = 0.1 # initial durable, std

        # exploration of initial states
        par.explore_states_endo_fac = 1.0
        par.explore_states_fixed_fac_low = 1.0
        par.explore_states_fixed_fac_high = 1.0


    def allocate(self):
        """ allocate arrays  """

        # a. unpack
        par = self.par
        sim = self.sim
        train = self.train

        dtype = train.dtype
        device = train.device

        if not par.full: # for solving without GPU
            par.T = 5
            par.T_retired = 3
            sim.N = 1_000
        
        # a. life cycle income
        par.kappa = torch.zeros(par.T,dtype=dtype,device=device)	# Martin - kappa
        par.kappa[0] = par.kappa_base
    
        for t in range(1,par.T_retired):
            par.kappa[t] = par.kappa[t-1]*(1+par.kappa_growth*(1-par.kappa_growth_decay)**(t-1))

        par.kappa[par.T_retired:] = par.kappa_retired * par.kappa_base

        # b. states, shocks and actions
        par.Nstates_fixed_pd = par.Nstates_fixed # number of fixed post-decision states
        par.Nstates = par.Nstates_dynamic + par.Nstates_fixed # number of states
        par.Nstates_pd = par.Nstates_dynamic_pd + par.Nstates_fixed_pd # number of post-decision states

        if par.KKT: 
            par.Nactions = 2
        else:
            par.Nactions = 1

        # c. quadrature
        _, par.psi_w = log_normal_gauss_hermite(1.0,par.Npsi)
        par.psi, _ = gauss_hermite(par.Npsi)
        _, par.xi_w = log_normal_gauss_hermite(1.0,par.Nxi)
        par.xi, _ = gauss_hermite(par.Nxi)

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
        if par.policy_predict == 'savings_rate':

            if par.KKT:
                train.policy_activation_final = ['sigmoid','softplus'] # policy activation functions
                train.min_actions = torch.tensor([0.0,0.0],dtype=dtype,device=device) # minimum action value
                train.max_actions = torch.tensor([0.9999,np.inf],dtype=dtype,device=device) # maximum action value
            else:
                train.policy_activation_final = ['sigmoid']
                train.min_actions = torch.tensor([0.0],dtype=dtype,device=device)
                train.max_actions = torch.tensor([0.9999],dtype=dtype,device=device)
        
        elif par.policy_predict == 'consumption':

            if par.KKT:
                train.policy_activation_final = ['softplus','softplus']
                train.min_actions = torch.tensor([0.0,0.0],dtype=dtype,device=device)
                train.max_actions = torch.tensor([1000.0,np.inf],dtype=dtype,device=device)
            else:
                train.policy_activation_final = ['softplus']
                train.min_actions = torch.tensor([0.0],dtype=dtype,device=device)
                train.max_actions = torch.tensor([1000.0],dtype=dtype,device=device)

        else:

            raise ValueError('policy_predict must be either savings_rate or consumption')
        
        # c. misc
        train.terminal_actions_known = True # use terminal actions
        train.use_input_scaling = False # use input scaling

        # d. algorithm specific
    
        # DeepSimulate
        if self.train.algoname == 'DeepSimulate':

            pass

        else:

            if par.policy_predict == 'savings_rate':

                if par.KKT:
                    train.epsilon_sigma = np.array([0.1,0.0]) # std of exploration shocks
                    train.epsilon_sigma_min = np.array([0.0,0.0])
                else:
                    train.epsilon_sigma = np.array([0.1])
                    train.epsilon_sigma_min = np.array([0.0])

            else:

                if par.KKT:
                    train.epsilon_sigma = np.array([0.5,0.0])
                    train.epsilon_sigma_min = np.array([0.0,0.0])
                else:
                    train.epsilon_sigma = np.array([0.5])
                    train.epsilon_sigma_min = np.array([0.0])
                        
        # DeepFOC
        if self.train.algoname == 'DeepFOC':
            if par.KKT:
                train.eq_w = torch.tensor([5.0,1.0],dtype=dtype,device=device) # 5 weight on FOC and 1 on budget constraint
            else:
                train.eq_w = torch.tensor([1.0],dtype=dtype,device=device)

            pass
            
        # DeepVPD
        if train.algoname == 'DeepVPD':

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
        if train.algoname == 'DeepFOC':
            train.eq_w = train.eq_w / torch.sum(train.eq_w) # normalize

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
        
        quad = torch.stack((xi.flatten(),psi.flatten()),dim=-1)
        quad_w = xi_w.flatten()*psi_w.flatten() 

        return 2.0**(1/2)*quad,quad_w
        
    #########
    # draw #
    #########

    def draw_initial_states(self,N,training=False):
        """ draw initial state (m,p,t) """

        par = self.par

        fac = 1.0 if not training else par.explore_states_endo_fac
        fac_low = 1.0 if not training else par.explore_states_fixed_fac_low
        fac_high = 1.0 if not training else par.explore_states_fixed_fac_high

        # a. draw cash-on-hand
        sigma_m0 = par.sigma_m0*fac
        m0 = par.mu_m0*np.exp(torch.normal(-0.5*sigma_m0**2,sigma_m0,size=(N,)))
        
        # b. draw permanent income
        sigma_p0 = par.sigma_p0*fac
        p0 = par.mu_p0*np.exp(torch.normal(-0.5*sigma_p0**2,sigma_p0,size=(N,)))

        # c. draw sigma_xi
        if par.Nstates_fixed > 0: 
            sigma_xi = torch_uniform(par.sigma_xi_low*fac_low,par.sigma_xi_high*fac_high,size=(N,))

        # d. draw sigma_psi
        if par.Nstates_fixed > 1: sigma_psi = torch_uniform(par.sigma_psi_low*fac_low,par.sigma_psi_high*fac_high,size=(N,))

        # e. draw rho_p  
        if par.Nstates_fixed > 2: rho_p = torch_uniform(par.rho_p_low*fac_low,np.fmin(par.rho_p_high*fac_high,1.0),size=(N,))
        
        # f. store
        if par.Nstates_fixed == 0:
            return torch.stack((m0,p0),dim=-1)
        elif par.Nstates_fixed == 1:
            return torch.stack((m0,p0,sigma_xi),dim=-1)
        elif par.Nstates_fixed == 2:
            return torch.stack((m0,p0,sigma_xi,sigma_psi),dim=-1)
        elif par.Nstates_fixed == 3:
            return torch.stack((m0,p0,sigma_xi,sigma_psi,rho_p),dim=-1)
        
    def draw_shocks(self,N):
        """ draw shocks """

        par = self.par

        xi = torch.normal(0.0,1.0,size=(par.T,N))
        psi = torch.normal(0.0,1.0,size=(par.T,N))

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
    
    terminal_actions = model_funcs.terminal_actions
    terminal_reward_pd = model_funcs.terminal_reward_pd
        
    state_trans_pd = model_funcs.state_trans_pd
    state_trans = model_funcs.state_trans
    exploration = model_funcs.exploration

    marginal_reward = model_funcs.marginal_reward
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

    def compute_MPC(self,Nbatch_share=0.01):
        """ compute MPC """

        par = self.par
        sim = self.sim
        
        with torch.no_grad():

            # a. baseline
            c = sim.outcomes[...,0]

            # b. add windfall
            states = deepcopy(sim.states)
            states[:,:,0] += par.Delta_MPC

            # b. alternative
            c_alt = torch.ones_like(c)
            c_alt[-1] = c[-1] + par.Delta_MPC
            Nbatch = int(Nbatch_share*sim.N)
            for i in range(0,sim.N,Nbatch):

                index_start = i
                index_end = i + Nbatch
                        
                states_ = states[:par.T-1,index_start:index_end,:]
                actions = self.eval_policy(self.policy_NN,states_)
                outcomes = self.outcomes(states_,actions)
                c_alt[:par.T-1,index_start:index_end] = outcomes[:,:,0]

            # c. MPC
            sim.MPC[:,:] = (c_alt-c)/par.Delta_MPC	

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
                states = sim.states[:par.T-1,index_start:index_end]
                actions = sim.actions[:par.T-1,index_start:index_end]
                outcomes = sim.outcomes[:par.T-1,index_start:index_end]

                # b. post-decision states
                states_pd = self.state_trans_pd(states,actions,outcomes)

                # c. next-period states
                states_next = self.state_trans(states_pd,train.quad)

                # d. next-period action
                actions_next_before = self.eval_policy(self.policy_NN,states_next[:par.T-2],t0=1)
                actions_next_after = self.terminal_actions(states_next[par.T-2])
                actions_next = torch.cat((actions_next_before,actions_next_after[None,...]),dim=0)

                # e. next-period consumption
                c_next = self.outcomes(states_next,actions_next).squeeze(dim=-1)

                # f. marginal utility next period
                marg_util_next = model_funcs.marg_util_c(c_next)

                # g. expected marginal utility next period
                exp_marg_util_next = torch.sum(train.quad_w[None,None,:]*marg_util_next, dim=-1)

                # h. euler error
                euler_error_Nbatch = model_funcs.inverse_marg_util(par.R*par.beta*exp_marg_util_next) / c - 1
                euler_error_Nbatch = torch.abs(euler_error_Nbatch)
                sim.euler_error[:par.T-1,index_start:index_end] = euler_error_Nbatch

    ########
    # misc #
    ########

    def compute_policy_on_grids(self,egm,transformation_function):
        """ 
        compute state-network on EGM grids

        NN: state-dependent Neural net that you want to get computed in grids
        transformation_function: function that takes state and action and yields transformation that you want to put on grid	
        
        """

        # a. unpack
        par = self.par
        train = self.train

        m_grid = egm.m_grid
        p_grid = egm.p_grid
        sigma_xi_grid = egm.sigma_xi_grid
        sigma_psi_grid = egm.sigma_psi_grid
        rho_p_grid = egm.rho_p_grid

        # a. create array for storing values
        egm_sol_shape = egm.sol_con.shape
        sol_con_grid = np.zeros(egm_sol_shape)

        # b. create tensor product data
        if par.Nstates_fixed == 0:
            p, m = np.meshgrid(p_grid,m_grid,indexing='ij')
            states_grid = np.concatenate((m.reshape(-1,1),p.reshape(-1,1)),axis=1)
        elif par.Nstates_fixed == 1:
            p,sigma_xi,m = np.meshgrid(p_grid,sigma_xi_grid,m_grid,indexing='ij')
            states_grid = np.concatenate((m.reshape(-1,1),p.reshape(-1,1),sigma_xi.reshape(-1,1)),axis=1)
        elif par.Nstates_fixed == 2:
            p,sigma_xi,sigma_psi,m = np.meshgrid(p_grid,sigma_xi_grid, sigma_psi_grid,m_grid,indexing='ij')
            states_grid = np.concatenate((m.reshape(-1,1),p.reshape(-1,1),sigma_xi.reshape(-1,1),sigma_psi.reshape(-1,1)),axis=1)
        elif par.Nstates_fixed == 3:
            p,sigma_xi,sigma_psi,rho_p,m = np.meshgrid(p_grid,sigma_xi_grid, sigma_psi_grid,rho_p_grid, m_grid,indexing='ij')
            states_grid = np.concatenate((m.reshape(-1,1),p.reshape(-1,1),sigma_xi.reshape(-1,1),sigma_psi.reshape(-1,1), rho_p.reshape(-1,1)),axis=1)			

        # c. compute
        states_grid = torch.tensor(states_grid,dtype=train.dtype,device=train.device)
        for t in range(par.T-1):
            
            # i. evaluate
            with torch.no_grad(): 
                output = self.eval_policy(self.policy_NN,states_grid,t=t)

            # ii. transformation
            output = transformation_function(states_grid,output).cpu().numpy()

            # iii. store
            for i_p in range(egm.Np):
                for i_sigma_psi in range(egm.Nsigma_psi):
                    for i_sigma_xi in range(egm.Nsigma_xi):
                        for i_rho_p in range(egm.Nrho_p):
                            m_index = i_rho_p + egm.Nrho_p*i_sigma_psi + egm.Nrho_p*egm.Nsigma_psi*i_sigma_xi+egm.Nrho_p*egm.Nsigma_psi*egm.Nsigma_xi*i_p
                            sol_con_grid[t,i_p,i_sigma_xi,i_sigma_psi,i_rho_p,:] = output[egm.Nm*m_index:egm.Nm*(m_index+1)]

        return sol_con_grid	
        
def select_euler_errors(model):
    """ compute mean euler error """

    par = model.par
    sim = model.sim

    savings_indicator = (sim.actions[:par.T_retired,:,0] > par.Euler_error_min_savings)
    return sim.euler_error[:par.T_retired,:][savings_indicator]

def mean_log10_euler_error_working(model):

    euler_errors = select_euler_errors(model).cpu().numpy()
    I = np.isclose(np.abs(euler_errors),0.0)

    return np.mean(np.log10(np.abs(euler_errors[~I])))