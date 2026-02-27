import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE' # without this python may crash when plotting from matplotlib
os.environ['MKL_THREADING_LAYER'] = 'GNU'
import numpy as np
import torch
torch.set_warn_always(False)

from consav.quadrature import log_normal_gauss_hermite
from copy import deepcopy
from EconDLSolvers import DLSolverClass
from consav.grids import nonlinspace

# local
import model_funcs

def get_omega_delta_d_ubar(D,delta,d_ubar,train=None,scale=0.6,omega_base=0.2):
    """ get omega, delta, and d_ubar for a given number of durables """

    omega = torch.tensor([np.exp(-scale*i) for i in range(D)], device=train.device, dtype=train.dtype) # omega is the initial stock of durables
    omega = omega / torch.sum(omega) * omega_base # scale to base value

    delta = torch.tensor([delta for i in range(D)], device=train.device, dtype=train.dtype)
    d_ubar = torch.tensor([d_ubar for i in range(D)], device=train.device, dtype=train.dtype)

    return omega, delta, d_ubar

class NonConvexDurablesModelClass(DLSolverClass):

    def setup(self,full=None):
        """ choose parameters """

        par = self.par
        sim = self.sim

        par.full = full if full is not None else torch.cuda.is_available()
        par.seed = 1 # seed for random number generator in torch

        # a. model
        par.D = 1 # number of durables

        # horizon
        par.T = 20 # number of periods

        # preferences
        par.beta = 0.965 # discount factor
        par.d_ubar = 1e-2 # minimum consumption
        par.rho = 2.0 # risk aversion

        # return, durable good and ince income
        par.R = 1.03 # gross return
        par.nu = 0.10 # adjustment cost parameter
        par.delta = 0.1 # depreciation rate
        
        par.sigma_xi = 0.1 # std of persistent income shock
        par.sigma_psi = 0.10 # std of transitory income shock
        par.eta = 0.95 # persistence of permanent income

        par.Nxi = 4 # number of persistent income shocks - quadrature
        par.Npsi = 4 # number of transitory income shocks - quadrature

        # taste shocks
        par.sigma_eps = 0.10 # scale of taste shocks
        par.exp_noise_taste = 1.05 # multiplier of sigma when simulation for exploration

        # number of states, shocks and actions
        par.Nshocks = 2 # number of shocks

        # b. simulation of life-time-reward
        sim.N = 100_000 # number of agents
        sim.reps = 0 # number of repetitions

        # initial states
        par.mu_m0 = 1.0 
        par.sigma_m0 = 0.1 
        par.mu_p0 = 1.0 
        par.sigma_p0 = 0.1
        par.mu_n10 = 0.0
        par.sigma_n10 = 0.01		
        par.mu_n20 = 0.0
        par.sigma_n20 = 0.01	

        
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
            sim.N = 5000 

        par.Nstates = 3 if par.D == 1 else 4 # number of states
        par.Nstates_pd = 3 if par.D == 1 else 4 # number of post-decision states
        par.Nactions = 5 if par.D == 1 else 12
        par.Noutcomes = 6 if par.D == 1 else 16
        par.NDC = 2 if par.D == 1 else 4 # number of discrete choices
        par.choices_from_iDC, par.iDC_from_choices, par.actions_from_iDC, par.actions_from_i_actions, par.Nactions = get_indices(par)

        # b. dependent parameters
        par.kappa = torch.ones(par.T, device=train.device)
        par.omega,par.delta,par.d_ubar = get_omega_delta_d_ubar(par.D,par.delta,par.d_ubar, train=train)

        # c. quad
        par.xi, par.xi_w = log_normal_gauss_hermite(par.sigma_xi, par.Nxi)
        par.psi, par.psi_w = log_normal_gauss_hermite(par.sigma_psi, par.Npsi)

        par.psi_w = torch.tensor(par.psi_w,dtype=dtype,device=device)
        par.psi = torch.tensor(par.psi,dtype=dtype,device=device)
        par.xi_w = torch.tensor(par.xi_w,dtype=dtype,device=device)
        par.xi = torch.tensor(par.xi,dtype=dtype,device=device)		

        # d. simulation
        sim.states = torch.zeros((par.T,sim.N,par.Nstates),dtype=dtype,device=device)
        sim.states_pd = torch.zeros((par.T,sim.N,par.Nstates_pd),dtype=dtype,device=device)
        sim.shocks = torch.zeros((par.T,sim.N,par.Nshocks),dtype=dtype,device=device)
        sim.outcomes = torch.zeros((par.T,sim.N,par.Noutcomes),dtype=dtype,device=device) 
        sim.actions = torch.zeros((par.T,sim.N,par.Nactions),dtype=dtype,device=device) 
        sim.reward = torch.zeros((par.T,sim.N,par.NDC),dtype=dtype,device=device)
        pos = np.array([1,5,10,25,50,100,500,1000,1500,2000,2500,3000,3500,4000,4500,5000])
        neg = np.array([1,5,10,25,50,100,500,1000,1500,2000,2500,3000,3500,4000,4500,5000])
        transfer_grid = np.concatenate((-np.flip(neg),np.zeros(1),pos))/10_000
        
        sim.transfer_grid = torch.tensor(transfer_grid,dtype=dtype,device=device)
        sim.Ntransfer = sim.transfer_grid.shape[0]
        sim.R_transfer = torch.zeros(sim.Ntransfer,dtype=dtype,device=device)
        sim.individual_R_transfer = torch.zeros((sim.Ntransfer,sim.N),dtype=dtype,device=device)        
        sim.taste_shocks = torch.zeros((par.T,sim.N,par.NDC),dtype=dtype,device=device)		
        sim.DC = torch.zeros((par.T,sim.N),dtype=dtype,device=device)
        
        sim.adj = torch.zeros((par.T,sim.N),dtype=dtype,device=device)
        sim.c = torch.zeros((par.T,sim.N),dtype=dtype,device=device)
        sim.d1 = torch.zeros((par.T,sim.N),dtype=dtype,device=device)
        sim.d2 = torch.zeros((par.T,sim.N),dtype=dtype,device=device)

        sim.euler_error_c = torch.zeros((par.T,sim.N),dtype=dtype,device=device)

    #########
    # train #
    #########

    def setup_train(self):
        """ default parameters for training """

        par = self.par
        train = self.train
        dtype = train.dtype
        device = train.device

        # a. neural net settings
        if par.full:
            train.Nneurons_value = np.array([500,500]) # number of neurons in each layer
            train.Nneurons_policy = np.array([500,500]) # number of neurons in each layer
        else:
            train.Nneurons_value = np.array([100,100])
            train.Nneurons_policy = np.array([100,100])

        train.budget_shares = False
        
        # d. misc
        train.epsilon_sigma_decay = 1.0 # decay of epsilon_sigma
        train.start_train_policy = 50 # start training policy net
        train.epsilon_sigma_consec_sigmoids = 0.1 # std. of exploration shocks wrt. all continuous choices, when solving with consecutive sigmoids
        train.N_value_NN = 3 # number of value nets
        train.use_FOC = False # use first order conditions
        train.NFOC_targets = 1 # number of targets in FOC
        
    def allocate_train(self):
        """ allocate memory training """

        train = self.train
        par = self.par
        train = self.train
        dtype = train.dtype
        device = train.device

        # a. dependent settings
        if train.budget_shares:
            
            # activation
            train.policy_activation_final = [None for _ in range(par.Nactions)] # no activation in policy net

            # exploration
            train.epsilon_sigma = torch.tensor([1.0 for _ in range(par.Nactions)]) # std of exploration shocks
            train.epsilon_sigma_min = torch.tensor([0.0 for _ in range(par.Nactions)]) # minimum epsilon_sigma

            # clipping
            train.min_actions = torch.tensor([-100000.0 for _ in range(par.Nactions)],dtype=dtype,device=device) # minimum action value
            train.max_actions = torch.tensor([100000.0 for _ in range(par.Nactions)],dtype=dtype,device=device) # maximum action value	
        
        else: # consecutive sigmoids

            # activation
            train.policy_activation_final = ['sigmoid' for _ in range(par.Nactions)] # no activation in policy net

            # exploration
            train.epsilon_sigma = torch.tensor([train.epsilon_sigma_consec_sigmoids for _ in range(par.Nactions)]) # std of exploration shocks
            train.epsilon_sigma_min = torch.tensor([0.0 for _ in range(par.Nactions)]) # minimum epsilon_sigma

            # clipping
            train.min_actions = torch.tensor([1e-8 for _ in range(par.Nactions)],dtype=dtype,device=device) # minimum action value
            train.max_actions = torch.tensor([0.9999 for _ in range(par.Nactions)],dtype=dtype,device=device) # maximum action value			

        # b. training samples
        train.states = torch.zeros((par.T,train.N,par.Nstates),dtype=dtype,device=device)
        train.states_pd = torch.zeros((par.T,train.N,par.Nstates_pd),dtype=dtype,device=device)
        train.shocks = torch.zeros((par.T,train.N,par.Nshocks),dtype=dtype,device=device)
        train.taste_shocks = torch.zeros((par.T,train.N,par.NDC),dtype=dtype,device=device)
        train.outcomes = torch.zeros((par.T,train.N,par.Noutcomes),dtype=dtype,device=device)
        train.DC = torch.zeros((par.T,train.N),dtype=dtype,device=device)
        train.actions = torch.zeros((par.T,train.N,par.Nactions),dtype=dtype,device=device) 
        train.reward = torch.zeros((par.T,train.N,par.NDC),dtype=dtype,device=device)
        
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
        n10 = par.mu_n10*torch.exp(torch.normal(-0.5*par.sigma_n10**2,par.sigma_n10,size=(N,)))

        if par.D == 1:
        
            return torch.stack((m0,p0,n10),dim=-1)
        
        else:

            n20 = par.mu_n20*torch.exp(torch.normal(-0.5*par.sigma_n20**2,par.sigma_n20,size=(N,)))

            return torch.stack((m0,p0,n10,n20),dim=-1)

    def draw_shocks(self,N, training = False):
        """ draw shocks """

        par = self.par

        # a. taste shocks
        if training: 
            dist = torch.distributions.Gumbel(0,par.exp_noise_taste * par.sigma_eps)
        else:
            dist = torch.distributions.Gumbel(0,par.sigma_eps)
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

    def numerical_integration(self):
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
    terminal_marginal_reward_pd = model_funcs.terminal_marginal_reward_pd
    eval_equations_DeepVPDDC = model_funcs.eval_equations_DeepVPDDC
    eval_equations_DeepVPDDC_terminal = model_funcs.eval_equations_DeepVPDDC_terminal

    def add_transfer(self,transfer):
        """ add transfer to initial states """

        par = self.par
        sim = self.sim

        sim.states[0,:,0] += transfer

    ############
    # simulate #
    ############

    def compute_model_moments(self):
        
        # a. unpack
        train = self.train
        sim = self.sim
        info = self.info
        par = self.par
        
        dtype = sim.states.dtype 
        device = sim.states.device
        Noutcomes_dc = 3 if par.D == 1 else 4

        # b. create and fill arrays
        c = torch.zeros((par.T,sim.N),dtype=dtype,device=device)
        d1 = torch.zeros((par.T,sim.N),dtype=dtype,device=device)
        d2 = torch.zeros((par.T,sim.N),dtype=dtype,device=device)

        for i_DC in range(par.NDC):
            offset = Noutcomes_dc * i_DC
            c[sim.DC == i_DC] = sim.outcomes[..., offset + 0][sim.DC == i_DC]
            
            if par.D == 1:
                d1[sim.DC == i_DC] = sim.outcomes[..., offset + 1][sim.DC == i_DC]
            else:
                d1[sim.DC == i_DC] = sim.outcomes[..., offset + 1][sim.DC == i_DC]
                d2[sim.DC == i_DC] = sim.outcomes[..., offset + 2][sim.DC == i_DC]

        # c. store
        info[('mean_coh',train.k)] = sim.states[...,0].mean(axis=-1)
        info[('mean_consumption',train.k)] = c.mean(axis=-1)
        if par.D == 1:
            info[('mean_d1',train.k)] = d1.mean(axis=-1)
        else:
            info[('mean_d1',train.k)] = d1.mean(axis=-1)
            info[('mean_d2',train.k)] = d2.mean(axis=-1)
        

    def more_simulation_outcomes(self):
        """ compute more simulation outcomes """

        par = self.par
        sim = self.sim
        Noutcomes_dc = 3 if par.D == 1 else 4
        
        # D = 1
        for i_DC in range(par.NDC):
            offset = Noutcomes_dc * i_DC
            sim.c[sim.DC == i_DC] = sim.outcomes[..., offset + 0][sim.DC == i_DC]
            
            if par.D == 1:
                sim.d1[sim.DC == i_DC] = sim.outcomes[..., offset + 1][sim.DC == i_DC]
            else:
                sim.d1[sim.DC == i_DC] = sim.outcomes[..., offset + 1][sim.DC == i_DC]
                sim.d2[sim.DC == i_DC] = sim.outcomes[..., offset + 2][sim.DC == i_DC]

    def euler_errors_DL(self, Nbatch_share=0.01):
        """ compute Euler errors for discrete choices with shock integration."""

        par = self.par
        sim = self.sim
        train = self.train

        dtype  = sim.c.dtype
        device = sim.c.device
        policy_NN = self.policy_NN

        if self.value_NN is not None:
            value_NN = self.value_NN
        else:
            value_NN = self.value_NNs

        # a. ensure at least one per batch
        Nbatch = max(1, int(Nbatch_share * sim.N))

        # b. per-DC block width in outcomes: [c, d_1..d_D, (typically a)]
        block_w = 2 + par.D

        for i in range(0, sim.N, Nbatch):

            idx0, idx1 = i, min(i + Nbatch, sim.N)

            with torch.no_grad():

                # i) unpack current-period stuff (t = 0..T-2)
                states_pd = sim.states_pd[:par.T-1, idx0:idx1]
                euler_error  = sim.euler_error_c[:par.T-1, idx0:idx1]
                d1 = sim.d1[:par.T-1, idx0:idx1]
                c_today = sim.c[:par.T-1,  idx0:idx1]

                # stack current-period durable stocks for today's inverse MU
                if par.D == 2:
                    d2 = sim.d2[:par.T-1, idx0:idx1]
                    d_stacked = torch.stack((d1, d2), dim=-1)
                else:
                    d_stacked = d1.unsqueeze(-1)

                # ii) next states, actions, outcomes, post-decision states
                states_next = self._state_trans(states_pd)  # (T-1, B, Nmc, ...)
                actions_next = self.eval_policy(policy_NN, states_next, t0=1)
                outcomes_next = self.outcomes(states_next, actions_next, t0=1)
                states_pd_next = self.state_trans_pd(states_next, actions_next, outcomes_next, t0=1)
                states_pd_next = states_pd_next.permute(0,1,2,4,3) # swap last two dimensions, shape = (T,N,Numint,Nstates_pd,NDC)

				# iii) per-DC value next period (KEEP ALL DCs — do not drop a column)
                reward_next = self.reward(states_next, actions_next, outcomes_next, t0=1)
                value_pd_next_ = self.eval_value(value_NN, states_pd_next[:-1], t0=1)[...,0]
                value_pd_next_terminal = self.terminal_reward_pd(states_pd_next[-1:])
                value_pd_next = torch.cat([value_pd_next_,value_pd_next_terminal],dim=0)
                value_next = reward_next + par.beta * value_pd_next

                # iv) choice probs from logit scale (temperature = par.sigma_eps)
                choice_prob = torch.softmax(value_next / par.sigma_eps, dim=-1)

                # v) expected marginal utility of c, integrating over shocks and DCs
                exp_marg_util_c = torch.zeros(
                    (par.T-1,Nbatch, par.NDC), dtype=dtype, device=device
                )

                for dc in range(par.NDC):
                    base = dc * block_w
                    c_next = outcomes_next[..., base + 0]
                    d_next = outcomes_next[..., base + 1 : base + 1 + par.D]

                    # inside the DC loop
                    mu_next = model_funcs.marg_u_c_func(c_next, d_next, par)
                    p_dc  = choice_prob[..., dc]                            
                    exp_marg_util_c[..., dc] = torch.sum(train.numint_weights[None, None, :] *
                                                        mu_next * p_dc, dim=-1)

                # sum over DCs
                exp_marg_util_final = torch.sum(exp_marg_util_c, dim=-1) 

                # vi) Euler error: u_c^{-1}(β R E_t[u_c']) / c_t - 1
                euler_error[:] = (
                    model_funcs.inv_marg_u_c_func(par.beta * par.R * exp_marg_util_final,
                                                d_stacked, par) / c_today - 1)
                                        

def get_indices(par):
    """ 
    
    Compute lookup/index structures for discrete and continuous actions:
    1. choices_from_iDC: maps i_DC -> discrete choice dicts (adjust1, adjust2?)
    2. iDC_from_choices: maps discrete choice dict -> i_DC
    3. actions_from_iDC: maps i_DC -> continuous action indices
    4. actions_from_i_actions: maps i_action -> action name

    """

    NDC = par.NDC
    choices_from_iDC = []
    iDC_from_choices = {}
    actions_from_iDC = [{} for _ in range(NDC)]
    actions_from_i_actions = []

    i_DC = 0
    i_action = 0

    # determine discrete choice sets based on D
    adjust1_options = ['keep', 'adjust']
    adjust2_options = ['keep', 'adjust'] if par.D == 2 else [None]

    for adjust1 in adjust1_options:
        for adjust2 in adjust2_options:

            # a. dkip adjust2 when D=1 (adjust2=None)
            dict_choice = {'adjust1': adjust1}
            if par.D == 2: dict_choice['adjust2'] = adjust2

            # b. store mapping from i_DC to choices
            choices_from_iDC.append(dict_choice)
            iDC_from_choices[tuple(dict_choice.values())] = i_DC

            # c. continuous actions
            
            # base actions
            actions_from_iDC[i_DC]['savings_rate'] = i_action
            actions_from_i_actions.append('savings')
            i_action += 1

            actions_from_iDC[i_DC]['consumption_rate'] = i_action
            actions_from_i_actions.append('consumption')
            i_action += 1

            # extra actions if discrete adjustments are active
            if adjust1 == 'adjust':
                actions_from_iDC[i_DC]['durable1_share'] = i_action
                actions_from_i_actions.append('durable1_share')
                i_action += 1

            if par.D == 2 and adjust2 == 'adjust':
                actions_from_iDC[i_DC]['durable2_share'] = i_action
                actions_from_i_actions.append('durable2_share')
                i_action += 1

            i_DC += 1

    Nactions = i_action
    choices_from_iDC = np.array(choices_from_iDC, dtype=object)

    return choices_from_iDC, iDC_from_choices, actions_from_iDC, actions_from_i_actions, Nactions