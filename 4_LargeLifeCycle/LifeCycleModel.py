import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE' # without this python may crash when plotting from matplotlib
import numpy as np
from copy import deepcopy
import torch
torch.set_warn_always(True)

from consav.quadrature import log_normal_gauss_hermite, normal_gauss_hermite
from EconDLSolvers import DLSolverClass, torch_uniform
import model_funcs

# class
class LifeCycleModelClass(DLSolverClass):

    #########
    # setup #
    #########

    def setup(self, full=None):
        """ choose parameters """

        par = self.par
        sim = self.sim

        par.full = full if not full is None else self.train.device != 'cpu'
        par.seed = 42  # seed for random number generator in torch

        # general parameters #
        par.T = 20  # lifetime
        par.sigma = 2  # CRRA

        # job-ladder block #

        # search and work disutility
        par.phi = 0.2
        par.chi_e = 1.0 / (1+par.phi)

        # job-switching probabilities
        par.pi_job_loss = 0.05
        par.pi_job_find = 0.2
        par.eta = 3.87  # Pareto shape parameter for job offer distribution

        # human capital
        par.mu_p_e = 0.0035
        par.mu_p_u = -0.0035
        par.sigma_p = 0.037
        par.p_min = 0.1

        # wage distribution
        par.chi_b = 0.1  # unemployment benefits parameter
        par.b_bar = 0.3  # maximum unemployment benefits
        par.sigma_w = 0.2  # wage dispersion parameter
        par.w_min = 1.0  # minimum wage

        # portfolio block
        # return
        par.r = 0.05  # return
        par.beta = 1/(1+par.r) * 0.95
        par.ra_sigma = 0.157  # return volatility
        par.rb = 0.02
        par.F = 0.05  # fixed cost of portfolio choice

        # housing block 
        par.alpha = 0.8875 # consumption preference
        par.delta_h = 0.022 # housing maintenance cost
        par.tau_h = 0.05 # proportional housing adjustment cost
        par.rent_price_ratio = 0.06  # rent-to-house-price ratio
        par.p_h = 1.0 # house prices
        par.rent = par.rent_price_ratio * par.p_h  # rent
        par.share_income_protected_renter = 0.1 # share of wage income protected for renters in case of negative coh, 
        # ensure renting is always feasible but costly when defaulting

        # mortgage
        par.tau_m = 0.05  # proportional mortgage adjustment cost
        par.llambda = 0.9  # 1-downpayment share
        par.kappa = 0.03  # mortgage spread
        par.r_m = par.r + par.kappa  # mortgage rate

        # taste shock
        par.sigma_eps = 0.1
        par.exp_noise_taste = 1.05 # multiplier of sigma when simulation for exploration

        # quadrature
        par.Nr = 5  # interest rate grid
        par.Nw = 5  # wage grid
        par.Np_eps = 5  # human capital grid

        # monte carlo
        par.Nmc = 100 # number of monte carlo draws for psi, xi
        par.do_antithetic_mc = True # use antithetic sampling

        # simulation    
        sim.N = 100_000  # number of agents
        sim.reps = 0  # number of repetitions
    
        par.mu_b0 = 0.0
        par.sigma_b0 = 0.1

        par.mu_p0 = 1.0
        par.sigma_p0 = 0.1


    def allocate(self):
        """ allocate arrays  """

        # unpack
        par = self.par
        sim = self.sim
        train = self.train

        dtype = train.dtype
        device = train.device

        if par.full == False:
            par.T = 3
            sim.N = 5000
            sim.reps = 0

        # states and shocks
        par.Nstates = 7
        par.Nstates_pd = 7
        par.Nshocks = 5  # number of shocks
        par.NDC = 2 * 2 * 3  # number of discrete choices
        par.Noutcomes = 8  # number of outcomes

        # Get actions index
        # actions index for discrete choices
        par.choices_from_iDC, par.iDC_from_choices, par.actions_from_iDC, par.actions_from_i_actions, par.Nactions = get_indices(par)
        par.iDC_work = np.array([i for i, choice in enumerate(par.iDC_from_choices) if 'work' in choice])
        par.iDC_adj = np.array([i for i, choice in enumerate(par.iDC_from_choices) if 'adjust' in choice])
        par.iDC_renter = np.array([i for i, choice in enumerate(par.iDC_from_choices) if 'renter' in choice])
        par.iDC_owner = np.array([i for i, choice in enumerate(par.iDC_from_choices) if 'owner' in choice])
        par.iDC_buyer = np.array([i for i, choice in enumerate(par.iDC_from_choices) if 'buyer' in choice])
        par.iDC_refinance = np.array([i for i, choice in enumerate(par.iDC_from_choices) if 'refinance' in choice])

        # quadrature grids
        
        # wages
        par.w_grid, par.w_weight = log_normal_gauss_hermite(
            mu=1.0, sigma=par.sigma_w, n=par.Nw)
        # par.p_grid, par.p_weight = pareto_gauss_laguerre(par.eta, x_min=1, N=par.Np)
        par.w_grid = torch.tensor(par.w_grid, dtype=dtype, device=device)
        par.w_weight = torch.tensor(par.w_weight, dtype=dtype, device=device)

        # human capital
        par.p_eps_grid, par.p_eps_w = normal_gauss_hermite(
            mu=0, sigma=1.0, n=par.Np_eps)
        par.p_eps_grid = torch.tensor(
            par.p_eps_grid, dtype=dtype, device=device)
        par.p_eps_w = torch.tensor(par.p_eps_w, dtype=dtype, device=device)

        # risky return
        par.ra, par.ra_w = normal_gauss_hermite(
            sigma=par.ra_sigma, mu=par.r, n=par.Nr)
        par.ra = torch.tensor(par.ra, dtype=dtype, device=device)
        par.ra_w = torch.tensor(par.ra_w, dtype=dtype, device=device)

        # simulation
        sim.states = torch.zeros(
            (par.T, sim.N, par.Nstates), dtype=dtype, device=device)
        sim.states_pd = torch.zeros(
            (par.T, sim.N, par.Nstates_pd), dtype=dtype, device=device)
        sim.shocks = torch.zeros(
            (par.T, sim.N, par.Nshocks), dtype=dtype, device=device)
        sim.outcomes = torch.zeros(
            (par.T, sim.N, par.Noutcomes, par.NDC), dtype=dtype, device=device)
        sim.actions = torch.zeros(
            (par.T, sim.N, par.Nactions), dtype=dtype, device=device)
        sim.reward = torch.zeros(
            (par.T, sim.N, par.NDC), dtype=dtype, device=device)
        sim.taste_shocks = torch.zeros(
            (par.T, sim.N, par.NDC), dtype=dtype, device=device)
        sim.DC = torch.zeros((par.T, sim.N), dtype=dtype, device=device)
        sim.euler_error = torch.zeros((par.T, sim.N), dtype=dtype, device=device)

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

        # a. number of neurons in each layer
        if par.full:
            train.Nneurons_value = np.array([500, 500])
            train.Nneurons_policy = np.array([500, 500])
        else:
            train.Nneurons_value = np.array([50, 50])
            train.Nneurons_policy = np.array([50, 50])

        # b. activation function for final layer
        train.policy_activation_final = ['sigmoid'] * par.Nactions
        
        # c. action bounds
        train.min_actions = torch.zeros(par.Nactions, dtype=dtype, device=device) 
        train.max_actions = torch.ones(par.Nactions, dtype=dtype, device=device) * 0.999

        i_housing_share = np.array([i for i, action in enumerate(par.actions_from_i_actions) if action == 'housing_share'])
        train.min_actions[i_housing_share] = 0.01  # minimum housing share

        # d. exploration noise for continuous actions (std of exploration shocks)
        train.epsilon_sigma = torch.ones(par.Nactions, dtype=dtype, device=device) * 0.25  # std of exploration shocks
        train.epsilon_sigma_decay = 1.0  # decay of epsilon_sigma
        train.epsilon_sigma_min = torch.zeros(par.Nactions, dtype=dtype, device=device)  # minimum epsilon_sigma

        # e. learning
        train.learning_rate_policy = 1e-4  # learning rate
        train.learning_rate_policy_decay = 0.9999
        train.learning_rate_policy_min = 1e-5

        train.learning_rate_value = 1e-3 # learning rate
        train.learning_rate_value_decay = 0.9999
        train.learning_rate_value_min = 1e-5

        # f. misc
        train.use_input_scaling = True  # use input scaling
        train.do_exo_actions = 100
        train.N_value_NN = 3 # use multiple value networks to increase precision
        train.use_quad = True
        train.N_target_batches = 3

        
    def allocate_train(self):
        """ allocate memory training """

        par = self.par
        train = self.train
        dtype = train.dtype
        device = train.device

        train.terminate_on_policy_loss = 'time_iteration'

        # b. simulation
        train.states = torch.zeros(
            (par.T, train.N, par.Nstates), dtype=dtype, device=device)
        train.states_pd = torch.zeros(
            (par.T, train.N, par.Nstates_pd), dtype=dtype, device=device)
        train.shocks = torch.zeros(
            (par.T, train.N, par.Nshocks), dtype=dtype, device=device)
        train.outcomes = torch.zeros(
            (par.T, train.N, par.Noutcomes, par.NDC), dtype=dtype, device=device)
        train.actions = torch.zeros(
            (par.T, train.N, par.Nactions), dtype=dtype, device=device)
        train.reward = torch.zeros(
            (par.T, train.N, par.NDC), dtype=dtype, device=device)
        train.taste_shocks = torch.zeros(
            (par.T, train.N, par.NDC), dtype=dtype, device=device)
        train.DC = torch.zeros((par.T, train.N), dtype=dtype, device=device)

    ##############
    # quadrature #
    ##############

    def numerical_integration(self):
        """ quadrature nodes and weights """

        par = self.par
        train = self.train

        if train.use_quad:
            # grid
            job_loss = torch.tensor(
                [0.0, 1.0], dtype=self.train.dtype, device=self.train.device)
            job_find = torch.tensor(
                [0.0, 1.0], dtype=self.train.dtype, device=self.train.device)

            job_loss_grid, job_find_grid, w_grid, p_eps_grid, ra_grid = torch.meshgrid(
                job_loss, job_find, par.w_grid, par.p_eps_grid, par.ra, indexing='ij')
            nodes = torch.stack((job_loss_grid.flatten(), job_find_grid.flatten(
            ), w_grid.flatten(), p_eps_grid.flatten(), ra_grid.flatten()), dim=-1)

            # weight
            job_loss_w = torch.tensor(
                [1-par.pi_job_loss, par.pi_job_loss], device=self.train.device, dtype=self.train.dtype)
            job_find_w = torch.tensor(
                [1-par.pi_job_find, par.pi_job_find], device=self.train.device, dtype=self.train.dtype)
            job_loss_w_grid, job_find_w_grid, w_weight, p_eps_w, ra_w = torch.meshgrid(job_loss_w.flatten(
            ), job_find_w.flatten(), par.w_weight.flatten(), par.p_eps_w.flatten(), par.ra_w.flatten(), indexing='ij')
            weights = job_loss_w_grid.flatten()*job_find_w_grid.flatten() * \
                w_weight.flatten()*p_eps_w.flatten()*ra_w.flatten()
        else:
            assert par.Nmc % 2 == 0, 'number of mc draws must be even for antithetic sampling'

            if par.do_antithetic_mc:
                assert par.Nmc % 2 == 0, "Nmc must be even for antithetic sampling"
                half = par.Nmc // 2
                T1, N = par.T - 1, train.N
                device, dtype = train.device, train.dtype

                # Containers
                job_loss = torch.empty((T1, N, par.Nmc), device=device, dtype=dtype)
                job_find = torch.empty((T1, N, par.Nmc), device=device, dtype=dtype)
                w       = torch.empty((T1, N, par.Nmc), device=device, dtype=dtype)
                p_eps   = torch.empty((T1, N, par.Nmc), device=device, dtype=dtype)
                ra      = torch.empty((T1, N, par.Nmc), device=device, dtype=dtype)

                # --- Bernoulli antithetic via uniforms (correct margins = p)
                U_loss = torch.rand((T1, N, half), device=device, dtype=dtype)
                U_find = torch.rand((T1, N, half), device=device, dtype=dtype)

                loss_1 = (U_loss < par.pi_job_loss).to(dtype)
                loss_2 = ((1.0 - U_loss) < par.pi_job_loss).to(dtype)
                find_1 = (U_find < par.pi_job_find).to(dtype)
                find_2 = ((1.0 - U_find) < par.pi_job_find).to(dtype)

                job_loss[..., :half] = loss_1
                job_loss[..., half:] = loss_2
                job_find[..., :half] = find_1
                job_find[..., half:] = find_2

                # --- Gaussian antithetic at the z-level
                z_w   = torch.randn((T1, N, half), device=device, dtype=dtype)
                z_p   = torch.randn((T1, N, half), device=device, dtype=dtype)
                z_ra  = torch.randn((T1, N, half), device=device, dtype=dtype)

                z_w_full  = torch.cat([z_w,  -z_w ], dim=2)
                z_p_full  = torch.cat([z_p,  -z_p ], dim=2)
                z_ra_full = torch.cat([z_ra, -z_ra], dim=2)

                # Wage (lognormal with mean 1): exp(σ z - 0.5 σ^2)
                w[:]     = torch.exp(par.sigma_w * z_w_full - 0.5 * par.sigma_w**2)

                # Human-capital shock N(0,1)
                p_eps[:] = z_p_full

                # Risky return Normal(r, σ)
                ra[:]    = par.r + par.ra_sigma * z_ra_full

                # Stack nodes and weights (broadcastable over shock axis)
                nodes   = torch.stack((job_loss, job_find, w, p_eps, ra), dim=-1)   # (T-1, N, Nmc, 5)
                weights = torch.full((par.Nmc,), 1.0 / par.Nmc, device=device, dtype=dtype)

            else:
                # Job finding / loss shocks
                job_loss = torch.bernoulli(par.pi_job_loss*torch.ones((par.T-1, train.N, par.Nmc)))
                job_find = torch.bernoulli(par.pi_job_find*torch.ones((par.T-1, train.N, par.Nmc)))

                # Wage shocks
                # pareto = torch.distributions.Pareto(scale=torch.tensor(1.0), alpha=torch.tensor(par.eta))
                # w = par.w_min * pareto.sample((par.T,N,))
                w = torch.normal(0.0, 1.0, size=(par.T-1, train.N, par.Nmc))
                w = torch.exp(par.sigma_w*w-0.5*par.sigma_w**2)

                # Human capital shocks
                p_eps = torch.normal(0, 1, (par.T-1, train.N, par.Nmc))

                # Return shocks
                ra = torch.normal(par.r, par.ra_sigma, size=(par.T-1, train.N, par.Nmc))
                nodes = torch.stack((job_loss, job_find, w, p_eps, ra), dim=-1)
                weights = torch.ones((par.Nmc,), device=train.device, dtype=train.dtype) / par.Nmc # shape (Nmc,)

        return nodes, weights

    #########
    # draw #
    #########

    def draw_initial_states(self, N, training=False):
        """ draw initial state (m,p,t) """

        par = self.par

        # a. assets
        a0 = torch.zeros((N,))
        b0 = par.mu_b0 * np.exp(torch.normal(-0.5*par.sigma_b0**2, par.sigma_b0, size=(N,)))

        # b. wages
        # pareto = torch.distributions.Pareto(scale=torch.tensor(1.0), alpha=torch.tensor(par.eta))
        # p0 = par.p_min * pareto.sample((N,))
        w0 = torch.normal(0.0, 1.0, size=(N,))
        w0 = torch.exp(par.sigma_w*w0-0.5*par.sigma_w**2)

        # c. employment status (everyone starts employed for simplicity)
        employment0 = torch.ones((N,))  # torch.randint(0, 2, (N,))

        # d. human capital
        p0 = torch.normal(-0.5*par.sigma_p0**2, par.sigma_p0, (N,))
        p0 = torch.exp(p0)
        p0 = torch.clamp(p0, min=par.p_min)

        # e. housing (no housing at the beginning)
        h0 = torch.zeros((N,))  # housing
        m0 = torch.zeros((N,))  # mortgage

        return torch.stack((a0, b0, employment0, w0, p0, h0, m0), dim=-1)

    def draw_shocks(self, N, training=False):
        """ draw shocks """

        par = self.par

        # a. taste shocks
        if training:
            dist = torch.distributions.Gumbel(0, par.exp_noise_taste*par.sigma_eps)
        else:
            dist = torch.distributions.Gumbel(0, par.sigma_eps)
        taste_shocks = dist.sample((par.T, N, par.NDC))

        # b. job finding / loss shocks
        job_loss = torch.bernoulli(par.pi_job_loss*torch.ones((par.T, N)))
        job_find = torch.bernoulli(par.pi_job_find*torch.ones((par.T, N)))

        # c. wage shocks
        # pareto = torch.distributions.Pareto(scale=torch.tensor(1.0), alpha=torch.tensor(par.eta))
        # w = par.w_min * pareto.sample((par.T,N,))
        w = torch.normal(0.0, 1.0, size=(par.T, N))
        w = torch.exp(par.sigma_w*w-0.5*par.sigma_w**2)

        # d. human capital shocks
        p_eps = torch.normal(0, 1, (par.T, N))

        # e. return shocks
        ra = torch.normal(par.r, par.ra_sigma, size=(par.T, N))

        return torch.stack((job_loss, job_find, w, p_eps, ra), dim=-1), taste_shocks

    def draw_exploration_shocks(self, epsilon_sigma, N):
        """ draw exploration shockss """

        par = self.par
        eps = torch.zeros((par.T, N, par.Nactions))
        for i_a in range(par.Nactions):
            eps[:, :, i_a] = torch.normal(0, epsilon_sigma[i_a], (par.T, N))

        return eps

    def compute_euler_errors(self,Nbatch_share=0.01):
        """ compute euler error"""

        par   = self.par
        sim   = self.sim
        train = self.train

        dtype  = sim.c.dtype
        device = sim.c.device
        policy_NN = self.policy_NN

        reset = False
        if not train.use_quad:
            reset = True
            Nnumint = train.Nnumint
            numint_nodes = train.numint_nodes
            numint_weights = train.numint_weights
            train.use_quad = True
            self._numerical_integration()

        if self.value_NN is not None:
            value_NN = self.value_NN
        else:
            value_NN = self.value_NNs

        # --- guard: ensure at least one per batch
        Nbatch = max(1, int(Nbatch_share * sim.N))

        # per-DC block width in outcomes: [c, d_1..d_D, (typically a)]
        #block_w = 2 + par.D  # matches your current layout (see indices below)
        
        for i in range(0, sim.N, Nbatch):
            idx0, idx1 = i, min(i + Nbatch, sim.N)

            with torch.no_grad():
                # i) unpack current-period stuff (t = 0..T-2)
                states_pd    = sim.states_pd[:par.T-1, idx0:idx1]
                euler_error  = sim.euler_error[:par.T-1, idx0:idx1]
                c            = sim.c[:par.T-1,  idx0:idx1]
                h_tilde      = sim.h_tilde[:par.T-1, idx0:idx1]

                # ii) next states, actions, outcomes, post-decision states
                states_next     = self._state_trans(states_pd)  # shape = (T-1, Nbatch, Numint, Nstates)
                actions_next    = self.eval_policy(policy_NN, states_next, t0=1) # shape = (T-1, Nbatch, Numint, Nactions)
                outcomes_next   = self.outcomes(states_next, actions_next, t0=1) # shape = (T-1, Nbatch, Numint, Noutcomes, NDC)
                states_pd_next  = self.state_trans_pd(states_next, actions_next, outcomes_next, t0=1) # shape = (T-1, Nbatch, Numint, Nstates_pd, NDC)
                states_pd_next = states_pd_next.permute(0,1,2,4,3) # swap last two dimensions, shape = (T-1, Nbatch, Numint, NDC, Nstates_pd)

                # iii) get choice probs from logit scale formula
                reward_next     = self.reward(states_next, actions_next, outcomes_next, t0=1)     # shape = (T-1, Nbatch, Numint, NDC)
                value_pd_next_   = self.eval_value(value_NN, states_pd_next[:-1], t0=1)[...,0] # shape = (T-1, Nbatch, Numint, NDC)
                value_pd_next_terminal   = self.terminal_reward_pd(states_pd_next[-1:]) # shape = (1, Nbatch, Numint, NDC)
                value_pd_next = torch.cat([value_pd_next_,value_pd_next_terminal],dim=0) # shape = (T-1, Nbatch, Numint, NDC)
                value_next = reward_next + par.beta * value_pd_next # shape = (T-1, Nbatch, Numint, NDC)
                choice_prob = torch.softmax(value_next / par.sigma_eps, dim=-1) # shape = (T-1, Nbatch, Numint, NDC)

                # iv) expected marginal utility of c, integrating over shocks and DCs
                e_next = states_next[...,2] # shape = (T-1, Nbatch, Numint)
                h_next = states_next[...,5] # shape = (T-1, Nbatch, Numint)
                m_next = states_next[...,6] # shape = (T-1, Nbatch, Numint)

                # compute c_next for all potential shocks / NDC next period
                c_next = outcomes_next[..., 2, :]  # shape = (T-1, Nbatch, Numint, NDC)
                coh_next = outcomes_next[..., 3, :]
                h_tilde_next = outcomes_next[..., 7, :]

                # get mask for unfeasible dc next period
                mask = model_funcs.get_unfeasible_mask(e_next, coh_next, h_next, m_next, actions_next, par, t=None)

                # compute marginal utility
                mu_next = model_funcs.marg_util_c(c_next, h_tilde_next, par.sigma, par.alpha) # shape = (T-1, Nbatch, Numint, NDC)
                # integrate over shocks
                exp_marg_util = torch.sum(train.numint_weights[None,None,:,None]*mu_next * choice_prob * (~mask), axis = 2) # shape = (T-1, Nbatch, NDC) -> integrate over shocks
                # integrte over choice probabilities
                exp_marg_util_final = torch.sum(exp_marg_util, axis = -1)  # shape = (T-1, Nbatch) -> integrate over NDC

                # vi) Euler error: u_c^{-1}(β R E_t[u_c']) / c_t - 1
                RHS = par.beta * (1+par.rb) * exp_marg_util_final
                euler_error[:] = model_funcs.inverse_marg_util(RHS, h_tilde, par.sigma, par.alpha) / c - 1

        if reset:
            train.use_quad = False
            train.Nnumint = Nnumint
            train.numint_nodes = numint_nodes
            train.numint_weights = numint_weights

    def more_simulation_outcomes(self):
        """ compute more simulation outcomes """

        par = self.par
        sim = self.sim
        

        def get_action(i_DC, t, mask, var, actions, par):
            if var in par.actions_from_iDC[i_DC].keys():
                return actions[t, mask, par.actions_from_iDC[i_DC][var]]
            else:
                return torch.zeros_like(actions[t,mask,0], dtype=actions.dtype, device=actions.device)
            

        sim.a_pd = torch.zeros_like(sim.states[...,0])
        sim.b_pd = torch.zeros_like(sim.states[...,0])
        sim.c = torch.zeros_like(sim.states[...,0])
        sim.coh = torch.zeros_like(sim.states[...,0])
        sim.ell = torch.zeros_like(sim.states[...,0])
        sim.h_pd = torch.zeros_like(sim.states[...,0])
        sim.m_pd = torch.zeros_like(sim.states[...,0])
        sim.h_tilde = torch.zeros_like(sim.states[...,0])

        sim.savings = torch.zeros_like(sim.states[...,0])
        sim.risky_share = torch.zeros_like(sim.states[...,0])
        sim.housing_share = torch.zeros_like(sim.states[...,0])
        sim.leverage = torch.zeros_like(sim.states[...,0])
        sim.constrained = torch.zeros_like(sim.states[...,0])

        for i_DC in range(par.NDC):
            for t in range(par.T):
                i_DC_mask = sim.DC[t] == i_DC

                outcomes_temp = sim.outcomes[t,i_DC_mask]

                sim.savings[t,i_DC_mask] = get_action(i_DC, t, i_DC_mask, 'savings', sim.actions, par)
                sim.risky_share[t,i_DC_mask] = get_action(i_DC, t, i_DC_mask, 'risky_share', sim.actions, par)
                sim.housing_share[t,i_DC_mask] = get_action(i_DC, t, i_DC_mask, 'housing_share', sim.actions, par)
                sim.leverage[t,i_DC_mask] = get_action(i_DC, t, i_DC_mask, 'leverage', sim.actions, par)
                sim.a_pd[t,i_DC_mask] = outcomes_temp[:, 0, i_DC]
                sim.b_pd[t,i_DC_mask] = outcomes_temp[:, 1, i_DC]
                sim.c[t,i_DC_mask] = outcomes_temp[:, 2, i_DC]
                sim.coh[t,i_DC_mask] = outcomes_temp[:, 3, i_DC]
                sim.ell[t,i_DC_mask] = outcomes_temp[:, 4, i_DC]
                sim.h_pd[t,i_DC_mask] = outcomes_temp[:, 5, i_DC]
                sim.m_pd[t,i_DC_mask] = outcomes_temp[:, 6, i_DC]
                sim.h_tilde[t,i_DC_mask] = outcomes_temp[:, 7, i_DC]

        sim.unconstrained = (sim.savings > 1e-3)
        sim.leverage_not_binding = (sim.leverage < 0.999)
            
                                 
    def get_individual_Rs(self, do_print=False):
        """ 
        Compute individual and average rewards as a function of transfers 
        """

        train = self.train 
        par = self.par 
        sim_base = deepcopy(self.sim)
        vals = torch.tensor([1,5,10,20,50,100,500,600,700,800,900,1000], device=train.device, dtype=train.dtype)
        transfer_grid = torch.concatenate((-torch.flip(vals, dims=[0]),torch.zeros(1, device=train.device, dtype=train.dtype),vals))/10_000
        individual_R_transfer = torch.zeros((transfer_grid.size(0),sim_base.N), device=train.device, dtype=train.dtype)
        R_transfer = torch.zeros((transfer_grid.size(0)), device=train.device, dtype=train.dtype)
        discount = par.beta**(torch.arange(par.T, device=train.device, dtype=train.dtype))
        
        for i,transfer in enumerate(transfer_grid):
            if do_print:
                print(i)
            self.sim = deepcopy(sim_base)
            self.sim.states[0,:,1] += transfer / (1+par.rb) # to liquid assets, need to express in coh
            self.simulate_R()

            reward = self.sim.reward.sum(dim=-1)
            
            individual_R_transfer[i] = torch.sum(reward * discount[:,None], axis = 0)
            R_transfer[i] = self.sim.R

        self.sim = deepcopy(sim_base)
        self.sim.R_transfer = R_transfer
        self.sim.individual_R_transfer = individual_R_transfer
        self.sim.transfer_grid = transfer_grid

        with torch.no_grad():
            self.sim.transfer_star = find_transfer_for_target(self.sim.individual_R_transfer.T, transfer_grid, self.sim.R.item())

    def compute_model_moments(self):
        
        # a. unpack
        train = self.train
        sim = self.sim
        info = self.info
        par = self.par
        DC = sim.DC
        device = DC.device

        # b. get indices
        iDC_work = torch.tensor([i for i,choice in enumerate(par.iDC_from_choices) if 'work' in choice], device=device)
        iDC_adj = torch.tensor([i for i,choice in enumerate(par.iDC_from_choices) if 'adjust' in choice], device=device)
        iDC_renter = torch.tensor([i for i,choice in enumerate(par.iDC_from_choices) if 'renter' in choice], device=device)
        iDC_buyer = torch.tensor([i for i,choice in enumerate(par.iDC_from_choices) if 'buyer' in choice], device=device)

        # c. compute moments
        consumption = torch.zeros_like(sim.states[:,:,0])
        savings_rate = torch.zeros_like(sim.states[:,:,0])
        for i_DC in range(par.NDC):
            savings_rate += get_action(i_DC,'savings',sim.actions,par)*(DC==i_DC)
            consumption += sim.outcomes[...,2,i_DC]*(DC==i_DC)

        work = torch.isin(DC,iDC_work).float().mean(axis=-1)
        adj = torch.isin(DC,iDC_adj).float().mean(axis=-1)
        renter = torch.isin(DC,iDC_renter).float().mean(axis=-1)
        buyer = torch.isin(DC,iDC_buyer).float().mean(axis=-1)

        # d. save 
        info[('mean_savings_rate',train.k)] = savings_rate.mean(axis=-1)
        info[('mean_consumption',train.k)] = consumption.mean(axis=-1)
        info[('share_worker',train.k)] = work
        info[('share_adj',train.k)] = adj
        info[('share_renter',train.k)] = renter
        info[('share_buyer',train.k)] = buyer

    ###################
    # model functions #
    ###################

    outcomes = model_funcs.outcomes
    reward = model_funcs.reward
    discount_factor = model_funcs.discount_factor
    terminal_reward_pd = model_funcs.terminal_reward_pd
    state_trans_pd = model_funcs.state_trans_pd
    state_trans = model_funcs.state_trans
    exploration = model_funcs.exploration

def find_transfer_for_target(R_grid, t_grid, R_target, eps=1e-12):
    """
    For each households in the simulation, compute the transfer that would give 
    them a lifetime reward = average lifetime reward
    Requires:
    - R_grid, a grid of reward as a function of a transfer grid, for each agent
    - t_grid, a grid of transfer
    - R_target, an average reward
    """
    N, G = R_grid.shape
    device, dtype = R_grid.device, R_grid.dtype

    # (N,1) target so searchsorted works row-wise
    target = torch.full((N, 1), R_target, device=device, dtype=dtype)

    # first index where R_grid >= target (per row)
    idx_hi = torch.searchsorted(R_grid, target, right=False).squeeze(1)
    # clamp to [1, G-1] so we can form a bracket; this also enables endpoint extrapolation
    idx_hi = idx_hi.clamp(1, G - 1)
    idx_lo = idx_hi - 1

    # gather bracket values
    R_lo = R_grid.gather(1, idx_lo.unsqueeze(1)).squeeze(1)  # (N,)
    R_hi = R_grid.gather(1, idx_hi.unsqueeze(1)).squeeze(1)  # (N,)
    t_lo = t_grid[idx_lo]                                    # (N,)
    t_hi = t_grid[idx_hi]                                    # (N,)

    # linear interpolation within the bracket (safe divide)
    denom = R_hi - R_lo
    w = torch.where(denom.abs() > eps, (R_target - R_lo) / denom, torch.zeros_like(denom))
    t_star = t_lo + w * (t_hi - t_lo)  # (N,)

    return t_star


def get_action(i_DC, var, actions, par):
    if var in par.actions_from_iDC[i_DC].keys():
        return actions[..., par.actions_from_iDC[i_DC][var]]
    else:
        return torch.zeros(actions.shape[:-1], dtype=actions.dtype, device=actions.device)

def get_indices(par):
    """ 
    Compute four indices:
    1. actions_from_iDC: maps i_DC to discrete choice strings (work_DC, portfolio_DC, housing_DC)
    2. iDC_from_actions: maps (work_DC, portfolio_DC, housing_DC) to i_DC
    3. actions_from_iDC: maps i_DC to continuous action indices (savings, risky_share, housing_share, leverage)
    4. i_action_from_actions: maps continuous action indices to (work_DC, portfolio_DC, housing_DC)
    """

    NDC = par.NDC
    choices_from_iDC = []
    iDC_from_choices = {}
    actions_from_iDC = [{} for _ in range(NDC)]
    actions_from_i_actions = []

    Nactions = 0
    i_DC = 0
    i_action = 0 

    for work_DC in ['not work', 'work']:
        for portfolio_DC in ['keep', 'adjust']:
            #for housing_DC in ['renter', 'owner', 'buyer', 'refinance']:
            for housing_DC in ['renter', 'owner', 'buyer']:
                iDC_from_choices[(work_DC, portfolio_DC, housing_DC)] = i_DC

                dict_choice = dict(
                    work=work_DC,
                    portfolio=portfolio_DC,
                    housing=housing_DC
                )

                choices_from_iDC.append(dict_choice)

                actions_from_iDC[i_DC]['savings'] = i_action 
                actions_from_i_actions.append('savings')
                i_action += 1

                if portfolio_DC == 'adjust':
                    actions_from_iDC[i_DC]['risky_share'] = i_action 
                    actions_from_i_actions.append('risky_share')
                    i_action += 1
                
                if housing_DC == 'buyer':
                    actions_from_iDC[i_DC]['housing_share'] = i_action 
                    actions_from_i_actions.append('housing_share')
                    i_action += 1

                    actions_from_iDC[i_DC]['leverage'] = i_action 
                    actions_from_i_actions.append('leverage')
                    i_action += 1
                
                if housing_DC == 'refinance':
                    actions_from_iDC[i_DC]['leverage'] = i_action 
                    actions_from_i_actions.append('leverage')
                    i_action += 1
                
                i_DC += 1
    
    Nactions = i_action

    choices_from_iDC = np.array(choices_from_iDC, dtype=object)

    return choices_from_iDC, iDC_from_choices, actions_from_iDC, actions_from_i_actions, Nactions