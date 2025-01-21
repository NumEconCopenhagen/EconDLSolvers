import numpy as np
from numba import njit, prange

import quantecon as qe
from consav.linear_interp import interp_2d, interp_3d
from consav.linear_interp import interp_2d_vec, interp_3d_vec

#############
# timestemp #
#############

def solve_timestep(par,egm,t):
    """ solve period t with NEGM """

    # last period
    if t == par.T-1:

        solve_last_period(par,egm)

    # other periods
    else:

        # compute post-decision functions
        compute_wq(par,egm,t)

        # solve keeper problem
        solve_keep(par,egm,t)

        # solve adjuster problem
        solve_adj(par,egm,t)

#####################
# general functions #
#####################

@njit
def utility(c,d,par):
    """ utility """

    cobb_d = c**(par.alpha)*(d+par.d_ubar)**(1-par.alpha)
    u = cobb_d**(1-par.rho) / (1-par.rho)

    return u

@njit
def inverse_marg_util_c(par,q,d):
    """ inverse marginal utility of consumption """

    dtot = d + par.d_ubar
    c_power = par.alpha*(1.0-par.rho) - 1.0
    d_power = (1.0-par.alpha)*(1-par.rho)
    denom = par.alpha*dtot**d_power
    return (q / denom)**(1.0/c_power)

@njit
def marg_u_c(par,c,d):
    
    marg_u = par.alpha*c**(par.alpha*(1-par.rho)-1)*(d+par.d_ubar)**((1-par.alpha)*(1-par.rho))    
    return marg_u

@njit(fastmath=True)
def p_plus_func(p,xi,par,egm):
    """ transition function for p """

    p_plus = p**(par.eta)*xi
    p_plus = np.fmax(p_plus,egm.p_min)
    p_plus = np.fmin(p_plus,egm.p_max)

    return p_plus

@njit(fastmath=True)
def n_plus_func(d,par,egm):
    """ transition function for n """

    n_plus = (1-par.delta)*d
    n_plus = np.fmax(n_plus,egm.n_min)
    n_plus = np.fmin(n_plus,egm.n_max)

    return n_plus

@njit(fastmath=True)
def m_plus_func(a,p_plus,psi,par,egm):
    """ transition function for m """

    income = psi*p_plus
    m_plus = par.R*a + income

    return m_plus

@njit(fastmath=True)
def x_plus_func(m_plus,n_plus,par,egm):
    """ transition function for x """

    x_plus = m_plus + (1-par.kappa)*n_plus

    return x_plus

@njit
def logsumexp(par,v0,v1):
    """ logsumexp """

    vmax = np.maximum(v0,v1)
    logsumexp = vmax + par.sigma_eps*np.log(np.exp((v0-vmax)/par.sigma_eps) + np.exp((v1-vmax)/par.sigma_eps))

    return logsumexp

@njit
def choice_probs(par,v0,v1):
    """ choice probs """

    vmax = np.maximum(v0,v1)
    prob1 = np.exp((v1-vmax)/par.sigma_eps) / (np.exp((v0-vmax)/par.sigma_eps) + np.exp((v1-vmax)/par.sigma_eps))

    return prob1

##################
# upper envelope #
##################

@njit
def upperenvelope(grid_a,m_vec,c_vec,inv_w_vec,grid_m,c_ast_vec,v_ast_vec,n,par):
    """ upperenvelope function """

    # for given m_vec, c_vec and w_vec (coming from grid_a)
    # find the optimal consumption choices (c_ast_vec) at the common grid (grid_m) 
    # using the upper envelope + also value the implied values-of-choice (v_ast_vec)

    Na = grid_a.size
    Nm = grid_m.size

    c_ast_vec[:] = 0
    v_ast_vec[:] = -np.inf

    # constraint
    # the constraint is binding if the common m is smaller
    # than the smallest m implied by EGM step (m_vec[0])

    im = 0
    while im < Nm and grid_m[im] <= m_vec[0]:
        
        # a. consume all
        c_ast_vec[im] = grid_m[im] 

        # b. value of choice
        u = utility(c_ast_vec[im],n,par)

        v_ast_vec[im] = u + par.beta*inv_w_vec[0]

        im += 1

    # apply the upper envelope algorithm
    
    for ia in range(Na-1):

        # a. a inteval and w slope
        a_low  = grid_a[ia]
        a_high = grid_a[ia+1]
        
        inv_w_low  = inv_w_vec[ia]
        inv_w_high = inv_w_vec[ia+1]

        if a_low > a_high:
            continue

        inv_w_slope = (inv_w_high-inv_w_low)/(a_high-a_low)
        
        # b. m inteval and c slope
        m_low  = m_vec[ia]
        m_high = m_vec[ia+1]

        c_low  = c_vec[ia]
        c_high = c_vec[ia+1]

        c_slope = (c_high-c_low)/(m_high-m_low)

        # c. loop through common grid
        for im in range(Nm):

            # i. current m
            m = grid_m[im]

            # ii. interpolate?
            interp = (m >= m_low) and (m <= m_high)            
            extrap_above = ia == Na-2 and m > m_vec[Na-1]

            # iii. interpolation (or extrapolation)
            if interp or extrap_above:

                # o. implied guess
                c_guess = c_low + c_slope*(m - m_low)
                a_guess = m - c_guess

                # oo. implied post-decision value function
                inv_w = inv_w_low + inv_w_slope*(a_guess - a_low)                

                # ooo. value-of-choice
                u = utility(c_guess,n,par)
                v_guess = u + par.beta*inv_w

                # oooo. update
                if v_guess > v_ast_vec[im]:
                    v_ast_vec[im] = v_guess
                    c_ast_vec[im] = c_guess

#################
# post-decision #
#################

@njit(parallel=True)
def compute_wq(par,egm,t):
    """ compute post-decision functions """
    
    # unpack
    w = egm.sol_w[t] # post-decision value
    q = egm.sol_q[t] # post-decision marginal value of cash

    # parallel loop over outermost post-dec state
    for i_p in prange(egm.Np):

        m_plus = np.zeros(egm.Na)
        x_plus = np.zeros(egm.Na)
        v_keep_plus = np.zeros(egm.Na)
        marg_u_c_keep_plus = np.zeros(egm.Na)
        v_adj_plus = np.zeros(egm.Na)
        marg_u_c_adj_plus = np.zeros(egm.Na)

        # loopover other post-decision states
        for i_n in range(egm.Nn):

            # unpack states
            p = egm.p_grid[i_p]
            n = egm.n_grid[i_n]

            # initialize at zero
            w[i_p,i_n,:] = 0.0
            q[i_p,i_n,:] = 0.0

            # loop over shocks
            for i_xi in range(par.Nxi):
                for i_psi in range(par.Npsi):

                    # unpack
                    psi = par.psi[i_psi]
                    xi = par.xi[i_xi]
                    psi_w = par.psi_w[i_psi]
                    xi_w = par.xi_w[i_xi]

                    # compute future  persistent income and durable
                    p_plus = p_plus_func(p,xi,par,egm)
                    n_plus = n_plus_func(n,par,egm) # d=n

                    # quadrature weight
                    weight = psi_w*xi_w

                    # next-period resources:
                    for i_a in range(egm.Na):

                        m_plus[i_a] = m_plus_func(egm.a_grid[i_a],p_plus,psi,par,egm)
                        x_plus[i_a] = x_plus_func(m_plus[i_a],n_plus,par,egm)
                    
                    for i_a in range(egm.Na):

                        # interpolate choice-specific value functions
                        v_keep_plus[i_a] = interp_3d(egm.p_grid,egm.n_grid,egm.a_grid,egm.sol_v_keep[t+1],p_plus,n_plus,m_plus[i_a]) # keeper
                        v_adj_plus[i_a] = interp_2d(egm.p_grid,egm.x_grid,egm.sol_v_adj[t+1],p_plus,x_plus[i_a]) # adjuster

                        # interpolate choice-specific marginal utilities
                        marg_u_c_keep_plus[i_a] = interp_3d(egm.p_grid,egm.n_grid,egm.a_grid,egm.sol_marg_u_c_keep[t+1],p_plus,n_plus,m_plus[i_a])
                        marg_u_c_adj_plus[i_a] = interp_2d(egm.p_grid,egm.x_grid,egm.sol_marg_u_c_adj[t+1],p_plus,x_plus[i_a])

                    # accumulate                    
                    for i_a in range(egm.Na):

                        # post-decision value - handle taste-shocks with logsumexp
                        w[i_p,i_n,i_a] += weight*logsumexp(par,v_keep_plus[i_a],v_adj_plus[i_a])
                        adj_prob = choice_probs(par,v_keep_plus[i_a],v_adj_plus[i_a])
                    
                        # post-decision marginal utility
                        q[i_p,i_n,i_a] += par.beta *par.R *weight *((1-adj_prob) *marg_u_c_keep_plus[i_a] + adj_prob *marg_u_c_adj_plus[i_a])

##################
# keeper problem #
##################

@njit(parallel=True)
def solve_keep(par,egm,t):
    """ solve keeper problem """

    # unpack
    v_keep = egm.sol_v_keep[t]
    marg_u_c_keep = egm.sol_marg_u_c_keep[t]
    c_keep = egm.sol_c_keep[t]

    # unpack post-decision functions
    q = egm.sol_q[t]
    q_c = egm.sol_q_c[t]
    q_m = egm.sol_q_m[t]

    # parallel loop over outermost post-dec state
    for i_p in prange(egm.Np):

        # temporary container
        v_ast_vec = np.zeros(egm.Nm)

        for i_n in range(egm.Nn):

            # use euler
            n = egm.n_grid[i_n]
            for i_a in range(egm.Na):
                q_c[i_p,i_n,i_a] = inverse_marg_util_c(par,q[i_p,i_n,i_a],n) # consumption from euler
                q_m[i_p,i_n,i_a] = egm.a_grid[i_a] + q_c[i_p,i_n,i_a] # endogenous wealth from def of a

            # apply upper envelope
            upperenvelope(egm.a_grid,q_m[i_p,i_n],q_c[i_p,i_n],egm.sol_w[t,i_p,i_n],egm.m_grid,c_keep[i_p,i_n,:],v_ast_vec,n,par)

            # save value of keep
            for i_m in range(egm.Nm):
                v_keep[i_p,i_n,i_m] = v_ast_vec[i_m]
                marg_u_c_keep[i_p,i_n,i_m] = marg_u_c(par,c_keep[i_p,i_n,i_m],n)

####################
# adjuster problem #
####################

@njit
def value_of_choice_adj(d,x,par,egm,i_p,t):
    """ value of choice for adjuster problem """

    # a. cash-on-hand and durables
    m = x - d
    n = d

    # b. consumption and end-of-period assets
    c = interp_2d(egm.n_grid,egm.m_grid,egm.sol_c_keep[t,i_p],n,m) 
    a = m-c

    # b. value-of-choice
    u = utility(c,d,par)
    w_ = interp_2d(egm.n_grid,egm.a_grid,egm.sol_w[t,i_p],n,a) 
    v = u + par.beta*w_

    return v,c

@njit
def obj_adj(choice,x,par,egm,i_p,t):
    """ objective function for adjuster problem """

    v,_ = value_of_choice_adj(choice[0],x,par,egm,i_p,t)

    return v
    

@njit(parallel=True)
def solve_adj(par,egm,t):
    """ solve adjuster problem """

    # unpack
    d_adj = egm.sol_d_adj[t]
    c_adj = egm.sol_c_adj[t]
    v_adj = egm.sol_v_adj[t]
    marg_u_c_adj = egm.sol_marg_u_c_adj[t]

    # parallel loop over outermost post-dec state
    for i_p in prange(egm.Np):

        # loop over x
        for i_x in range(egm.Nx):

            # a. unpack states
            x = egm.x_grid[i_x]

            # b. bounds
            bounds = np.zeros((1,2))
            bounds[0,0] = 0.0
            bounds[0,1] =  np.fmin(np.fmax(x-1e-8,x/2),egm.n_max)

            # c. multi-start optimization
            guess = np.zeros(1)
            Nguess = 2
            guess_vec = np.linspace(bounds[0,0],bounds[0,1],Nguess)
            v_adj[i_p,i_x] = -np.inf

            for j in range(Nguess):
                
                guess[0] = guess_vec[j]           
                results = qe.optimize.nelder_mead(obj_adj,guess,bounds=bounds,args=(x,par,egm,i_p,t))
                d_adj_ = results.x[0]
                v_adj_,c_adj_ = value_of_choice_adj(d_adj_,x,par,egm,i_p,t)

                if v_adj_ > v_adj[i_p,i_x]:                    
                    v_adj[i_p,i_x] = v_adj_
                    d_adj[i_p,i_x] = d_adj_
                    c_adj[i_p,i_x] = c_adj_

            # d. marginal           
            marg_u_c_adj[i_p,i_x] = marg_u_c(par,c_adj[i_p,i_x],d_adj[i_p,i_x])

###############
# last period #
###############

@njit
def value_of_choice_adj_last(d,x,par):
    """ value of choice for adjuster problem - in last period"""

    # a. consumption
    c = x - d

    # b. utility
    u = utility(c,d,par)

    return u,c

@njit
def obj_adj_last(choice,x,par):
    """ objective function for adjuster problem - in last period """

    u,_ = value_of_choice_adj_last(choice[0],x,par)

    return u

@njit(parallel=True)
def solve_last_period(par,egm):

    # unpack keep
    c_keep = egm.sol_c_keep
    v_keep = egm.sol_v_keep
    marg_u_c_keep = egm.sol_marg_u_c_keep

    # unpack adj
    c_adj = egm.sol_c_adj
    d_adj = egm.sol_d_adj
    v_adj = egm.sol_v_adj
    marg_u_c_adj = egm.sol_marg_u_c_adj

    # a. keeper problem
    for i_p in prange(egm.Np):
        for i_n in range(egm.Nn):
            for i_m in range(egm.Nm):

                # i. unpack states
                m = egm.m_grid[i_m]
                p = egm.p_grid[i_p]
                n = egm.n_grid[i_n]

                # ii. consumption
                c_keep[-1,i_p,i_n,i_m] = m

                # iii. value
                v_keep[-1,i_p,i_n,i_m] = utility(c_keep[-1,i_p,i_n,i_m],n,par)
                marg_u_c_keep[-1,i_p,i_n,i_m] = marg_u_c(par,c_keep[-1,i_p,i_n,i_m],n)

    # b. adjuster problem
    for i_p in prange(egm.Np):
        for i_x in range(egm.Nx):

            # i. unpack states
            p = egm.p_grid[i_p]
            x = egm.x_grid[i_x]

            # ii. optimization
            bounds = np.zeros((1,2))
            bounds[0,0] = 0.0
            bounds[0,1] = np.fmin(x-1e-8,egm.n_max)

            if i_x == 0:
                guess = np.array([0.0])
            else:
                guess[0] = d_adj[-1,i_p,i_x-1]

            results = qe.optimize.nelder_mead(obj_adj_last,guess,bounds=bounds,args=(x,par))

            # iii. finalize
            d_adj[-1,i_p,i_x] = results.x[0]
            v_adj[-1,i_p,i_x], c_adj[-1,i_p,i_x] = value_of_choice_adj_last(d_adj[-1,i_p,i_x],x,par)
            marg_u_c_adj[-1,i_p,i_x] = marg_u_c(par,c_adj[-1,i_p,i_x],d_adj[-1,i_p,i_x])

############
# simulate #
############

def simulate(model,ns):
    """ simulate model - when solved with egm """

    # a. unpack
    par = model.par
    egm = model.egm    
    
    states = ns.states
    states_pd = ns.states_pd
    shocks = ns.shocks
    reward = ns.reward
    taste_shocks = ns.taste_shocks

    adj = ns.adj
    c = ns.c
    d = ns.d
    DC = ns.DC

    m = states[:,:,0]
    p = states[:,:,1]
    n = states[:,:,2]

    m_pd = states_pd[:,:,0]
    p_pd = states_pd[:,:,1]
    n_pd = states_pd[:,:,2]

    xi = shocks[:,:,0]
    psi = shocks[:,:,1]

    
    # b. time loop
    for t in range(par.T):

        # i. conditional actions
        x = m[t] + (1-par.kappa) * n[t]

        # keep
        c_keep = np.zeros((ns.N,))
        d_keep = n[t]
        v_keep = np.zeros((ns.N,))

        interp_3d_vec(egm.p_grid,model.egm.n_grid,egm.m_grid,egm.sol_c_keep[t],p[t],n[t],m[t],c_keep)
        interp_3d_vec(egm.p_grid,model.egm.n_grid,egm.m_grid,egm.sol_v_keep[t],p[t],n[t],m[t],v_keep)
            
        # adj
        c_adj = np.zeros((ns.N,))
        d_adj = np.zeros((ns.N,))
        v_adj = np.zeros((ns.N,))

        interp_2d_vec(egm.p_grid,egm.x_grid,egm.sol_c_adj[t],p[t],x,c_adj)
        interp_2d_vec(egm.p_grid,egm.x_grid,egm.sol_d_adj[t],p[t],x,d_adj)
        interp_2d_vec(egm.p_grid,egm.x_grid,egm.sol_v_adj[t],p[t],x,v_adj)

        # ii. values-of-choice
        v_keep_taste = v_keep + taste_shocks[t,:,0]
        v_adj_taste = v_adj + taste_shocks[t,:,1]

        # iii. discrete choice
        I = v_adj_taste > v_keep_taste
        DC[t] = I

        # iv. actions

        adj[t,:] = np.where(I,1.0,0.0)

        c[t,~I] = c_keep[~I]
        d[t,~I] = d_keep[~I]
        
        c[t,I] = c_adj[I]
        d[t,I] = d_adj[I]

        # vi. reward
        cobb_d = c[t]**(par.alpha)*(d[t]+par.d_ubar)**(1-par.alpha)
        u = cobb_d**(1-par.rho) / (1-par.rho)
        
        reward[t,I,0] = 0.0
        reward[t,I,1] = u[I] + taste_shocks[t,I,1]
        
        reward[t,~I,0] = u[~I] + taste_shocks[t,~I,0]
        reward[t,~I,1] = 0.0

        # vii. post-decision states
        m_pd[t,~I] = m[t,~I]-c[t,~I]
        m_pd[t,I] = x[I]-c[t,I]-d[t,I] 
        p_pd[t] = p[t]
        n_pd[t] = d[t]

        # viii. next period
        if t < par.T-1:

            # a. permanent income
            p[t+1] = p_pd[t]**par.eta*xi[t+1]
            
            # b. cash-in-hand
            income = p[t+1]*psi[t+1]
            m[t+1] = par.R*m_pd[t]+income
            
            # c. durables
            n[t+1] = (1-par.delta)*n_pd[t]