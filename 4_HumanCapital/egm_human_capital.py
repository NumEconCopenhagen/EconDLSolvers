import numpy as np
import numba as nb

from EconModel import jit
from consav.linear_interp import interp_1d, interp_2d, interp_1d_vec_mon_noprep, interp_3d, interp_4d
# import quantecon as qe
from scipy.optimize import minimize
from consav import golden_section_search

#########
# solve #
#########


@nb.njit
def inverse_marg_util(par, u):
    """Inverse function of marginal utility of consumption """

    return 1/u


@nb.njit
def marg_util_con(par, c):
    """ marginal utility of consumption """

    return 1/c

@nb.njit
def utility(c, ell, vphi, nu):
	""" utility function """

	return np.log(c) - vphi * ell**(1+nu)/(1+nu)

@nb.njit
def wage(hk, alpha):
    return 1+ alpha * hk

@nb.njit
def egm_retired(t, par, dp):

    sol_con = dp.sol_con
    sol_ell = dp.sol_ell
    sol_q = dp.sol_q
    sol_w = dp.sol_w
    a_grid = dp.a_grid
    
    if t == par.T-1:
        for i_a in range(dp.Na):
            c = a_grid[i_a] * par.R + par.y[t]
            sol_con[t,:,:,:,i_a] =c
            sol_ell[t,:,:,:,i_a] = 0 
            sol_q[t,:,:,i_a] = c**-1
            sol_w[t,:,:,i_a] = np.log(c) 

    else: 
        c_temp = (par.beta * par.R * sol_q[t+1,0,0,:])**-1
        a_endo = (c_temp + a_grid - par.y[t]) / par.R
        for i_a in range(dp.Na):
            c = interp_1d(a_endo, c_temp, a_grid[i_a])
            
            # Update consumption function 
            sol_con[t,:,:,:,i_a] = c
            sol_q[t,:,:,i_a] = c**-1
            
            # Update W 
            a_next = a_grid[i_a] * par.R + par.y[t] - c
            sol_w[t,:,:,i_a] = np.log(c) + par.beta * interp_1d(a_grid, sol_w[t+1,0,0,:], a_next)


@nb.njit(parallel = True)
def egm_work(t, par, dp):

    a_grid = dp.a_grid
    hk_grid = dp.hk_grid
    p_grid = dp.p_grid
    ell_grid = dp.ell_grid
    sol_q = dp.sol_q
    sol_w = dp.sol_w

    a_endo = np.zeros((dp.Np, dp.Nhk, par.Npsi, dp.Na, dp.Nhk))
    c_endo = np.zeros((dp.Np, dp.Nhk, par.Npsi, dp.Na, dp.Nhk))

    c_first_stage = np.zeros((dp.Np, dp.Nhk, par.Npsi, dp.Na, dp.Nhk))
    V_first_stage = np.zeros((dp.Np, dp.Nhk, par.Npsi, dp.Na, dp.Nhk))

    for i_hk in nb.prange(dp.Nhk):
        hk = hk_grid[i_hk]
        for i_p in range(dp.Np):
            p = p_grid[i_p]
            for i_psi in range(par.Npsi):
                psi = par.psi[i_psi]
                for i_ell in range(dp.Nell):
                    ell = ell_grid[i_ell]
                    hk_next = hk + ell
                    income = p * par.y[t] * psi * wage(hk, par.alpha) * ell
                    for i_a_next in range(dp.Na):
                        a_next = a_grid[i_a_next]
                        if t == par.T_retired-1:
                            q = sol_q[t+1,i_p,0,i_a_next]
                        else:
                            q = interp_1d(hk_grid, sol_q[t+1,i_p,:,i_a_next], hk_next)
                        c_endo[i_p,i_hk,i_psi,i_a_next,i_ell] = (par.beta * q * par.R)**-1 
                        a_endo[i_p,i_hk,i_psi,i_a_next,i_ell] = (c_endo[i_p,i_hk,i_psi,i_a_next,i_ell] + a_next - income) / par.R

                    for i_a in range(dp.Na):
                        c_first_stage[i_p,i_hk,i_psi,i_a,i_ell] = interp_1d(a_endo[i_p,i_hk,i_psi,:,i_ell], c_endo[i_p,i_hk,i_psi,:,i_ell], a_grid[i_a])
                        a_next = a_grid[i_a] * par.R + income - c_first_stage[i_p,i_hk,i_psi,i_a,i_ell]
                        if a_next < a_grid[0]:
                            a_next = 0 
                            c_first_stage[i_p,i_hk,i_psi,i_a,i_ell] = a_grid[i_a] * par.R + income
                        if c_first_stage[i_p,i_hk,i_psi,i_a,i_ell] == 0:
                            V_first_stage[i_p,i_hk,i_psi,i_a,i_ell] = -1e5
                        else:
                            u = np.log(c_first_stage[i_p,i_hk,i_psi,i_a,i_ell]) + par.vphi * ((1-ell)**(1-par.nu))/(1-par.nu)
                            V_first_stage[i_p,i_hk,i_psi,i_a,i_ell] = u + par.beta * interp_2d(hk_grid, a_grid, sol_w[t+1,i_p,:,:], hk_next, a_next)

    second_stage(t, dp, par, V_first_stage, c_first_stage) 

@nb.njit(parallel = True)
def second_stage(t, dp, par, V_first_stage, c_first_stage):

    ell_grid = dp.ell_grid
    sol_q = dp.sol_q
    sol_w = dp.sol_w
    sol_ell = dp.sol_ell
    sol_con = dp.sol_con

    V_temp = np.zeros((dp.Np, dp.Nhk, par.Npsi, dp.Na))
    Va_temp = np.zeros((dp.Np, dp.Nhk, par.Npsi, dp.Na))

    for i_hk in nb.prange(dp.Nhk):
        for i_p in range(dp.Np):
            for i_psi in range(par.Npsi):
                for i_a in range(dp.Na):        
                    i_ell_opt = np.argmax(V_first_stage[i_p,i_hk,i_psi,i_a,:])
                    ell_opt = ell_grid[i_ell_opt]
                    sol_ell[t,i_p,i_hk,i_psi,i_a] = ell_opt 
                    sol_con[t,i_p,i_hk,i_psi,i_a] =  c_first_stage[i_p,i_hk,i_psi,i_a,i_ell_opt]
                    V_temp[i_p,i_hk,i_psi,i_a] = V_first_stage[i_p,i_hk,i_psi,i_a,i_ell_opt] #
                    Va_temp[i_p,i_hk,i_psi,i_a] = sol_con[t,i_p,i_hk,i_psi,i_a]**-1

    sol_q[t], sol_w[t] = compute_expectation(par, dp, V_temp, Va_temp)

@nb.njit(parallel = True)
def compute_expectation(par, dp, V, Va):

    q = np.zeros((dp.Np, dp.Nhk, dp.Na))
    w = np.zeros((dp.Np, dp.Nhk, dp.Na))

    xi = par.xi

    for i_a_next in nb.prange(dp.Na):
        for i_hk_next in range(dp.Nhk):
            for i_p in range(dp.Np):
                p = dp.p_grid[i_p]
                for i_psi_next in range(par.Npsi):
                    for i_xi_next in range(par.Nxi):
                        p_next = p**(par.rho_xi) * xi[i_xi_next]

                        V_temp = interp_1d(dp.p_grid,V[:,i_hk_next,i_psi_next,i_a_next],p_next)
                        Va_temp = interp_1d(dp.p_grid,Va[:,i_hk_next,i_psi_next,i_a_next],p_next)
                        
                        q[i_p,i_hk_next,i_a_next] += par.psi_w[i_psi_next] * par.xi_w[i_xi_next] * Va_temp
                        w[i_p,i_hk_next,i_a_next] += par.psi_w[i_psi_next] * par.xi_w[i_xi_next] * V_temp

    return q, w


##############
# simulation #
##############


@nb.njit
def policy_interp(sol_con, sol_ell, p_grid, hk_grid, a_grid, psi_grid, N, t, states, con, ell):
    a = states[:,0]
    p = states[:,1]
    hk = states[:,2]
    psi = states[:,3]

    for i in range(N):
        con[i] = interp_4d(p_grid, hk_grid, psi_grid, a_grid, sol_con[t], p[i], hk[i], psi[i], a[i])
        ell[i] = interp_4d(p_grid, hk_grid, psi_grid, a_grid, sol_ell[t], p[i], hk[i], psi[i], a[i])


def simulate(par, dp, sim):
    """ simulate model """

    # a. unpack
    states = sim.states  # shape (T,N,Nstates)
    states_pd = sim.states_pd  # shape (T,N,Nstates_pd)
    outcomes = sim.outcomes  # shape (T,N,Noutcomes)
    shocks = sim.shocks  # shape (T,N,Nshocks)
    reward = sim.reward  # shape (T,N,Nstates)

    # grids and solutions
    a_grid = dp.a_grid
    p_grid = dp.p_grid
    hk_grid = dp.hk_grid
    sol_con = dp.sol_con
    sol_ell = dp.sol_ell

    psi_grid = par.psi
    # states
    b = states[:, :, 0]
    p = states[:, :, 1]
    hk = states[:,:,2]

    b_pd = states_pd[:, :, 0]
    p_pd = states_pd[:, :, 1]
    hk_pd = states_pd[:, :, 2]

    xi = shocks[:, :, 0]
    psi = shocks[:, :, 1]

    states[1:,:,3] = psi[1:,:]
    psi = states[:, :, 3]

    c = outcomes[:, :, 0]
    ell = outcomes[:, :, 1]

    # b. time loop
    for t in range(par.T):

        # a. final period
        if t == par.T-1:
            m = b[t] * par.R + par.y[t]
            c[t] = m 
            ell[t] = 0

        # b. all other periods
        else:
            policy_interp(sol_con, sol_ell, p_grid, hk_grid, a_grid, psi_grid, sim.N, t, states[t], c[t], ell[t])


        if t < par.T_retired:
            income = par.y[t]*p[t]*psi[t]*wage(hk[t],par.alpha)*ell[t]
        else:
            income = par.y[t]

        m = b[t] * par.R + income
        counter = 0
        for i in range(sim.N):
            if c[t,i] > m[i]:
                # print('c > m')
                counter += 1
                c[t,i] = m[i]
                # ell[t,i] = 0
        if counter > 0:
            print('c > m', counter)

        # c. reward
        reward[t] = np.log(c[t]) - par.vphi*(ell[t]**(1+par.nu))/(1+par.nu)

        if t < par.T-1:
            # d. compute m


            m = b[t] * par.R + income

            # e. post-decision states
            p_pd[t] = p[t]
            hk_pd[t] = hk[t] + ell[t]
            b_pd[t] = m - c[t]

            # iii. cash-on-hand
            b[t+1] = b_pd[t]
            hk[t+1] = hk_pd[t]   
            p[t+1] = (p_pd[t]**par.rho_xi)*xi[t+1]
        
