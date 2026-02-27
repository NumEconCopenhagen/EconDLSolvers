import numpy as np
import torch


""" 
Helper functions
"""

def u(c, h, ell, sigma, alpha, chi_e):

    bundle = c**alpha * h ** (1 - alpha)

    if sigma == 1.0:
        return torch.log(bundle) - ell * chi_e
    else:
        return bundle ** (1.0 - sigma) / (1.0 - sigma) - ell * chi_e

def marg_util_c(c, h, sigma, alpha):
    bundle = c**alpha * h ** (1 - alpha) 
    return alpha * bundle ** (-sigma)  * c**(alpha-1) * h**(1-alpha)

def inverse_marg_util(mu, h, sigma, alpha):
    # mu = u_c(c,h); assumes mu>0, h>0, alpha in (0,1)
    if abs(sigma - 1.0) < 1e-12:
        return alpha / mu
    expo = 1.0 / (1.0 - alpha * (1.0 - sigma))
    return (alpha * h**((1.0 - alpha) * (1.0 - sigma)) / mu) ** expo

def compute_repayment_owner(par, m, t0=0, t=None):
    if t is None:
        t_vec = torch.arange(t0, par.T, device=m.device, dtype=m.dtype)
        if m.ndim == 2:
            t_vec = t_vec[:, None]
        elif m.ndim == 3:
            t_vec = t_vec[:, None, None]
        repayment_owner = par.r_m * \
            (1+par.r_m)**(par.T-t_vec) / ((1+par.r_m)**(par.T-t_vec)-1)*m
    else:
        repayment_owner = par.r_m * (1+par.r_m)**(par.T-t) / ((1+par.r_m)**(par.T-t)-1)*m

    assert torch.isfinite(repayment_owner).all(
    ), "compute_repayment_owner: non-finite value detected"

    return repayment_owner


def get_action(i_DC, var, actions, par):
    return actions[..., par.actions_from_iDC[i_DC][var]]


def portfolio_block(i_DC, coh, states, actions, par):
    a = states[..., 0]  # risky assets
    savings = get_action(i_DC, 'savings', actions, par)
    fin_wealth_next = coh * savings

    portfolio_DC = par.choices_from_iDC[i_DC]['portfolio']

    if portfolio_DC == 'keep':
        a_pd = a
        b_pd = fin_wealth_next
    elif portfolio_DC == 'adjust':
        risky_share = get_action(i_DC, 'risky_share', actions, par)
        a_pd = fin_wealth_next * risky_share
        b_pd = fin_wealth_next * (1 - risky_share)

    return a_pd, b_pd


def housing_block(i_DC, coh, states, actions, repayment_owner, par):

    savings = get_action(i_DC, 'savings', actions, par)

    h = states[..., 5]  # housing
    m = states[..., 6]  # mortgage

    housing_DC = par.choices_from_iDC[i_DC]['housing']
    coh_refinance = torch.zeros_like(coh)

    if housing_DC == 'renter':
        consumption = coh * (1.0 - savings) * par.alpha
        h_pd = torch.zeros_like(h)  # no housing for renter
        m_pd = torch.zeros_like(h)  # no mortgage for renter
        h_tilde = coh * (1.0 - savings) * (1-par.alpha) / par.rent
    elif housing_DC == 'owner':
        consumption = coh * (1.0 - savings)
        h_pd = h
        m_pd = m * (1 + par.r_m) - repayment_owner
        h_tilde = h
    elif housing_DC == 'buyer':
        housing_share = get_action(i_DC, 'housing_share', actions, par)
        leverage = get_action(i_DC, 'leverage', actions, par)

        # housing expenditure of the buyer
        consumption = coh * (1 - savings) * (1 - housing_share)
        x_h = coh * (1-savings) * housing_share
        downpayment = (1 - leverage * par.llambda)
        h_pd = x_h / (par.p_h * downpayment)
        m_pd = leverage * par.llambda * par.p_h * h_pd
        h_tilde = h_pd
    elif housing_DC == 'refinance':
        raise NotImplementedError('Refinance is not yet implemented!')
        leverage = get_action(i_DC, 'leverage', actions, par)
        """ 
        Note that the definition of leverage here is different from 
        the one when DC = 'buyer'. 
        When buying, lower bound is zero, max bound is leverage constraint depending on housing choice
        When refinancing, lower bound is min repayment to ensure non-negative consumption,
        max bound is leverage constraint depending on current housing
        This should not be an issue since it's a different continuous action predicted by the network.
        """

        min_m = m * (1+par.r_m+par.tau_m) - (coh - 1e-4) # m' = m(1+r+tau) - repayment, max repayment is (coh-eps) in order to ensure non-zero consumption
        min_m = torch.clamp(min_m, min=0.0)  # ensure non-negative mortgage
        max_m = par.llambda * par.p_h * h
        max_m = torch.clamp(max_m, min=min_m)  # impose min(max(m))

        m_pd = min_m + leverage * (max_m - min_m)
        h_pd = h

        inc_refinance = m_pd - m * (1 + par.r_m + par.tau_m)
        coh_refinance = coh + inc_refinance
        assert torch.all(coh_refinance >= 0), "Cash-on-hand after refinancing must be non-negative"
        consumption = coh_refinance * (1 - savings)
        h_tilde = h

    return consumption, h_pd, m_pd, h_tilde, coh_refinance

def compute_coh(inc, work_DC, portfolio_DC, housing_DC, par):

    inc_j_not_work = inc[..., 0]  # income when not working
    inc_j_work = inc[..., 1]  # income when working
    inc_s_keep = inc[..., 2]  # income from keeping savings
    inc_s_adj = inc[..., 3]  # income from adjusting savings
    inc_h_renter = inc[..., 4]  # income when renting
    inc_h_owner = inc[..., 5]  # income when owning
    inc_h_buyer = inc[..., 6]  # income when buying
    inc_h_refinance = inc[..., 7]  # income when buying

    if work_DC == 'work':
        inc_j = inc_j_work
    else:  # not working
        inc_j = inc_j_not_work

    if portfolio_DC == 'keep':
        inc_s = inc_s_keep
    else:  # adjust
        inc_s = inc_s_adj

    if housing_DC == 'renter':
        inc_h = inc_h_renter
    elif housing_DC == 'owner':
        inc_h = inc_h_owner
    elif housing_DC == 'buyer':
        inc_h = inc_h_buyer
    else:  # refinance
        raise NotImplementedError('Refinance is not yet implemented!')
        inc_h = inc_h_refinance

    coh = inc_j + inc_s + inc_h
    if housing_DC == 'renter':
        # renter should always be feasible, but defaulting is costly and repaid in terms of wage too
        # here, we allow the renter to keep x% of its wage income
        coh = torch.clamp(coh, min=inc_j * par.share_income_protected_renter)
        
    return coh

def get_unfeasible_mask(e, coh, h, m, actions, par, t=None):
    """
    Return boolean mask of shape (..., NDC)
    where True means infeasible.
    """
    shape = actions.shape[:-1] + (par.NDC,)
    mask = torch.zeros(shape, dtype=torch.bool, device=e.device)

    for i_DC in range(par.NDC):
        work_DC, portfolio_DC, housing_DC = par.choices_from_iDC[i_DC].values()

        m_i = torch.zeros(shape[:-1], dtype=torch.bool, device=e.device)

        if work_DC == 'work':
            m_i |= (e == 0.0)
        if portfolio_DC == 'adjust':
            m_i |= (coh[..., i_DC] <= 0.0)
        if housing_DC == 'owner':
            m_i |= (h == 0.0) | (coh[..., i_DC] < 0.0)
        if housing_DC == 'buyer':
            m_i |= (coh[..., i_DC] <= 0.0)
            if t is None:
                m_i[-1] = True
            elif t == par.T - 1:
                m_i[:] = True
        if housing_DC == 'refinance':
            raise NotImplementedError('Refinance is not yet implemented!')
            m_i |= (h == 0.0) | (coh[..., i_DC] < 0.0) 
            m_i[:] = True
            if t is None:
                m_i[-1] = True
            elif t == par.T - 1:
                m_i[:] = True

        mask[..., i_DC] = m_i

    return mask


""" 
Main model functions
"""


def outcomes(model, states, actions, t0=0, t=None, i_DC = None):
    par = model.par

    # a. unpack states

    # Portfolio block
    a = states[..., 0]  # assets
    b = states[..., 1]  # bonds

    # Job-ladder block
    e = states[..., 2]  # employment opportunity. 
    w = states[..., 3]  # wage
    p = states[..., 4]  # human capital

    # Housing block
    h = states[..., 5]  # housing
    m = states[..., 6]  # mortgage

    # c. compute cash-on-hands for each discrete choice

    # Job-ladder block
    inc_j_not_work = torch.clamp(par.chi_b * p, max=par.b_bar)
    inc_j_work = p * w

    # Portfolio block
    inc_s_keep = b 
    inc_s_adj  = b + a - par.F

    # Housing block
    repayment_owner = compute_repayment_owner(par, m, t0, t)
    inc_h_renter = (par.p_h * (1-par.tau_h) - par.delta_h) * h - (1+par.tau_m+par.r_m) * m
    # NOTE: depreciation term uses delta_h * h instead of delta_h * p_h * h (depreciation is paid in physical units, not price units)
    inc_h_owner = - (par.delta_h * h + repayment_owner)
    inc_h_buyer = (par.p_h * (1-par.tau_h) - par.delta_h) * h - m * (1+par.tau_m+par.r_m)
    # refinancing cost and income are computed later in the housing block
    inc_h_refinance = - par.delta_h * h

    inc = torch.stack((inc_j_not_work, inc_j_work,
                      inc_s_keep, inc_s_adj,
                      inc_h_renter, inc_h_owner, inc_h_buyer, inc_h_refinance), dim=-1)

    # d. compute relevant variables for each discrete choice
    shape = actions.shape[:-1] + (
        par.Noutcomes,
        par.NDC,
        )
    outcomes = torch.zeros(shape, dtype=actions.dtype, device=actions.device)
    
    for i_DC in range(par.NDC):

        work_DC, portfolio_DC, housing_DC = list(par.choices_from_iDC[i_DC].values())
        if housing_DC == 'refinance':
            raise NotImplementedError('Refinance is not yet implemented!')
        coh = compute_coh(inc, work_DC, portfolio_DC, housing_DC, par)

        a_pd, b_pd = portfolio_block(i_DC, coh, states, actions, par)
        consumption, h_pd, m_pd, h_tilde, coh_refinance = housing_block(
            i_DC, coh, states, actions, repayment_owner, par)

        outcomes[..., 0, i_DC] = a_pd  # risky assets next period
        outcomes[..., 1, i_DC] = b_pd  # safe assets next period
        outcomes[..., 2, i_DC] = consumption  # consumption
        outcomes[..., 3, i_DC] = coh_refinance if housing_DC == 'refinance' else coh # cash-on-hand this period
        outcomes[..., 4, i_DC] = 1.0 if par.choices_from_iDC[i_DC]['work'] == 'work' else 0.0 # employment this period
        outcomes[..., 5, i_DC] = h_pd  # housing next period
        outcomes[..., 6, i_DC] = m_pd  # mortgage next period
        outcomes[..., 7, i_DC] = h_tilde  # housing flow consumption

    return outcomes


def reward(model, states, actions, outcomes, t0=0, t=None):
    par = model.par
    penalty = -1e8

    # unpack
    e = states[..., 2]
    h = states[..., 5]
    m = states[..., 6]

    # outcomes already has decision dimension
    c       = outcomes[..., 2, :]   # (..., NDC)
    coh     = outcomes[..., 3, :]
    ell     = outcomes[..., 4, :]
    h_tilde = outcomes[..., 7, :]

    # mask over all decisions
    mask = get_unfeasible_mask(e, coh, h, m, actions, par, t=t)  # (..., NDC)

    # fill with penalty
    reward = torch.full_like(c, penalty)

    # compute utilities only where feasible
    reward[~mask] = u(
        c[~mask],
        h_tilde[~mask],
        ell[~mask],
        par.sigma, par.alpha, par.chi_e
    )

    # check
    has_feasible = (~mask).any(dim=-1)
    assert has_feasible.all(), "At least one action must be feasible for every state."

    return reward



def state_trans_pd(model, states, actions, outcomes, t0=0, t=None):

    par = model.par

    # Unpack states
    w = states[..., 3]  # wage
    p = states[..., 4]  # human capital

    shape = actions.shape[:-1] + (
        par.Nstates_pd,
        par.NDC,
    )

    pd = torch.zeros(shape, dtype=model.train.dtype,
                     device=model.train.device)  # of size T,N,NDC

    for i_DC in range(par.NDC):
        a_pd = outcomes[..., 0, i_DC]  # risky assets next period
        b_pd = outcomes[..., 1, i_DC]  # safe assets next period
        ell = outcomes[..., 4, i_DC]  # employment next period
        h_pd = outcomes[..., 5, i_DC]  # housing next period
        m_pd = outcomes[..., 6, i_DC]  # mortgage next period

        pd[..., 0, i_DC] = a_pd
        pd[..., 1, i_DC] = b_pd
        pd[..., 2, i_DC] = ell
        pd[..., 3, i_DC] = w
        pd[..., 4, i_DC] = p
        pd[..., 5, i_DC] = h_pd
        pd[..., 6, i_DC] = m_pd

    return pd


def state_trans(model, states_pd, shocks, t0=0, t=None):

    train = model.train
    par = model.par

    # Unpack post-decision states
    a_pd = states_pd[..., 0]
    b_pd = states_pd[..., 1]

    ell_pd = states_pd[..., 2]
    w_pd = states_pd[..., 3]
    p_pd = states_pd[..., 4]

    h_pd = states_pd[..., 5]
    m_pd = states_pd[..., 6]

    # Unpack shocks
    job_loss = shocks[..., 0]
    job_find = shocks[..., 1]
    w_shocks = shocks[..., 2]
    p_shocks = shocks[..., 3]
    ra_shocks = shocks[..., 4]

    # Compute next period state-variables

    # risky assets
    a_next = a_pd * (1.0 + ra_shocks)

    # safe assets
    b_next = b_pd * (1.0 + par.rb)

    # employment opportunity next period
    i_e = (ell_pd * (1 - job_loss) + (1 - ell_pd) * job_find) == 1.0
    e_next = torch.zeros_like(
        b_pd, dtype=model.train.dtype, device=model.train.device)
    e_next[i_e] = 1.0

    # wage next period
    # if working and no job loss, keep old wage when w_shocks <= w_pd
    keep_old_wage = (ell_pd == 1) & (job_loss == 0.0) & (w_shocks <= w_pd)
    w_next = torch.where(keep_old_wage, w_pd, w_shocks)

    # Compute human capital next period
    p_next = (
        torch.log(p_pd)
        + ell_pd * par.mu_p_e
        + (1 - ell_pd) * par.mu_p_u
        + par.sigma_p * p_shocks
    )
    
    p_next = torch.clamp(p_next, min=np.log(par.p_min))
    p_next = torch.exp(p_next)

    return torch.stack((a_next, b_next, e_next, w_next, p_next, h_pd, m_pd), dim=-1)


def discount_factor(model, states, t0=0, t=None):
    par = model.par
    beta = par.beta * torch.ones_like(states[..., 0])
    return beta


def exploration(model, states, actions, eps, t=None):
    return actions + eps


def terminal_reward_pd(model, states_pd):
    train = model.train
    device = train.device
    dtype = train.dtype

    value_pd = torch.zeros_like(states_pd[..., 0], dtype=dtype, device=device)
    return value_pd
