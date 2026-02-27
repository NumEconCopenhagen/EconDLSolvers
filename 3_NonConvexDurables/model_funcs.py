import torch
import torch.nn.functional as F

def get_action(i_DC, actions, par, var=None):

    if var is None:

        idxs = list(par.actions_from_iDC[i_DC].values())
        return actions[..., idxs]
    
    else:

        return actions[..., par.actions_from_iDC[i_DC][var]]

###########
# reward #
###########

def u(c,d,par):
    """ utility function """

    # a. non-durable consumption
    c_util = c**(1-torch.sum(par.omega,dim=-1))

    # b. durable consumption for durable 1
    d1_util = (d[...,0]+par.d_ubar[...,0])**par.omega[...,0]

    # c. durable consumption for durable 2
    if par.D == 1:
        d2_util = 1
    elif par.D == 2:
        d2_util = (d[...,1]+par.d_ubar[...,0])**par.omega[...,1]
    else:
        raise ValueError(f'Unsupported number of durables: {par.D}')
    
    u = (c_util * d1_util * d2_util)** (1-par.rho) / (1-par.rho)

    return u

def marg_u_c_func(c,d,par):
    """ marginal utility of consumption """

    # a. non-durable consumption component
    c_comp = c**((1-torch.sum(par.omega,dim=-1)) * (1-par.rho) -1.0)

    # b. durable consumption component for durable 1	
    d1_comp = (d[...,0]+par.d_ubar[...,0])**((par.omega[...,0]) * (1-par.rho))

    # c. durable consumption component for durable 2
    if par.D == 1:
        d2_comp = 1
    elif par.D == 2:
        d2_comp = (d[...,1]+par.d_ubar[...,1])**((par.omega[...,1]) * (1-par.rho))
    else:
        raise ValueError(f'Unsupported number of durables: {par.D}')

    return (1-torch.sum(par.omega)) * c_comp * d1_comp * d2_comp

def inv_marg_u_c_func(mu,d,par):
    """ inv. marginal utility of consumption """

    num = mu
    denom = (1-torch.sum(par.omega, dim=-1)) * torch.prod((d + par.d_ubar)**par.omega, dim=-1)**(1-par.rho)
    exponent = 1/((1-torch.sum(par.omega, dim=-1)) * (1-par.rho)-1)

    return (num / denom)**exponent

def outcomes(model, states, actions, t0=0, t=None):
    
    train = model.train
    if train.budget_shares:
        return outcomes_softmax(model,states,actions,t0=t0,t=t)
    else:
        return outcomes_sigmoid(model,states,actions,t0=t0,t=t)

def outcomes_softmax(model, states, actions, t0=0, t=None):
    """ compute model outcomes for all discrete choices (D=1 or D=2)."""
    
    par = model.par
    D = par.D # number of durables

    # a. unpack states
    m = states[..., 0]  # cash-on-hand
    n1 = states[..., 2] # durable 1 stock
    n2 = states[..., 3] if D == 2 else None  # durable 2 stock if it exists

    results = []  # collect per-i_DC outcomes

    # b. loop over all discrete choices
    for i_DC, choice in enumerate(par.choices_from_iDC):

        # i. fetch actions for this discrete choice
        acts = get_action(i_DC, actions, par)

        # ii. which durables are adjusted
        adjust1 = (choice['adjust1'] == 'adjust')
        adjust2 = (D == 2 and choice.get('adjust2') == 'adjust')

        # iii. effective cash-on-hand after any adjustment
        mbar = m.clone()
        if adjust1: mbar += (1 - par.nu) * n1
        if adjust2 and D == 2: mbar += (1 - par.nu) * n2

        # iv. spending vector: first 2 are always [c_rate, s_rate]
        spending_vector = F.softmax(acts, dim=-1)
        c_rate = spending_vector[...,0]
        s_rate = spending_vector[...,1]

        # v. base outcomes
        c = c_rate * mbar
        m_pd = s_rate * mbar

        if D == 1:
            
            if adjust1:
                d1 = spending_vector[...,2] * mbar
            else:
                d1 = n1

            results.append(torch.stack((c,d1,m_pd),dim=-1))

        else:  # D == 2
            
            # durable 1
            if adjust1:
                d1 = spending_vector[...,2] * mbar
            else:
                d1 = n1

            # Ddrable 2
            if adjust2:
                idx_d2 = 3 if adjust1 else 2
                d2 = spending_vector[...,idx_d2] * mbar
            else:
                d2 = n2

            results.append(torch.stack((c,d1,d2,m_pd),dim=-1))

    # c. flatten all outcomes into [..., par.NDC*Noutcomes_dc] as reward() expects
    return torch.cat(results, dim=-1)

def outcomes_sigmoid(model, states, actions, t0=0, t=None):
    """Compute model outcomes for all discrete choices (D=1 or D=2)."""
    
    par = model.par
    D = par.D  # number of durables

    # a. unpack states ---
    m = states[..., 0]          # cash-on-hand
    n1 = states[..., 2]         # durable 1 stock
    n2 = states[..., 3] if D == 2 else None  # durable 2 stock if it exists

    results = []  # collect per-i_DC outcomes

    # b. loop over all discrete choices ---
    for i_DC, choice in enumerate(par.choices_from_iDC):

        # ii. fetch actions for this discrete choice
        acts = get_action(i_DC, actions, par)

        # iii. which durables are adjusted
        adjust1 = (choice['adjust1'] == 'adjust')
        adjust2 = (D == 2 and choice.get('adjust2') == 'adjust')

        # iv. effective cash-on-hand after any adjustment
        mbar = m.clone()
        if adjust1: mbar += (1 - par.nu) * n1
        if adjust2 and D == 2: mbar += (1 - par.nu) * n2

        if D == 1:

            if adjust1:
                
                c_rate = acts[...,0]
                exp_rate = acts[...,1]
                exp = exp_rate * mbar
                c = c_rate * exp
                d1 = (1-c_rate) * exp
                m_pd = (1-exp_rate) * mbar

            else:

                c_rate = acts[...,0]
                c = c_rate * mbar
                d1 = n1
                m_pd = mbar * (1-c_rate)

            results.append(torch.stack((c, d1, m_pd), dim=-1))

        else: # D == 2

            if (not adjust1 and not adjust2):
                c_rate = acts[...,0]
                c = c_rate * mbar
                d1 = n1
                d2 = n2
                m_pd = mbar * (1-c_rate)				
            elif (adjust1 and not adjust2):
                c_rate = acts[...,0]
                exp_rate = acts[...,1]
                exp = exp_rate * mbar
                c = c_rate * exp
                d1 = (1-c_rate) * exp
                m_pd = (1-exp_rate) * mbar
                d2 = n2
            elif (not adjust1 and adjust2):
                c_rate = acts[...,0]
                exp_rate = acts[...,1]
                exp = exp_rate * mbar
                c = c_rate * exp
                d2 = (1-c_rate) * exp
                m_pd = (1-exp_rate) * mbar
                d1 = n1
            else:
                c_rate = acts[...,0]
                exp_rate = acts[...,1]
                d1_rate = acts[...,2]
                exp = mbar * exp_rate
                m_pd = mbar * (1-exp_rate)
                c = c_rate * exp
                d_exp = (1-c_rate) * exp
                d1 = d_exp * d1_rate
                d2 = d_exp * (1-d1_rate)						

            results.append(torch.stack((c, d1, d2, m_pd), dim=-1))

    # c. flatten all outcomes into [..., par.NDC*Noutcomes_dc] as reward() expects
    return torch.cat(results, dim=-1)

def reward(model, states, actions, outcomes, t0=0, t=None):
    """ compute utility for all discrete choices (D=1 or D=2)."""

    par = model.par
    D = par.D
    Noutcomes_dc = 3 if D == 1 else 4

    rewards = []
    for i_DC in range(par.NDC):
        
        offset = Noutcomes_dc * i_DC
        c = outcomes[..., offset + 0]
        
        d_list = [outcomes[..., offset + 1]]
        if D == 2: d_list.append(outcomes[..., offset + 2])

        d = torch.stack(d_list, dim=-1)

        rewards.append(u(c,d,par))

    return torch.stack(rewards,dim=-1)

def discount_factor(model,states,t0=0,t=None):
    """ discount factor """

    par = model.par

    beta = par.beta*torch.ones_like(states[...,0])
    
    return beta 

def exploration(model,states,actions,eps,t=None):

    return actions + eps

def terminal_reward_pd(model,states_pd):
    """ terminal reward """

    # Case I: states_pd.shape = (1,...,Nstates_pd)
    # Case II: states_pd.shape = (N,Nstates_pd)

    train = model.train
    dtype = train.dtype
    device = train.device

    value_pd = torch.zeros((*states_pd.shape[:-1],),dtype=dtype,device=device)

    return value_pd
    # Case I: shapes = (1,...,)
    # Case II: shapes = (N,)


##############
# transition #
##############

def state_trans_pd(model, states, actions, outcomes, t0=0, t=None):
    """Compute post-decision states for all discrete choices (D=1 or D=2)."""

    par = model.par
    D = par.D

    # a. unpack states
    m = states[..., 0] # cash-on-hand
    p = states[..., 1]  # permanent income
    n1 = states[..., 2] # durable 1 stock
    n2 = states[..., 3] if D == 2 else None

    # b. number of outcome elements per discrete choice
    Noutcomes_dc = 3 if D == 1 else 4

    # c. loop over discrete choices
    pd_list = []
    for i_DC, choice in enumerate(par.choices_from_iDC):

        offset = i_DC * Noutcomes_dc

        # i. extract post-decision cash-on-hand
        m_pd = outcomes[..., offset + (2 if D == 1 else 3)]
        p_pd = p  # permanent income doesn't change

        # ii. durable 1
        adjust1 = (choice['adjust1'] == 'adjust')
        n1_pd = outcomes[..., offset + 1] if adjust1 else n1

        # iii. durable 2 (if exists)
        if D == 2:
            adjust2 = (choice.get('adjust2') == 'adjust')
            n2_pd = outcomes[..., offset + 2] if adjust2 else n2
            pd_choice = torch.stack((m_pd, p_pd, n1_pd, n2_pd), dim=-1)
        else:
            pd_choice = torch.stack((m_pd, p_pd, n1_pd), dim=-1)

        pd_list.append(pd_choice)

    # d. stack over discrete choices
    return torch.stack(pd_list, dim=-1)  # shape (..., Nstates_pd, NDC)

def state_trans(model,states_pd,shocks,t0=0,t=None):
    """ state transition with quadrature """

    par = model.par
    train = model.train

    # a. unpack
    a = states_pd[...,0]
    p = states_pd[...,1]
    d1 = states_pd[...,2]
    d2 = states_pd[...,3] if par.D == 2 else None

    xi = shocks[...,0]
    psi = shocks[...,1]
        
    # b. permanent income state
    p_plus = p**par.eta*xi

    # c. compute income for cash-in-hand
    if t is None:

        kappa = par.kappa[:-1][:,None,None]
        income = kappa * p_plus * psi

    else:

        income = par.kappa[t] * p_plus * psi
    
    # d. cash-on-hand state
    m_plus = par.R*a+income

    # e. durable state
    n1_plus = (1-par.delta[...,0])*d1*torch.ones_like(m_plus)
    if par.D == 1:
        return torch.stack((m_plus,p_plus,n1_plus),dim=-1)
    else:
        n2_plus = (1-par.delta[...,1])*d2*torch.ones_like(m_plus)
        return torch.stack((m_plus,p_plus,n1_plus,n2_plus),dim=-1)

#######
# FOC #
#######

# NOT CLEANED UP YET

def marginal_reward(model, states, actions, outcomes, discount_factor, q_pd_plus, t0=0, t=None):
    """
    Compute marginal reward for DeepVPDDC FOC targeting.

    For the Euler equation: u'(c) = beta * R * E[u'(c')]
    The marginal reward is R * u'(c) for each discrete choice.

    Args:
        states: (..., Nstates)
        actions: (..., Nactions)
        outcomes: (..., NDC * Noutcomes_dc) - flattened outcomes for all DCs
        discount_factor: (...)
        q_pd_plus: (..., NDC, NFOC_targets) or None - marginal value from next period
        t0, t: time indices

    Returns:
        q: (..., NDC) - marginal reward for each discrete choice
    """
    par = model.par
    D = par.D
    Noutcomes_dc = 3 if D == 1 else 4

    q_list = []
    for i_DC in range(par.NDC):
        offset = Noutcomes_dc * i_DC

        # Extract consumption and durables for this discrete choice
        c = outcomes[..., offset + 0]
        d_list = [outcomes[..., offset + 1]]
        if D == 2:
            d_list.append(outcomes[..., offset + 2])
        d = torch.stack(d_list, dim=-1)

        # Marginal utility of consumption: R * u'(c)
        marg_u_c = par.R * marg_u_c_func(c, d, par)
        q_list.append(marg_u_c)

    # Stack over discrete choices: shape = (..., NDC)
    q = torch.stack(q_list, dim=-1)

    return q

def eval_equations_DeepVPDDC(model, states, actions, outcomes, states_pd, q_pd):
    """
    Evaluate FOC equations for DeepVPDDC (non-terminal periods).

    The Euler equation for each discrete choice:
        u'(c) = beta * dV_pd/dm_pd

    Using Fischer-Burmeister for complementary slackness with borrowing constraint.

    Args:
        states: (T, N, Nstates)
        actions: (T, N, Nactions)
        outcomes: (T, N, NDC * Noutcomes_dc)
        states_pd: (T, N, NDC, Nstates_pd)
        q_pd: (T, N, NDC, NFOC_targets) - marginal value of post-decision state

    Returns:
        eq: (T, N, NDC) - squared equation errors per discrete choice
    """
    par = model.par
    train = model.train
    D = par.D
    Noutcomes_dc = 3 if D == 1 else 4

    # Defensive shape assertions
    assert q_pd.shape[-2] == par.NDC, f"q_pd shape[-2]={q_pd.shape[-2]} != NDC={par.NDC}"
    assert q_pd.shape[-1] == train.NFOC_targets, f"q_pd shape[-1]={q_pd.shape[-1]} != NFOC_targets={train.NFOC_targets}"

    eq_list = []
    for i_DC in range(par.NDC):
        offset = Noutcomes_dc * i_DC

        # a. Extract consumption and durables
        c = outcomes[..., offset + 0]
        d_list = [outcomes[..., offset + 1]]
        if D == 2:
            d_list.append(outcomes[..., offset + 2])
        d = torch.stack(d_list, dim=-1)

        # b. Marginal utility of consumption
        marg_u_c = marg_u_c_func(c, d, par)

        # c. Get marginal value of post-decision wealth for this discrete choice
        # q_pd[..., i_DC, 0] is dV_pd/dm_pd
        dvpd_dmpd = q_pd[..., i_DC, 0].clamp(1e-10, 1e10)

        # d. Euler equation: u'(c) / (beta * dV_pd/dm_pd) - 1 = 0
        FOC = marg_u_c / (par.beta * dvpd_dmpd) - 1

        # e. Borrowing constraint: m_pd >= 0
        # Get post-decision wealth for this discrete choice
        m_pd = states_pd[..., i_DC, 0]

        # f. Fischer-Burmeister complementary slackness
        eq_dc = fischer_burmeister(FOC, m_pd) ** 2

        eq_list.append(eq_dc)

    # Stack: shape = (T, N, NDC)
    eq = torch.stack(eq_list, dim=-1)

    return eq

def eval_equations_DeepVPDDC_terminal(model, states, actions, outcomes, states_pd):
    """
    Evaluate FOC equations for DeepVPDDC at terminal period.

    At terminal period, the optimal policy is to consume all wealth (no continuation).
    The FOC simplifies to: m_pd should be zero (or satisfy terminal condition).

    Args:
        states: (1, N, Nstates)
        actions: (1, N, Nactions)
        outcomes: (1, N, NDC * Noutcomes_dc)
        states_pd: (1, N, NDC, Nstates_pd)

    Returns:
        eq: (1, N, NDC) - squared equation errors
    """
    par = model.par
    D = par.D
    Noutcomes_dc = 3 if D == 1 else 4

    eq_list = []
    for i_DC in range(par.NDC):
        # At terminal, optimal is to consume all wealth (no bequest motive)
        # FOC becomes: m_pd = 0 (consume all wealth)
        m_pd = states_pd[..., i_DC, 0]

        # Terminal condition: post-decision wealth should be zero
        eq_dc = m_pd ** 2

        eq_list.append(eq_dc)

    eq = torch.stack(eq_list, dim=-1)

    return eq

def terminal_marginal_reward_pd(model, states_pd):
    """
    Terminal marginal reward for post-decision states with discrete choices.

    At terminal period, returns zeros for marginal values (no bequest motive).

    Args:
        states_pd: Post-decision states with shape (..., NDC, Nstates_pd) or (..., Nstates_pd)

    Returns:
        q_pd_terminal: Zeros with shape (..., NFOC_targets) or (..., NDC, NFOC_targets)
    """
    train = model.train
    dtype = train.dtype
    device = train.device

    # Output shape: replace last dimension (Nstates_pd) with NFOC_targets
    output_shape = (*states_pd.shape[:-1], train.NFOC_targets)
    q_pd_terminal = torch.zeros(output_shape, dtype=dtype, device=device)

    return q_pd_terminal

def fischer_burmeister(a,b):
    """ Fischer-Burmeister function - torch """
     
    return torch.sqrt(a**2 + b**2) - a - b

