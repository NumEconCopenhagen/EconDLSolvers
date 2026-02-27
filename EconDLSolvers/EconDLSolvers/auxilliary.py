import time
from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F

from . import neural_nets

############################
# random number generation #
############################

def torch_uniform(a,b,size):
    """ uniform random numbers in [a,b] """

    return a + (b-a)*torch.rand(size)

##########
# create #
##########

def create_value_NN(model,value_target=True,Noutputs_value=1):

    # a. unpack
    train = model.train
    par = model.par

    # b. neural net	
    model.value_NN = neural_nets.StateValue(par,train,Noutputs=Noutputs_value).to(train.dtype).to(train.device)		
    if value_target: model.value_NN_target = deepcopy(model.value_NN)

    # c. optimizer	
    model.value_opt = torch.optim.Adam(
        model.value_NN.parameters(), 
        lr=train.learning_rate_value
    )

    # d. scheduler
    model.value_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        model.value_opt,
        gamma=train.learning_rate_value_decay
    )

def create_NN(model,policy=True,value=True,policy_target=True,value_target=True,Noutputs_value=1):
    """ create neural nets """

    # a. unpack
    train = model.train
    par = model.par
    
    # b. policy
    if policy:
        
        # i. neural net
        if train.manual_init_policy:
            assert hasattr(model,'manual_init_policy'), "The model must have the method 'manual_init_policy' for manual initialization of the policy NN"
            manual_initialization = model.manual_init_policy
        else:
            manual_initialization = None

        model.policy_NN = neural_nets.Policy(par,train,manual_initialization=manual_initialization).to(train.dtype).to(train.device)

        if policy_target: model.policy_NN_target = deepcopy(model.policy_NN)

        # ii. optimizer
        model.policy_opt = torch.optim.Adam(
            model.policy_NN.parameters(),
            lr=train.learning_rate_policy
        )

        # iii. scheduler
        model.policy_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            model.policy_opt,
            gamma=train.learning_rate_policy_decay
        )
    
    # c. value
    if value:

        if train.N_value_NN is None:
        
            create_value_NN(model,value_target=value_target,Noutputs_value=Noutputs_value)
        
        else:

            # i. containers
            model.value_NNs = []
            if value_target: model.value_NN_targets = []
            model.value_opts = []
            model.value_schedulers = []

            # ii. create all
            for j in range(train.N_value_NN):
                
                create_value_NN(model,value_target=value_target,Noutputs_value=Noutputs_value)

                model.value_NNs.append(model.value_NN)
                if value_target: model.value_NN_targets.append(model.value_NN_target)
                model.value_opts.append(model.value_opt)
                model.value_schedulers.append(model.value_scheduler)

            # iii. reset
            model.value_NN = None
            if value_target: model.value_NN_target = None
            model.value_opt = None
            model.value_scheduler = None

###########
# NN step #
###########

def NN_step(NN,opt,loss,clip_grad_value=None):
    """ single step of training """
    
    opt.zero_grad()
    loss.backward()
    
    found_bad_grad = False
    for p in NN.parameters():
        if p.grad is not None:
            if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                found_bad_grad = True
                break

    if not found_bad_grad: 
        
        if clip_grad_value is not None: torch.nn.utils.clip_grad_norm_(NN.parameters(),clip_grad_value)
        opt.step()

##########
# policy #
##########

def train_policy(model,policy_loss_f,*args):
    """ update policy network """

    t0 = time.perf_counter()

    # a. unpack
    train = model.train
    policy_NN = model.policy_NN
    policy_opt = model.policy_opt
    info = model.info

    if train.k < train.start_train_policy:
        return
    
    # b. prepare
    best_loss = np.inf
    best_policy_NN = None
    best_optim_state = None
    no_improvement = 0

    # c. loop over epochs
    for epoch in range(train.Nepochs_policy):

        # i. loss and update
        policy_loss = policy_update_step(model,policy_loss_f,*args)
        info[('policy_loss',train.k,epoch)] = policy_loss

        if np.isnan(policy_loss): break

        # ii. save best
        if policy_loss < best_loss:
            no_improvement = 0
            best_loss = policy_loss
            if train.epoch_use_best:
                best_policy_NN = deepcopy(policy_NN.state_dict())
                best_optim_state = deepcopy(policy_opt.state_dict())
        else:
            no_improvement += 1
        
        # iii. terminate
        if train.epoch_termination:
            if no_improvement >= train.Delta_epoch_policy and epoch >= train.epoch_policy_min:
                break

    # c. load best
    if train.epoch_use_best and best_loss < np.inf:
        policy_NN.load_state_dict(best_policy_NN)
        policy_opt.load_state_dict(best_optim_state)

    # d. finalize
    info[('policy_epochs',train.k)] = epoch+1
    info[('policy_loss',train.k)] = best_loss
    info['time.update_NN.train_policy'] += time.perf_counter()-t0

def policy_update_step(model,policy_loss_f,*args):

    # a. unpack
    train = model.train
    policy_NN = model.policy_NN
    policy_opt = model.policy_opt

    # b. loss
    policy_loss = policy_loss_f(model,*args)

    # c. update
    if torch.isnan(policy_loss):
        return np.nan
    else:
        NN_step_policy(train,policy_NN,policy_opt,policy_loss)
        return policy_loss.item()

def NN_step_policy(train,NN,opt,loss):
    """ single step of training - for timing"""

    NN_step(NN,opt,loss,train.clip_grad_policy)
    if train.allow_synchronize and torch.cuda.is_available(): torch.cuda.synchronize(device=train.device)

#########
# value #
#########

def train_value_(model,value_loss_f,*args,info_details=True):

    # a. unpack
    train = model.train
    value_NN = model.value_NN
    value_opt = model.value_opt
    info = model.info

    # b. compute target
    target = compute_target(model,*args)

    # c. prepare
    best_loss = np.inf
    best_value_NN = None
    best_optim_state = None
    no_improvement = 0

    # d. loop over epochs
    for epoch in range(train.Nepochs_value):

        # i. loss
        value_loss = value_loss_f(model,target,*args)

        # ii. update
        NN_step_value(train,value_NN,value_opt,value_loss)

        # iii. save best
        value_loss = value_loss.item()
        if value_loss < best_loss:
            no_improvement = 0 # reset no improvement
            best_loss = value_loss
            if train.epoch_use_best:
                best_value_NN = deepcopy(value_NN.state_dict())
                best_optim_state = deepcopy(value_opt.state_dict())
        else:
            no_improvement += 1 # increase no improvement
                
        # iv. terminate
        if train.epoch_termination:
            if no_improvement >= train.Delta_epoch_value and epoch >= train.epoch_value_min:
                break

        if info_details: info[('value_loss',train.k,epoch)] = value_loss
    
    # d. load best
    if train.epoch_use_best and best_loss < np.inf:
        value_NN.load_state_dict(best_value_NN)
        value_opt.load_state_dict(best_optim_state)

    # e. finalize
    info[('value_epochs',train.k)] = epoch+1
    info[('value_loss')] = best_loss

def train_value(model,value_loss_f,*args):
    """ update value network """

    t0 = time.perf_counter()
    train = model.train
    info = model.info

    if train.N_value_NN is None:
        
        train_value_(model,value_loss_f,*args)
    
    else:

        # a. initialize
        epochs = 0
        best_loss = -np.inf

        # b. loop
        for j in range(train.N_value_NN):

            # i. set
            model.value_NN = model.value_NNs[j]
            if train.use_target_value: model.value_NN_target = model.value_NN_targets[j]
            model.value_opt = model.value_opts[j]
            model.value_scheduler = model.value_schedulers[j]

            # ii. train
            train_value_(model,value_loss_f,*args,info_details=False)

            # iii. update
            epochs += info[('value_epochs',train.k)]
            best_loss = max(best_loss,info[('value_loss')])

            # iv. reset
            model.value_NN = None
            if train.use_target_value: model.value_NN_target = None
            model.value_opt = None
            model.value_scheduler = None			

        # c. save
        info[('value_epochs',train.k)] = epochs
        info[('value_loss',train.k)] = best_loss

    info['time.update_NN.train_value'] += time.perf_counter()-t0

def value_loss_f(model,target,*args):
    """ value loss """

    # a. unpack
    train = model.train
    value_NN = model.value_NN

    value_pred = model.eval_value(value_NN,*args)
    value_loss = F.mse_loss(value_pred[...,0].reshape(-1,1),target)

    if train.allow_synchronize and torch.cuda.is_available(): torch.cuda.synchronize(device=train.device)
    return value_loss

def NN_step_value(train,NN,opt,loss):
    """ single step of training - for timing"""

    NN_step(NN,opt,loss,train.clip_grad_value)
    if train.allow_synchronize and torch.cuda.is_available(): torch.cuda.synchronize(device=train.device)

##################
# compute target #
##################

def choice_prob(par,v0):
    """ compute choice probabilities """

    choice_probs = torch.zeros_like(v0)
    vmax = torch.max(v0,dim=-1,keepdim=True)[0]
    v_vmax = (v0-vmax)/par.sigma_eps
    for d in range(par.NDC):
        choice_probs[...,d] = torch.exp(v_vmax[...,d])/torch.sum(torch.exp(v_vmax),dim=-1)
    
    return choice_probs

def compute_target(model,states_pd):
    """ compute target """

    # states_pd.shape = (T,N,Nstates_pd) = (par.T-1,N,Nstates_pd)
    
    # a. unpack
    par = model.par
    train = model.train

    value_NN = model.value_NN_target if train.use_target_value else model.value_NN
    policy_NN = model.policy_NN_target if train.use_target_policy else model.policy_NN

    # b. check batch
    assert train.N % train.N_target_batches == 0, "N must be divisable by batch size"
    Nb = int(train.N / train.N_target_batches)

    # c. compute target
    with torch.no_grad():

        target_value_pd = torch.zeros((par.T-1,train.N),dtype=train.dtype,device=train.device) # shape (T-1,N)
        if train.use_FOC: target_q_pd = torch.zeros((par.T-1,train.N,train.NFOC_targets),dtype=train.dtype,device=train.device) # shape (T-1,N,NFOCs)
  
        # loop over batches
        for batch in range(train.N_target_batches):

            # i. unpack batch on household indices: Nb
            states_pd_b = states_pd[:,batch*Nb:(batch+1)*Nb,:] # shape = (T-1,Nb,Nstates_pd)
            
            # ii. fetch weights for numerical integration
            if train.update_numint_weights:

                numint_weights = model.numint_weights_update(states_pd_b)
            
            else:

                if train.use_quad:
                    numint_weights = train.numint_weights[None,:] # shape = (1,Nnumint)
                else:
                    numint_weights = train.numint_weights # (for e.g. Monte Carlo) shape = (T,Nb,Nnumint)

            # iii. future states
            states_plus_b = model._state_trans(states_pd_b) # shape = (T-1,Nb,Nnumint,Nstates)

            # iv. future actions
            if train.terminal_actions_known: # handle terminal actions if known
                actions_plus_b_ = model.eval_policy(policy_NN,states_plus_b[:-1],t0=1)		
                actions_plus_terminal_b = model.terminal_actions(states_plus_b[-1:])
                actions_plus_b = torch.cat([actions_plus_b_,actions_plus_terminal_b],dim=0)
            else:
                actions_plus_b = model.eval_policy(policy_NN,states_plus_b,t0=1) # shape = (T-1,Nb,Nnumint,Nactions)

            # v. future reward
            outcomes_plus_b = model.outcomes(states_plus_b,actions_plus_b,t0=1) # shape = (T-1,Nb,Nnumint,Noutcomes)
            reward_plus_b = model.reward(states_plus_b,actions_plus_b,outcomes_plus_b,t0=1) # shape = (T-1,Nb,Nnumint)

            # vi. future post decision states
            states_pd_plus_b = model.state_trans_pd(states_plus_b,actions_plus_b,outcomes_plus_b,t0=1) # shape = (T-1,Nb,Nnumint,Nstates_pd)
            if train.algoname == 'DeepVPDDC': 
                states_pd_plus_b = states_pd_plus_b.permute(0,1,2,4,3) # swap last two dimensions, shape = (T-1,Nb,Nnumint,NDC,Nstates_pd)

            # vi. future post-decision value
            valueq_pd_plus_b = model.eval_value(value_NN,states_pd_plus_b[:-1],t0=1)
            
            value_pd_plus_b_= valueq_pd_plus_b[...,0] # shape = (T-2,Nb,Nnumint,Nstates_pd)
            value_pd_plus_terminal_b = model.terminal_reward_pd(states_pd_plus_b[-1:]) # shape = (1,Nb,Nnumint,Nstates_pd)
            value_pd_plus_b = torch.cat([value_pd_plus_b_,value_pd_plus_terminal_b],dim=0) # shape = (T-1,Nb,Nnumint,Nstates_pd)

            # vii. future value before expectation
            discount_factor_plus_b = model.discount_factor(states_pd_plus_b,t0=1) # shape = (T-1,Nb,Nnumint,Nstates_pd)
            value_plus_b = reward_plus_b + discount_factor_plus_b*value_pd_plus_b # shape = (T-1,Nb,Nnumint,Nstates_pd)

            # viii. expected future value (TODO: DeepV?)
            if train.algoname == 'DeepVPD': 
                target_value_pd_b = torch.sum(train.numint_weights[None,None,:]*value_plus_b,dim=2) # shape = (T-1,Nb,Nstates_pd)
            elif train.algoname == 'DeepVPDDC':
                target_value_pd_numint_b = par.sigma_eps*torch.logsumexp(value_plus_b/par.sigma_eps,dim=-1,keepdim=False) # sum over discrete choice dimension -> shape = (T,Nb,Nnumint)
                target_value_pd_b = torch.sum(numint_weights*target_value_pd_numint_b,dim=-1,keepdim=False)
            else:
                raise NotImplementedError(f'Algorithm {train.algoname} not implemented in compute_target')
            
            # ix. fill out from batch
            target_value_pd[:,batch*Nb:(batch+1)*Nb] = target_value_pd_b

            # x. expected future marginal reward
            if train.use_FOC:

                if train.algoname == 'DeepVPD':

                    q_pd_plus_b_ = valueq_pd_plus_b[...,1:] # shape = (T-2,Nb,Nnumint,NFOC_targets)
                    q_pd_plus_terminal_b = model.terminal_marginal_reward_pd(states_pd_plus_b[-1:]) # shape = (1,Nb,Nnumint,NFOC_targets)
                    q_pd_plus_b = torch.cat([q_pd_plus_b_,q_pd_plus_terminal_b],dim=0) # shape = (T-1,Nb,Nnumint,NFOC_targets)
                    q_plus_b = model.marginal_reward(states_plus_b,actions_plus_b,outcomes_plus_b,discount_factor_plus_b,q_pd_plus_b,t0=1) # shape = (T-1,Nb,Nnumint,NFOC_targets)
                    if q_plus_b.dim() == 3:
                        q_plus_b = q_plus_b[...,None] # shape = (T-1,Nb,Nnumint,1) # add 1 third dim if only one FOC target
                    #element = torch.sum(train.numint_weights[None, None, :, None]*q_plus_b,dim=2,keepdim=True)
                    target_q_pd[:,batch*Nb:(batch+1)*Nb,:] = torch.sum(train.numint_weights[None, None, :, None]*q_plus_b,dim=2,keepdim=False)

                if train.algoname == 'DeepVPDDC':

                    # a. compute q_pd_plus_b
                    # valueq_pd_plus_b shape: (T-2, Nb, Nnumint, NDC, 1+NFOC_targets)
                    q_pd_plus_b_ = valueq_pd_plus_b[..., 1:]  # shape = (T-2, Nb, Nnumint, NDC, NFOC_targets)

                    # states_pd_plus_b[-1:] shape: (1, Nb, Nnumint, NDC, Nstates_pd)
                    q_pd_plus_terminal_b = model.terminal_marginal_reward_pd(states_pd_plus_b[-1:])  # shape = (1, Nb, Nnumint, NDC, NFOC_targets)
                    q_pd_plus_b = torch.cat([q_pd_plus_b_, q_pd_plus_terminal_b], dim=0)  # shape = (T-1, Nb, Nnumint, NDC, NFOC_targets)

                    # Returns shape (T-1, Nb, Nnumint, NDC)
                    q_plus_b = model.marginal_reward(states_plus_b, actions_plus_b, outcomes_plus_b, discount_factor_plus_b, q_pd_plus_b, t0=1)

                    # d. Ensure correct shape - add NFOC_targets dim if missing
                    if q_plus_b.dim() == 4:  # (T-1, Nb, Nnumint, NDC)
                        q_plus_b = q_plus_b[..., None]  # shape = (T-1, Nb, Nnumint, NDC, 1)

                    # e. Compute choice probabilities from continuation values
                    # value_plus_b shape: (T-1, Nb, Nnumint, NDC)
                    choice_probs_b = choice_prob(par, value_plus_b)  # shape = (T-1, Nb, Nnumint, NDC)

                    # f. Weight marginal rewards by choice probabilities
                    # q_plus_b: (T-1, Nb, Nnumint, NDC, NFOC_targets)
                    # choice_probs_b: (T-1, Nb, Nnumint, NDC) -> expand for NFOC_targets
                    choice_probs_expanded = choice_probs_b[..., None]  # shape = (T-1, Nb, Nnumint, NDC, 1)

                    # weighted marginal reward per discrete choice
                    weighted_q = q_plus_b * choice_probs_expanded  # shape = (T-1, Nb, Nnumint, NDC, NFOC_targets)

                    # sum over discrete choices
                    weighted_q_sumDC = torch.sum(weighted_q, dim=-2)  # shape = (T-1, Nb, Nnumint, NFOC_targets)

                    # g. Integrate over shocks (numerical integration)
                    # numint_weights shape: (Nnumint,) -> expand to (1, 1, Nnumint, 1)
                    numint_weights_expanded = train.numint_weights[None, None, :, None]
                    target_q_pd[:, batch*Nb:(batch+1)*Nb, :] = torch.sum(numint_weights_expanded * weighted_q_sumDC, dim=2)  # shape = (T-1, Nb, NFOC_targets)

            else:
                
                target_q_pd = None

        # d. reshape to (T*N,...) as used in other functions
        target_value_pd = target_value_pd.reshape(-1,1) # shape = (T*N,1)
        if train.use_FOC: target_q_pd = target_q_pd.reshape(-1,train.NFOC_targets) # reshape to (T*N,NFOC_targets)

        return target_value_pd, target_q_pd

###################
# target networks #
###################

def evaluate_schedule(model,schedule):

    info = model.info
    train = model.train

    assert train.K_time is not None, 'K_time must not be None'
    assert np.isinf(train.K_time) == False, 'K_time must not be infinite'

    # b. factor
    t_share_now = (time.perf_counter()-info['t0_solve'])/(train.K_time*60.0)
    
    fac = 0.0
    for t_share,fac_ in schedule.items():
        if t_share < t_share_now:			
            fac = fac_
            continue
        else:
            break

    return fac
        
def update_target_value_network(model):
    """ update target value network """

    # a. unpack
    train = model.train
    info = model.info

    # b. variable tau
    if not model.train.tau_schedule is None:

        fac = evaluate_schedule(model,model.train.tau_schedule)	
        tau = (1-fac)*train.tau + fac*train.tau_final
        info[('tau',train.k)] = tau

    else: 

        tau = train.tau

    # c. update
    if train.N_value_NN is None:

        update_target_network(tau,model.value_NN,model.value_NN_target)

    else:
        
        for j in range(train.N_value_NN):
            update_target_network(tau,model.value_NNs[j],model.value_NN_targets[j])

def update_target_policy_network(model):
    """ update target value network """

    update_target_network(model.train.tau,model.policy_NN,model.policy_NN_target)

def update_target_network(adjustment_param,net,target_net):
    """ update target network """

    for target_param,param in zip(target_net.parameters(),net.parameters()):
        target_param.data.copy_(adjustment_param*param.data+(1-adjustment_param)*target_param.data)

#############
# scheduler #
#############

def update_lr(model,opt,schedule):

    train = model.train

    # a. factor
    fac = evaluate_schedule(model,schedule)
    
    # c. learning rate
    lr = (1.0-fac)*train.learning_rate_policy + fac*train.learning_rate_policy_min

    # d. apply
    for param_group in opt.param_groups:

        if isinstance(param_group['lr'], torch.Tensor):
            param_group['lr'].fill_(lr)
        else:
            param_group['lr'] = lr

def scheduler_step(model):
    """ step scheduler """

    train = model.train

    if hasattr(model,'policy_scheduler') and (train.start_train_policy is None or train.k >= train.start_train_policy):

        if train.learning_rate_policy_schedule is None:
            
            if model.policy_scheduler.get_last_lr()[0] > train.learning_rate_policy_min:
                model.policy_scheduler.step()
        
        else:

            update_lr(model,model.policy_opt,train.learning_rate_policy_schedule)

    if hasattr(model,'value_scheduler'):

        if train.N_value_NN is None:

            if train.learning_rate_value_schedule is None:
                
                if model.value_scheduler.get_last_lr()[0] > train.learning_rate_value_min:
                    model.value_scheduler.step()

            else:

                update_lr(model,model.value_opt,train.learning_rate_value_schedule)

        else:
            for j in range(train.N_value_NN):

                if train.learning_rate_value_schedule is None:

                    if model.value_schedulers[j].get_last_lr()[0] > train.learning_rate_value_min:
                        model.value_schedulers[j].step()

                else:

                    update_lr(model,model.value_opts[j],train.learning_rate_value_schedule)

############
# transfer #
############

def compute_transfer(R_transfer,transfer_grid,R,do_extrap=False):

    if R < R_transfer[0]:
        if not do_extrap: return np.nan
        fac = (R_transfer[0]-R) / (R_transfer[1]-R_transfer[0])
        transfer = transfer_grid[0] - fac * (transfer_grid[1]-transfer_grid[0])
    elif R > R_transfer[-1]:
        if not do_extrap: return np.nan
        fac = (R-R_transfer[-1]) / (R_transfer[-1]-R_transfer[-2])
        transfer = transfer_grid[-1] + fac * (transfer_grid[-1]-transfer_grid[-2])
    else:
        transfer = np.interp(R,R_transfer,transfer_grid)

    return transfer

######################################
# default model-specific functions #
######################################

def discount_factor(model,states,t0=0,t=None):
    """ discount factor """

    par = model.par
    beta = par.beta*torch.ones_like(states[...,0])
    return beta 

def terminal_reward_pd(model,states_pd):
    """ terminal reward """

    train = model.train
    h = torch.zeros(*states_pd.shape[:-1],dtype=train.dtype,device=states_pd.device)
    return h

def outcomes(model,states,actions,t0=0,t=None):
    """ default outcomes  - outcomes = actions """

    return actions

def draw_exploration_shocks(self,epsilon_sigma,N):
    """ draw exploration shocks - need to be changed """

    par = self.par

    eps = torch.zeros((par.T,N,par.Nactions))

    for i_action in range(par.Nactions):
        epsilon_sigma_i = epsilon_sigma[i_action]
        eps[:,:,i_action] = torch.normal(0,epsilon_sigma_i,(par.T,N))

    return eps

def exploration(model,states,actions,eps,t=None):

    return actions + eps