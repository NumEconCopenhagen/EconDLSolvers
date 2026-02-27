import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

########
# eval #
########

def eval_NN(model,NN,x,t0=0,t=None):

    par = model.par
    train = model.train

    # a. time indices
    if not t is None:

        # x.shape = (N,Ninputs)

        T = 1
        if train.backward:
            x_ = x.reshape((-1,x.shape[-1])) # x_.shape = (T,N,Ninputs)
        else:
            x_ = x
        N = x_.shape[0]


        time_indices = torch.tensor([t]*N,dtype=torch.long,device=x.device) # shape = (N,) 

    else:

        # x.shape = (T,...,Ninputs)

        x_ = x.reshape((x.shape[0],-1,x.shape[-1])) # x_.shape = (T,N,Ninputs)

        T = x_.shape[0]
        N = x_.shape[1]

        time_indices = torch.arange(t0,t0+T,dtype=torch.long,device=x_.device) # shape = (T,)
        time_indices = time_indices.repeat_interleave(N) # shape = (T*N,)
            
   # b. time input
    if train.time_input_type == 'one_hot':
        time_inputs = F.one_hot(time_indices,par.T) # shape = (T*N,Ninputs_time)            
    elif train.time_input_type == 'embedding':
        time_inputs = NN.time_embedding(time_indices) # shape = (T*N,Ninputs_time)        
    elif train.time_input_type == 'manual':
        time_inputs = model.time_inputs(time_indices) # shape = (T*N,Ninputs_time)
    else:
        raise ValueError(f'time_input_type {train.time_input_type} not available')
        
    # c. input transformation
    if train.input_transformation:
        x_full = model.input_transformation(x,t=t,t0=t0) # x_full.shape = (T,N,Ninputs_all)
    else:
        x_full = x

    # d. combine
    x_full = x_full.reshape((T*N,-1)) # x_full.shape = (T*N,Ninputs_all)
    x_with_time = torch.cat((x_full,time_inputs),dim=-1) # shape = (N,Ninputs_all+Ninputs_time)	

    # e. return
    y = NN(x_with_time)
    return y.reshape(*x.shape[:-1],-1) # shape = (T,...,Noutputs)

def eval_NN_t(model,NN,x,t=None):

    par = model.par
    train = model.train
        
    # a. input transformation
    if train.input_transformation:
        x_full = model.input_transformation(x,t=t) # x_full.shape = (T,N,Ninputs_all)
    else:
        x_full = x

    # b. evaluate network
    y = NN(x_full)
    return y.reshape(*x.shape[:-1],-1) # shape = (T,...,Noutputs)

##########
# policy #
##########

class Policy(nn.Module):
    """ Policy function """
    
    def __init__(self,par,train,backward=False,normal_init=False,manual_initialization=None):

        super(Policy,self).__init__()

        # inputs
        Ninputs = par.Nstates + train.Ninputs_aux 
        if not backward: Ninputs += train.Ninputs_time

        # outputs
        Noutputs = par.Nactions

        if backward:
            Nneurons = train.Nneurons_policy_t
        else:
            Nneurons = train.Nneurons_policy

        # activation functions
        self.intermediate_activation = getattr(F,train.policy_activation_intermediate)

        # time embedding layer
        if train.time_input_type == 'embedding' and not backward:
            self.time_embedding = nn.Embedding(par.T,train.Ninputs_time)

        # standard layers
        self.layers = nn.ModuleList([None]*(Nneurons.size+1))
    
        # input layer
        self.layers[0] = nn.Linear(Ninputs,Nneurons[0])

        # hidden layers
        for i in range(1,len(self.layers)-1):
            self.layers[i] = nn.Linear(Nneurons[i-1],Nneurons[i])
        
        # output layer
        self.layers[-1] = nn.Linear(Nneurons[-1],Noutputs)

        # manual initialization
        if manual_initialization is not None: 
            manual_initialization(self.layers)

        # initialize weights
        if backward and normal_init: self.apply(make_low_variance_init(std=train.NN_init_std))
        
    def forward(self,inputs):
        """ Forward pass"""

        # input layer
        s = self.intermediate_activation(self.layers[0](inputs))
    
        # hidden layers
        for i in range(1,len(self.layers)-1):
            s = self.intermediate_activation(self.layers[i](s))

        # output layer
        return self.layers[-1](s)

def eval_policy(model,NN,states,t0=0,t=None):

    par = model.par
    train = model.train

    # a. raw actions
    actions_raw = eval_NN(model,NN,states,t0,t)

    # b. apply final activation functions
    actions_list = []

    if type(train.policy_activation_final) is list:

        if train.policy_activation_final is None or (not train.policy_activation_final[0] == 'softmax'):

            for i in range(par.Nactions): # apply activation to each action

                # i. find activation function
                if train.policy_activation_final[i] is None:
                    f = None
                elif hasattr(F,train.policy_activation_final[i]):
                    f = getattr(F,train.policy_activation_final[i])
                else:
                    raise ValueError(f'policy_activation_final {train.policy_activation_final[i]} function not available')
                
                # ii. apply activation function
                if f is None:
                    action_i = actions_raw[...,i]
                else:
                    action_i = f(actions_raw[...,i])

                # iii. append
                actions_list.append(action_i[...,None])

        else:

            Nsoftmax = 0
            for i in range(par.Nactions):
                if train.policy_activation_final[i] == 'softmax':
                    Nsoftmax += 1
                else:
                    break

            actions_softmax = F.softmax(actions_raw[...,:Nsoftmax],dim=-1)

            for i in range(par.Nactions): # apply activation to each action

                if i < Nsoftmax:
                    
                    action_i = actions_softmax[...,i]                    

                else:

                    # i. find activation function
                    if train.policy_activation_final[i] is None:
                        f = None
                    elif hasattr(F,train.policy_activation_final[i]):
                        f = getattr(F,train.policy_activation_final[i])
                    else:
                        raise ValueError(f'policy_activation_final {train.policy_activation_final[i]} function not available')
                    
                    # ii. apply activation function
                    if f is None:
                        action_i = actions_raw[...,i]
                    else:
                        action_i = f(actions_raw[...,i])

                # iii. append
                actions_list.append(action_i[...,None])

        actions = torch.cat(actions_list,dim=-1)

    else:
        
        # if policy_activation_final is not a list, apply the same activation to all actions
        f = getattr(F,train.policy_activation_final)
        if train.policy_activation_final == 'softmax':
            actions = f(actions_raw,dim=-1)
        else:
            actions = f(actions_raw)
    
    # c. clamp and return
    actions = actions.clamp(train.min_actions,train.max_actions)

    return actions

def eval_policy_t(model,NN,states,t=None):

    par = model.par
    train = model.train

    # a. raw actions
    if train.use_simult_in_backward:
        actions_simult = eval_NN(model,model.policy_NN,states,t=t)
    else:
        actions_simult = 0.0
    
    actions_t = eval_NN_t(model,NN,states,t=t)

    actions_raw = actions_t + actions_simult

    # b. apply final activation functions
    actions_list = []

    for i in range(par.Nactions): # apply activation to each action

        # i. find activation function
        if train.policy_activation_final[i] is None:
            f = None
        elif hasattr(F,train.policy_activation_final[i]):
            f = getattr(F,train.policy_activation_final[i])
        else:
            raise ValueError(f'policy_activation_final {train.policy_activation_final[i]} function not available')
        
        # ii. apply activation function
        if f is None:
            action_i = actions_raw[...,i]
        else:
            action_i = f(actions_raw[...,i])

        # iii. append
        actions_list.append(action_i[...,None])
    
    actions = torch.cat(actions_list,dim=-1).reshape(actions_raw.shape)

    # c. clamp and return
    actions = actions.clamp(train.min_actions,train.max_actions)

    return actions

#########
# value #
#########

class StateValue(nn.Module):
    """ Value function """

    def __init__(self,par,train,Noutputs=1,backward=False,normal_init=False):
        
        super(StateValue, self).__init__()

        # inputs
        if train.algoname == 'DeepVPD' or train.algoname == 'DeepVPDDC':
            Nstates = par.Nstates_pd
        else:
            Nstates = par.Nstates


        Ninputs = Nstates + train.Ninputs_aux 
        if not backward: Ninputs += train.Ninputs_time

        # activation functions
        self.intermediate_activation = getattr(F,train.value_activation_intermediate)
        
        # time embedding layer
        if train.time_input_type == 'embedding':
            self.time_embedding = nn.Embedding(par.T,train.Ninputs_time)

        # standard layers
        self.layers =  nn.ModuleList([None]*(train.Nneurons_value.size+1))
    
        # input layer
        self.layers[0] = nn.Linear(Ninputs,train.Nneurons_value[0])

        # hidden layers
        for i in range(1,len(self.layers)-1):
            self.layers[i] = nn.Linear(train.Nneurons_value[i-1],train.Nneurons_value[i])

        # output layer
        self.layers[-1] = nn.Linear(train.Nneurons_value[-1],Noutputs)

        if backward and normal_init: self.apply(make_low_variance_init(std=train.NN_init_std))

    def forward(self, state):
        """ Forward pass"""

        # states is shape N x (Nfeatures+T)

        # input layer
        s = self.intermediate_activation(self.layers[0](state))

        # hidden layers
        for i in range(1,len(self.layers)-1):
            s = self.intermediate_activation(self.layers[i](s))

        # output layer
        return self.layers[-1](s)

def eval_value(model,NN,states_pd,t0=0,t=None):

    par = model.par
    train = model.train

    if isinstance(NN,list):
        values = []
        for j in range(train.N_value_NN):
            value_j = eval_NN(model,NN[j],states_pd,t0,t)
            values.append(value_j)
        value_pd = torch.mean(torch.stack(values,dim=0),dim=0)
    else:
        value_pd = eval_NN(model,NN,states_pd,t0,t)
    
    return value_pd

def eval_value_t(model,NN,states_pd,t0=0,t=None):

    par = model.par
    train = model.train

    if train.use_simult_in_backward:
        if train.N_value_NN is None:
            value_simult = eval_NN(model,model.value_NN,states_pd,t=t)
        else:
            values = []
            for j in range(train.N_value_NN):
                value_j = eval_NN(model,model.value_NNs[j],states_pd,t=t)
                values.append(value_j)
            value_simult = torch.mean(torch.stack(values,dim=0),dim=0)
    else:
        value_simult = 0.0
    
    value_t = eval_NN_t(model,NN,states_pd,t=t) # shape stated_pd when DeepVPDDC backward = (Nbatch,Nstates_pd,NDC)
    
    value = value_t + value_simult
    
    return value

###############################
# low variance initialization #
###############################

def make_low_variance_init(std):

    def init_fn(module):
    
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight,mean=0.0,std=std)
            if module.bias is not None:
                nn.init.constant_(module.bias,0.0)

    return init_fn
