import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

########
# eval #
########

def eval_NN(NN,NNT,x,scale_vec=None,t0=0,t=None):

    # a. scale
    if scale_vec is not None: x = x*scale_vec

    # b. add time dummies
    if t is not None:

        # x.shape = (Nx,Ninputs)

        Nx = x.shape[0]

        time_dummies = torch.zeros((Nx,NNT),dtype=x.dtype,device=x.device)
        time_dummies[:,t] = 1
        
        x_time_dummies = torch.cat((x,time_dummies),dim=-1) # shape = (Nx,Ninputs+NNT)	

        return NN(x_time_dummies) # shape = (Nx,Noutputs)
    
    else:
        
        # x.shape = (T,...,Ninputs)

        x_ = x.reshape((x.shape[0],-1,x.shape[-1])) # shape = (T,Nx,Ninputs)

        T = x_.shape[0]
        Nx = x_.shape[1]

        eyeT = torch.eye(NNT,NNT,dtype=x.dtype,device=x.device)[t0:t0+T,:] # shape = (T,NNT)
        eyeT = eyeT.repeat_interleave(Nx,dim=0).reshape((T,Nx,NNT)) # shape = (T,Nx,NNT)

        x_time_dummies = torch.cat((x_,eyeT),dim=-1) # shape = (T,Nx,Ninputs+NNT)
        x_time_dummies = x_time_dummies.reshape((T*Nx,-1)) # shape = (T*Nx,Ninputs+NNT)		

        return NN(x_time_dummies).reshape(*x.shape[:-1],-1) # shape = (T,...,Noutputs)
    
##########
# Policy #
##########

class Policy(nn.Module):
    """ Policy function """
    
    def __init__(self,par,train):

        super(Policy,self).__init__()

        self.Nstates = par.Nstates
        self.T = par.T
        self.Nactions = par.Nactions
        self.intermediate_activation = getattr(F,train.policy_activation_intermediate)

        self.layers = nn.ModuleList([None]*(train.Nneurons_policy.size+1))
    
        # input layer
        self.layers[0] = nn.Linear(self.Nstates+self.T, train.Nneurons_policy[0])

        # hidden layers
        for i in range(1,len(self.layers)-1):
            self.layers[i] = nn.Linear(train.Nneurons_policy[i-1],train.Nneurons_policy[i])
        
        # output layer
        self.layers[-1] = nn.Linear(train.Nneurons_policy[-1],self.Nactions)
        
        if len(train.policy_activation_final) == 1:
            
            if hasattr(F,train.policy_activation_final[0]):
                self.policy_activation_final = getattr(F,train.policy_activation_final[0])
            else:
                raise ValueError(f'policy_activation_final {train.policy_activation_final[0]} function not available')

        else:

            self.policy_activation_final = []

            for i in range(len(train.policy_activation_final)):

                if train.policy_activation_final[i] == None:
                    self.policy_activation_final.append(None)
                else:
                    if hasattr(F,train.policy_activation_final[i]):
                        self.policy_activation_final.append(getattr(F,train.policy_activation_final[i]))
                    else:
                        raise ValueError(f'policy_activation_final {train.policy_activation_final[i]} function not available')

    def forward(self,state):
        """ Forward pass"""

        # input layer
        s = self.intermediate_activation(self.layers[0](state))
    
        # hidden layers
        for i in range(1,len(self.layers)-1):
            s = self.intermediate_activation(self.layers[i](s))

        # output layer
        if type(self.policy_activation_final) is not list:

            action = self.policy_activation_final(self.layers[-1](s))

        else:

            s_noact = self.layers[-1](s) # output layer without activation
            for i_a in range(self.Nactions): # apply activation to each action
                if self.policy_activation_final[i_a] is None:
                    action_i_a = s_noact[:,i_a].view(-1,1)
                else:
                    action_i_a = self.policy_activation_final[i_a](s_noact[:,i_a].view(-1,1))
                if i_a == 0:
                    action = action_i_a
                else:
                    action = torch.cat([action, action_i_a], 1)

        return action

def eval_policy(model,NN,states,t0=0,t=None):

    par = model.par
    train = model.train

    if train.use_input_scaling: 
        scale_vec = model.par.scale_vec_states
    else:
        scale_vec = None

    actions = eval_NN(NN,par.T,states,scale_vec,t0,t)
    actions = actions.clamp(train.min_actions,train.max_actions)

    return actions

#########
# Value #
#########

class StateValue(nn.Module):
    """ Value function """

    def __init__(self,par,train,outputs=1):
        
        super(StateValue, self).__init__()

        if train.algoname == 'DeepVPD':
            self.Nstates = par.Nstates_pd
        else:
            self.Nstates = par.Nstates

        self.T = par.T
        self.intermediate_activation = getattr(F,train.value_activation_intermediate)
        self.outputs = outputs
        
        self.layers =  nn.ModuleList([None]*(train.Nneurons_value.size+1))
    
        # input layer
        self.layers[0] = nn.Linear(self.Nstates+self.T,train.Nneurons_value[0])

        # hidden layers
        for i in range(1,len(self.layers)-1):
            self.layers[i] = nn.Linear(train.Nneurons_value[i-1], train.Nneurons_value[i])

        # output layer
        self.layers[-1] = nn.Linear(train.Nneurons_value[-1], outputs)

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

def eval_value(model,NN,states,t0=0,t=None):

    par = model.par
    train = model.train

    if train.use_input_scaling: 
        scale_vec = model.par.scale_vec_states
    else:
        scale_vec = None

    if isinstance(NN,list):
        values = []
        for j in range(train.N_value_NN):
            value_j = eval_NN(NN[j],par.T,states,scale_vec,t0,t)
            values.append(value_j)
        value = torch.mean(torch.stack(values,dim=0),dim=0)
    else:
        value = eval_NN(NN,par.T,states,scale_vec,t0,t)
    
    return value

def eval_value_pd(model,NN,states_pd,t0=0,t=None):

    par = model.par
    train = model.train

    if train.use_input_scaling: 
        scale_vec = model.par.scale_vec_states_pd
    else:
        scale_vec = None

    if isinstance(NN,list):
        values = []
        for j in range(train.N_value_NN):
            value_j = eval_NN(NN[j],par.T,states_pd,scale_vec,t0,t)
            values.append(value_j)
        value_pd = torch.mean(torch.stack(values,dim=0),dim=0)
    else:
        value_pd = eval_NN(NN,par.T,states_pd,scale_vec,t0,t)
    
    return value_pd

#####
# Q #
#####

class ActionValue(nn.Module):
    """ Value function """

    def __init__(self,par,train):
        
        super(ActionValue, self).__init__()

        self.Nstates = par.Nstates
        self.T = par.T
        self.Nactions = par.Nactions
        self.intermediate_activation = getattr(F,train.value_activation_intermediate)

        self.layers =  nn.ModuleList([None]*(train.Nneurons_value.size+1))
        input_dim = par.Nstates + self.T + par.Nactions

        # input layer
        self.layers[0] = nn.Linear(input_dim, train.Nneurons_value[0])

        # hidden layers
        for i in range(1,len(self.layers)-1):
            self.layers[i] = nn.Linear(train.Nneurons_value[i-1], train.Nneurons_value[i])

        # output layer
        self.layers[-1] = nn.Linear(train.Nneurons_value[-1], 1)

    def forward(self, state, action):
        """ Forward pass"""

        # concatenate state and action
        state_action = torch.cat([state, action], 1)

        # input layer
        s = self.intermediate_activation(self.layers[0](state_action))

        # hidden layers
        for i in range(1,len(self.layers)-1):
            s = self.intermediate_activation(self.layers[i](s))
        
        # output layer
        return self.layers[-1](s)

class StateValueDouble(nn.Module):
    """ Double Q value network"""

    def __init__(self,par,train):

        super(StateValueDouble, self).__init__()

        self.Nstates = par.Nstates
        self.T = par.T
        self.intermediate_activation = getattr(F,train.value_activation_intermediate)

        # layers for both Q1 and Q2 networks
        self.layers =  nn.ModuleList([None]*(train.Nneurons_value.size+1)*2)
        self.network_depth = len(self.layers)//2 # number of layers in each Q network
        
        input_dim = self.Nstates+self.T

        # Q1 architecture

        # input layer
        self.layers[0] = nn.Linear(input_dim, train.Nneurons_value[0])

        # hidden layers
        for i in range(1,self.network_depth-1):
            self.layers[i] = nn.Linear(train.Nneurons_value[i-1], train.Nneurons_value[i])

        # output layer
        self.layers[self.network_depth-1] = nn.Linear(train.Nneurons_value[-1], 1)

        # Q2 architecture

        # input layer
        self.layers[self.network_depth] = nn.Linear(input_dim, train.Nneurons_value[0])

        # hidden layers
        for i in range(self.network_depth+1,len(self.layers)-1):
            Nneurons_index = i - self.network_depth
            self.layers[i] = nn.Linear(train.Nneurons_value[Nneurons_index-1], train.Nneurons_value[Nneurons_index])

        # output layer
        self.layers[-1] = nn.Linear(train.Nneurons_value[-1], 1)		

    def forward(self, state):
        """ Forward pass of both Q1 and Q2"""
        
        # Q1 forward pass

        # input layer
        s = self.intermediate_activation(self.layers[0](state))

        # hidden layers
        for i in range(1,self.network_depth-1):
            s = self.intermediate_activation(self.layers[i](s))

        # output layer
        v1 = self.layers[self.network_depth-1](s)

        # Q2 forward pass

        # input layer
        s = self.intermediate_activation(self.layers[self.network_depth](state))

        # hidden layers
        for i in range(self.network_depth+1,len(self.layers)-1):
            s = self.intermediate_activation(self.layers[i](s))

        # output layer
        v2 = self.layers[-1](s)

        return v1, v2

    def V1(self, state):
        """ Forward pass of V1"""
        
        # V1 forward pass

        # input layer
        s = self.intermediate_activation(self.layers[0](state))

        # hidden layers
        for i in range(1,self.network_depth-1):
            s = self.intermediate_activation(self.layers[i](s))

        # output layer
        v1 = self.layers[self.network_depth-1](s)

        return v1

class ActionValueDouble(nn.Module):
    """ Double Q value network"""

    def __init__(self,par,train):

        super(ActionValueDouble, self).__init__()

        self.Nstates = par.Nstates
        self.T = par.T
        self.Nactions = par.Nactions
        self.intermediate_activation = getattr(F,train.value_activation_intermediate)

        # layers for both Q1 and Q2 networks
        self.layers =  nn.ModuleList([None]*(train.Nneurons_value.size+1)*2)

        self.network_depth = len(self.layers)//2 # number of layers in each Q network
        input_dim = self.Nstates + self.T + self.Nactions

        # Q1 architecture

        # input layer
        self.layers[0] = nn.Linear(input_dim, train.Nneurons_value[0])

        # hidden layers
        for i in range(1,self.network_depth-1):
            self.layers[i] = nn.Linear(train.Nneurons_value[i-1], train.Nneurons_value[i])

        # output layer
        self.layers[self.network_depth-1] = nn.Linear(train.Nneurons_value[-1], 1)

        # Q2 architecture

        # input layer
        self.layers[self.network_depth] = nn.Linear(input_dim, train.Nneurons_value[0])

        # hidden layers
        for i in range(self.network_depth+1,len(self.layers)-1):
            Nneurons_index = i - self.network_depth
            self.layers[i] = nn.Linear(train.Nneurons_value[Nneurons_index-1], train.Nneurons_value[Nneurons_index])

        # output layer
        self.layers[-1] = nn.Linear(train.Nneurons_value[-1], 1)		

    def forward(self, state, action):
        """ Forward pass of both Q1 and Q2"""
        
        # concatenate state and action
        state_action = torch.cat([state, action], 1)

        # Q1 forward pass

        # input layer
        s = self.intermediate_activation(self.layers[0](state_action))

        # hidden layers
        for i in range(1,self.network_depth-1):
            s = self.intermediate_activation(self.layers[i](s))

        # output layer
        q1 = self.layers[self.network_depth-1](s)

        # Q2 forward pass

        # input layer
        s = self.intermediate_activation(self.layers[self.network_depth](state_action))

        # hidden layers
        for i in range(self.network_depth+1,len(self.layers)-1):
            s = self.intermediate_activation(self.layers[i](s))

        # output layer
        q2 = self.layers[-1](s)

        return q1, q2

    def Q1(self, state, action):
        """ Forward pass of Q1"""

        # concatenate state and action
        state_action = torch.cat([state, action], 1)

        # Q1 forward pass

        # input layer
        s = self.intermediate_activation(self.layers[0](state_action))

        # hidden layers
        for i in range(1,self.network_depth-1):
            s = self.intermediate_activation(self.layers[i](s))
            
        # output layer
        q1 = self.layers[self.network_depth-1](s)

        return q1
    
def eval_actionvalue(model,states,action,value_NN,t0=0,Q1=False):

    # states.shape = (T,...,Nstates)

    par = model.par
    train = model.train

    # a. scale
    if train.use_input_scaling: states *= model.par.scale_vec_states

    # b. add time dummies
    states_ = states.reshape((states.shape[0],-1,states.shape[-1])) # shape = (T,N,Nstates)

    T = states_.shape[0]
    N = states_.shape[1]
    NNT = par.T

    eyeT = torch.eye(NNT,NNT,dtype=states_.dtype,device=states_.device)[t0:t0+T,:]  # shape = (T,NNT)
    eyeT = eyeT.repeat_interleave(N,dim=0).reshape((T,N,NNT)) # shape = (T,N,NNT)

    states_time_dummies = torch.cat((states_,eyeT),dim=-1) # shape = (T,N,Nstates+NNT)
    states_time_dummies = states_time_dummies.reshape((T*N,-1)) # shape = (T*N,Nstates+NNT)

    actions_ = action.reshape(-1,action.shape[-1]) # shape = (T*N,Nactions)
    if train.DoubleQ:
        if Q1:
            value = value_NN.Q1(states_time_dummies,actions_) # shape (T*N,1)

            return value.reshape(*states.shape[:-1],1) # shape = (T,...,1)
        else:
            value1, value2 = value_NN(states_time_dummies,actions_)
            return value1.reshape(*states.shape[:-1],1), value2.reshape(*states.shape[:-1],1)
    else:
        value = value_NN(states_time_dummies,actions_) # shape (T*N,1)

        return value.reshape(*states.shape[:-1],1) # shape = (T,...,1)