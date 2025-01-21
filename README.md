# EconDLSolvers

This package provides **deep learning solvers for finite-horizon models in economics**.

The three main algorithms are:

1. `DeepSimulate`: Uses only simulation, only neural network for policy.
1. `DeepFOC`: Uses first order conditions, only neural network for policy.
1. `DeepVPD`: Uses Bellman equations, both neural network for value and policy.

Replication material for "Deep Learning Algorithmsfor Solving Finite-Horizon Models" (Druedahl and Røpke, 2024) can be produced with this code.

## Installation

The package can be installed  with

```
pip install -e EconDLSolvers\.
```

## Overview

Consider a **single agent problem** over `T` periods indexed by `t`.

1. Agents take some `actions` eachs period.
1. Beginning-of-period `states` determine some `outcomes`, a `reward` (utility) and the post-decision states `states_pd`. 
1. The goal of the agent is to maximize *expected discounted rewards* .
1. The next period states follow from the post-decision states and some stochastic `shocks`. 
1. Agents form expectations using *quadrature*.

To solve a specific model the user must specify:

1. The number of states, actions, outcomes and shocks 
1. Laws of motions for states given actions and outcomes (and shocks)
1. Random draws of intial states and shocks
1. Quadrature nodes and weights (not for `DeepSimulate`)

Solution and simulation of the model is then handled generically. In order to obtain this, many of the functions must be written to handle two different input cases:

1. Case I: Inputs are a sample over many periods as in a *simultanous* solution method. This is denoted by `t=None` and some initial time period `t0`. The final time period is implied by the shape of the input.
1. Case II: Inputs are a sample over a specific period as in a *sequential* simulation. This is denoted by some `t` (and `t0` is then irrelevant).

In the special case of `DeepSimulate` only case II is relevant.

The model has some different name spaces:

1. `.par`: General model parameters.
1. `.sim`: Simulation parameters and simulation data.
1. `.train`: Training parameters and training data.

Random numbers are generated for 3 reasons:

1. Initialization of neural network.
1. Simulation samples.
1. Training samples, including drawing batches from the replay buffer.

The user must specify a seed and all random numbers should be drawn using PyTorch for reproduciability. To perform computations on a GPU all arrays must be specified as `torch.tensor` with a specific `dtype` (typically `float32`) and some `device` (either `cpu` or GPU number i with `cuda:i` or just `i`).

In `DeepFOC` and `DeepVPD` algorithms exploration is due to

1. Drawining initial states and shocks.
1. Random actions.
2. Initial periods with exogenous actions.

`DeepFOC` and `DeepVPD` work in two loops:

1. Outer over *iterations*: In each iteration a training sample is drawn, added to the buffer memory and the a training batch is drawn.
1. Inner over *epochs*: In each epoch the traning batch is used for updating the neural networks.

We use epoch termination such that not too much time is spend optimizing the neural nets for the current training batch.

The basic class setup is:

```
from EconDLSolvers import DLSolverClass
class MyModelClass(DLSolverClass):

    # DLSolverClass(algoname=algonname,device=device)

    #########
    # setup #
    #########

    # note: arrays must be torch.Tensor with dtype and on device

    def setup(self):
        # fill independent .par
        # fill independent .sim 

        par.seed = # seed for random number generation
        par.T =  # number of periods
        par.Nstates =  # number of states
        par.Nstates_pd = # number of post-decision states
        par.Nactions = # number of actions
        par.Noutcomes = # number of outcomes
        par.Nshocks =  # number of shocks
        sim.N = # number of agents

    def allocate(self):
        # fill dependent .par
        # fill dependent .sim        

        sim.states = # simulation states, shape (par.T,sim.N,par.Nstates) (dtype=dtype,device=device)
        sim.states_pd = # simulation states_pd, shape (par.T,sim.N,par.Nstates_pd) (dtype=dtype,device=device)
        sim.shocks = # simulation shocks, shape (par.T,sim.N,par.Nshocks) (dtype=dtype,device=device)
        sim.actions = # simulation actions, shape (par.T,sim.N,par.Nactions) (dtype=dtype,device=device)
        sim.outcomes = # simulation outcomes, shape (par.T,sim.N,par.Noutcomes) (dtype=dtype,device=device)
        sim.reward = # simulation reward, shape (par.T,sim.N) (dtype=dtype,device=device)

    def setup_train(self):
        # fill indpendent in .train

    def allocate_train(self):
        # fill dependent in .train

        train.states = # training sample states, shape (par.T,train.N,par.Nstates) (dtype=dtype,device=device)
        train.states_pd = # training sample states_pd, shape (par.T,train.N,par.Nstates_pd) (dtype=dtype,device=device)
        train.shocks = # training sample shocks, shape (par.T,train.N,par.Nshocks) (dtype=dtype,device=device)
        train.actions = # training sample actions, shape (par.T,train.N,par.Nactions) (dtype=dtype,device=device)
        train.outcomes = # training sample outcomes, shape (par.T,train.N,par.Noutcomes) (dtype=dtype,device=device)
        train.reward = # training sample reward, shape (par.T,train.N) (dtype=dtype,device=device)        

        train.policy_activation_final = [] # policy activation functions, len = par.Nactions
        train.min_actions = torch.tensor([],dtype=dtype,device=device), shape = (par.Nactions,)
        train.max_actions = torch.tensor([],dtype=dtype,device=device), shape = (par.Nactions,)

    ##########
    # shocks #
    ##########

    # note: automatically transformed as dtype and moved to device

    def quad(self): # not in DeepSimulate
        return shock_values, shockweights # shape = (Nquad,par.Nshocks), shape = (Nquad,),
    
    def draw_initial_states(self,N,training=False):
        return initial_states # shape = (N,Nstates)

    def draw_shocks(self,N,training=False):
        return shocks # shape = (T,N,Nshocks)

    def draw_exploration_shocks(self,epsilon_sigma,N):
        # epsilon_sigma.shape = (Nactions,)
        return eps # shape = (T,N,Nactions)

    def draw_exo_actions(self,N):
        return exo_actions # shape = (T,N,Nactions)
    
    ###############
    # model funcs #
    ###############

	# Case I, solving simultanously across periods: t is None -> t in t0,...,t0+T-1 <= par.T-1:
	# Case II, simulating sequentially: t in 0,...,par.T-1, t0 irrelevant:

    def outcomes(model,states,actions,t0=0,t=None):
        return outcomes # I: shape = (T,...,Noutcomes), II: shape = (N,Noutcomes)

    def reward(model,states,actions,outcomes,t0=0,t=None):
        return u # I: shape = (T,...), # II: shape = (N,)

    def discount_factor(model,states,t0=0,t=None):
        return beta # I: shape (T,...), II: shape (N,)

    # only if train.terminal_actions_known = True
    def terminal_actions(model,states):
        return actions # I: shape = (1,...,Nactions), II: shape = (1,...,Nactions)

    def terminal_reward_pd(model,states_pd):
        return value_pd # I: shapes = (1,...,1), II: shapes = (N,1)

    def state_trans_pd(model,states,actions,t0=0,t=None):
        return states_pd # I: shape = (T,...,Nstates_pd), II: shape = (N,Nstates_pd)

    def state_trans(model,states_pd,quad,=None):
        return states_plus # I: shape = (T,N,Nquad,Nstates), II: shape = (N,Nstates)

    # not in DeepSimulate
    def exploration(model,states,actions,eps,t=None):
        return actions + eps # I: shape = (T,...,Nactions), II: shape = (N,Nactions)

    # only DeepFOC
    def eval_equations_FOC(model,states,states_plus,actions,actions_plus):
        return squared_equations # shape = (T,N)

mymodel = MyModelClass(algoname=algonname,device=device,dtype=torch.float32,par={},sim={},train={})
mymodel.solve(do_print=True,do_print_all=True)
```

At initialization the following happens:

1. `train.algoname=algoname`, `device=device` and `train.detype=detype` are set.
1. `setup()` and `setup_train()` are called (internal algorithm-specific defaults are set beforehand).
1. Values in `.par` and `.train` are updated with call-specific inputs dicts
1. `allocate()`, `allocate_train()` and `quad()` are called. 
1. Seed in random number generator in `torch` is set with `par.seed`
1. Algorithm and neural nets  are created and intialized
1. Initial states and shocks in `.sim` are drawn

The model can then be solved with the `.solve()` method.

## Manuel termination

When `.solve()` is called a file called `solving.json` is produced. If the `terminate` value is changed to `true` the solution terminated.

By default the model has `train.transfer_grid`. If the model has an `.add_transfer()` method then the convergence approach from Druedahl and Røpke (2024) can be used. By default `.solve` produces `convergence.txt`, which contains a series of the times for the best $R$'s so far, together with the required transfer starting from the current best $R$. If `train.convergence_plot=True` a plot is produced as `convergence.png` instead.

The `.solve()` method take a `postfix` argument if the filenames should be appended.

After termination various timing and additional results are saved in the dictionary `.info`.

## Random numbers

Two different solutions can be generated with:

```
mymodel = MyModelClass(algoname=algonname,device=device)
mymodel.solve()

torch_rng_state = torch.get_rng_state()
mymodel_alt = MyModelClass(algoname=algonname,device=device,torch_rng_state=torch_rng_state)
mymodelmymodel_alt.solve(do_print=True,do_print_all=True)
```

## Hyperparameters

A long list of hyperparameter can be chosen in `.train`.

For the **neural network and learning rates**:

1. `Nneurons_policy`: Neurons for policy network, `np.array([Nneurons1,Nneurons2,...])`
1. `policy_activation_intermediate`: Activation functions for policy network, `list(str)`, e.g. relu or tahn
1. `policy_activation_final`: Activation functions for final layer in policy network, `list(str)`, e.g. sigmoid or softplus etc
1. `value_activation_intermediate`: Activation functions for policy network, `list(str)`, e.g. relu or tahn
1. `Nneurons_value`: Neurons for policy network, `np.array([Nneurons1,Nneurons2,...])`
1. `N_value_NN´: Number of value networks (in DeepVPD) 

1. `learning_rate_policy`: Learning rate policy functions
1. `learning_rate_policy_decay`: Decay in learning rate
1. `learning_rate_policy_min`: Minimum learning rate
1. `learning_rate_value`: Learning rate value functions
1. `learning_rate_value_decay`: Decay in learning rate
1. `learning_rate_value_min`: Minimum learning rate

For **training sample**:

1. `N`: Sample size
1. `buffer_memory`: Multiples of sample size in replay buffer
1. `batch_size`: Batch size

For **exploration**:

1. `epsilon_sigma`: Scale of exploration shocks, 'np.array([scale_action1, scale_action2, ...])'
1. `epsilon_sigma_decay`: Decay in scale of exploration shocks
1. `epsilon_min`: Minimum scale of exploration shocks 'np.array([scale_action1, scale_action2, ...])'
1. `do_exo_actions_periods`: Number of periods with exogenous actions

For **smoothing**:

1. `start_train_policy`: start training policy after this number of iterations
1. `tau`: target smoothing coefficient
1. `use_target_policy`: Use target in update policy 'True/False'
1. `use_target_value`: Use target in update value 'True/False'

For **termination of iterations**:

1. `K`: Maximum number of iterations before termination
1. `K_time`: Maximum number of minutes termination
1. `sim_R_freq`: Iterations between each simulation of lifetime reward (`sim.R`)
1. `only_time_termination`: Only termination from time (or maximum number of iterations) 'True/False'

1. ´convergence_plot`: Print convergence plot in convergence.png
1. ´transfer_grid`:Grid for calculation of transfer
1. ´Delta_transfer`: Transfer tolerance for improvement
1. ´Delta_time`: Time tolerance for improvement
1. ´K_time_min`: Minimum number of minutes before termination

1. `terminate_on_policy_loss`: Terminate if policy loss is below tolerance 'True/False'
1. `tol_policy_loss`: Tolerance for policy loss

For **termination of epochs**:

1. `epoch_termination`: terminate epochs early if no improvement 'True/False'
1. `Nepochs_policy`: number of epochs in update to update policy
1. `Delta_epoch_policy`: number of epochs without improvement before termination
1. `epoch_policy_min`: minimum number of epochs
1. `Nepochs_value`: number of epochs in update to update value
1. `Delta_epoch_value`: number of epochs without improvement before termination
1. `epoch_value_min`: minimum number of epochs

For **clipping gradints:**

1. `clip_grad_policy`: Limit value of gradient to this number
1. `clip_grad_value`: Limit value of gradient to this number

For **scaling**:

1. `use_input_scaling`: True/False
1. `scale_vec_states`: Scaling vector for inputs to neural networks, states
1. `scale_vec_states_pd`: Scaling vector for inputs to neural networks, post-decision states

**Misc.**:

1. `terminal_actions_known`: Boolean for whether terminal actions are known

For `DeepVPD`:

1. `use_FOC`: use analytical FOC,  'True/False'
1. `FOC_weight_pol`: weight on FOC when evaluating policy loss
1. `FOC_weight_val`: weight on FOC when evaluating value loss
1. `value_weight_pol`: weight on value of choice when evaluating policy loss
1. `value_weight_val`: weight on value of choice when evaluating value loss

For `DeepQ`:

1. `DoubleQ`: whether to use double Q learning 'True/False'

## Misc

1. Memory can be tracked with:

    ``` 
    mymodel = MyModelClass(algoname=algonname,device=device,show_memory=True)
    mymodel.solve(do_print=True,show_memory=True)
    ```

2. Solution can be timed with:

    ``` 
    mymodel.time_solve()
    ```

3. Multiple GPUs can be used with:

    ```
    model = MyModelClass(algoname=algoname,device=0)
    model.solve_DPP(do_print=True)
    ```

4. Models can be saved and loaded with:

    ```
    model.save('test.pkl')
    model_load = BufferStockModelClass(device=device,load='test.pkl')
    ```

## Getting started

Look at `0_SimpleConSav\` for a simple example.

## Replication for Druedahl and Røpke (2024)

The results for the buffer-stock model are reproduced as:

1. Run `BufferStock\generate_DP_files.py`
1. Run `BufferStock\generate_DL_files.py`
1. Run `BufferStock\04_Results.ipynb`

The results for the model with multiple durable goods and convex adjustment costs are reproduced as:

1. Run `Durables\generate_DP_files.py`
1. Run `Durables\generate_DL_files.py`
1. Run `Durables\04_Results.ipynb`

The results for the model with multiple durable goods and non-convex adjustment costs are reproduced as:

1. Run `NonConvexDurables\01_Run_DP.ipynb`
1. Run `NonConvexDurables\02_Run_DL.ipynb`

All output is saved in `output/`.