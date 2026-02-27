# EconDLSolvers

This package provides **deep learning solvers for finite-horizon models in economics**.

The three main algorithms are:

1. `DeepSimulate`: Uses only simulation, only neural network for policy.
1. `DeepFOC`: Uses first order conditions, only neural network for policy.
1. `DeepVPDDC`: Uses Bellman equations, both neural network for value and policy, and allow for discrete choices.

A variant of `DeepVPDDC` is called and does not allow for discrete choices.

The standard approach is to solve the problem *simultanously across all periods*. All algorithms, except `DeepSimulate`, also have a `*Backward` version.

Replication material for the following papers can be produced with this code:

1. DR26: "Deep Learning Algorithms for Solving Convex Finite-Horizon Models" (Druedahl and Røpke, 2026)
1. DHR26: "Deep Learning Solutions of Large Non-Convex Life-Cycle Models" (Druedahl, Huleux and Røpke, 2026)

In DHR26, the DeepVPDC algorithm is called DeepV for simplicity.

## Installation

The package can be installed  with

```
pip install -e EconDLSolvers\.
```

In `runpod.md` we explain how our code can be run at `runpod.io`.

## Overview

Consider a **single agent problem** over `T` periods indexed by `t`.

1. Agents take some `actions` eachs period.
1. Beginning-of-period `states` determine some `outcomes`, a `reward` (utility) and the post-decision states `states_pd`. 
1. The goal of the agent is to maximize *expected discounted rewards* .
1. The next period states follow from the post-decision states and some stochastic `shocks`. 
1. Agents form expectations using *quadrature* or with *monte carlo*.

To solve a specific model the user must specify:

1. The number of states, actions, outcomes and shocks 
1. Laws of motions for states given actions, outcomes and shocks
1. Random draws of intial states and shocks
1. Quadrature or Monte Carlo nodes and weights (not for `DeepSimulate`)

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

The user must specify a seed and all random numbers should be drawn using PyTorch for reproduciability. To perform computations on a GPU all arrays must be specified as a `torch.Tensor` with a specific `dtype` (typically `float32`) and some `device` (either `cpu` or GPU number i with `cuda:i` or just `i`).

Exploration is due to

1. Drawing initial states and shocks.
1. Noise in actions.

`DeepFOC` and `DeepVPD(DC)` work in two loops:

1. Outer over *episodes*: In each episode a training sample is drawn, added to the buffer memory and a training batch is drawn from the buffer.
1. Inner over *epochs*: In each epoch the traning batch is used for updating the neural networks.

We use epoch termination to avoid spending too much time optimizing the neural nets for the current training batch.

The basic class setup is:

```
from EconDLSolvers import DLSolverClass
class MyModelClass(DLSolverClass):

    # DLSolverClass(algoname,device,dtype=torch.float32)

    #########
    # setup #
    #########

    # note: arrays must be torch.tensor with dtype and on device

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

        sim.N = # number of agents in each validation sample
        sim.reps = # number of extra validation samples (default is 0)

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

        train.policy_activation_final = [] # policy activation functions, len = par.Nactions
        train.min_actions = torch.tensor([],dtype=dtype,device=device), shape = (par.Nactions,)
        train.max_actions = torch.tensor([],dtype=dtype,device=device), shape = (par.Nactions,)

    def allocate_train(self):
        # fill dependent in .train

        train.states = # training sample states, shape (par.T,train.N,par.Nstates) (dtype=dtype,device=device)
        train.states_pd = # training sample states_pd, shape (par.T,train.N,par.Nstates_pd) (dtype=dtype,device=device)
        train.shocks = # training sample shocks, shape (par.T,train.N,par.Nshocks) (dtype=dtype,device=device)
        train.actions = # training sample actions, shape (par.T,train.N,par.Nactions) (dtype=dtype,device=device)
        train.outcomes = # training sample outcomes, shape (par.T,train.N,par.Noutcomes) (dtype=dtype,device=device)
        train.reward = # training sample reward, shape (par.T,train.N) (dtype=dtype,device=device)        

    ##########
    # shocks #
    ##########

    # note: automatically transformed as dtype and moved to device

    def numerical_integration(self): # not in DeepSimulate
    
    def draw_initial_states(self,N,training=False):

    def draw_shocks(self,N,training=False):

    def draw_exploration_shocks(self,epsilon_sigma,N):
    
    ###############
    # model funcs #
    ###############

    def outcomes(model,states,actions,t=None,t0=0):

    def reward(model,states,actions,outcomes,t=None,t0=0):

    def discount_factor(model,states,t=None,t0=0):

    # only if train.terminal_actions_known = True
    def terminal_actions(model,states):

    def terminal_reward_pd(model,states_pd):

    def state_trans_pd(model,states,actions,t=None,t0=0):

    def state_trans(model,states_pd,shocks,t=None):

    def exploration(model,states,actions,eps,t=None,t0=0):

    # only DeepFOC
    def eval_equations_FOC(model,states,states_plus,actions,actions_plus):

mymodel = MyModelClass(algoname=algonname,device=device,dtype=torch.float32,par={},sim={},train={})
mymodel.solve(do_print=True,do_print_all=True)
```

At initialization the following happens:

1. `train.algoname=algoname`, `train.device=device` and `train.dtype=dtype` are set.
1. `setup()` and `setup_train()` are called (internal algorithm-specific defaults are set beforehand).
1. Values in `.par`, `.sim` and `.train` are updated with call-specific inputs dicts
1. `allocate()`, `allocate_train()` and `numerical_integration()` are called. 
1. Seed in random number generator in `torch` is set with `par.seed`
1. Algorithm and neural nets  are created and intialized
1. Initial states and shocks in `.sim` are drawn

The model can then be solved with the `.solve()` method.

## Manual termination

When `.solve()` is called a file called `solving.json` is produced. If the `terminate` value is changed to `true` the solution terminates.

By default the model has `train.transfer_grid`. If the model has an `.add_transfer()` method then the convergence approach from Druedahl and Røpke (2025) can be used. By default `.solve` produces `convergence.txt`, which contains a series of the timestamps for the best $R$'s so far, together with the required transfer starting from the current best $R$. If `train.convergence_plot=True` a plot is produced as `convergence.png` instead.

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

1. `Nneurons_policy`: neurons for policy network, `np.array([Nneurons1,Nneurons2,...])`
1. `policy_activation_intermediate`: activation functions for policy network, `list(str)`, e.g. relu or tahn
1. `policy_activation_final`: activation functions for final layer in policy network, `list(str)`, e.g. sigmoid or softplus etc

1. `value_activation_intermediate`: activation functions for policy network, `list(str)`, e.g. relu or tahn
1. `Nneurons_value`: neurons for policy network, `np.array([Nneurons1,Nneurons2,...])`
1. `N_value_NN`: number of value networks (in DeepVPD) 

1. `learning_rate_policy`: learning rate maximum for policy functions
1. `learning_rate_policy_decay`: decay in learning rate
1. `learning_rate_policy_min`: minimum learning rate
1. `learning_rate_policy_schedule`: `{t_share:fac}` time share with `lr = fac*lr_max + (1-fac)*lr_min` or `None` (default)

1. `learning_rate_value`: learning rate maximum for value functions
1. `learning_rate_value_decay`: decay in learning rate
1. `learning_rate_value_min`: minimum learning rate
1. `learning_rate_value_schedule`: `{t_share:fac}` time share with `lr = fac*lr_max + (1-fac)*lr_min` or `None` (default)

1. `manual_init_policy`: manual initilization of policy NN using `.manual_init_policy`.

A special case for `policy_activation_final` arises if the first element is `softmax`. In this case the `softmax` function is applied simultanously to the neighboring outputs with `softmax`.

For **training sample**:

1. `N`: Sample size
1. `buffer_memory`: multiples of sample size in replay buffer
1. `batch_size`: batch size

For **replay buffer**:

1. `store_actions`: store actions in replay buffer. Default is `False`
1. `store_reward`: store rewards in replay buffer. Default is `False`
1. `store_pd`: store post-decision states in replay buffer. Default is `False`
1. `i_t_index`: include t index in replay buffer. Default is `False`

For **exploration**:

1. `epsilon_sigma`: scale of exploration shocks, `torch.tensor([scale_action1, scale_action2, ...])`
1. `epsilon_sigma_decay`: decay in scale of exploration shocks
1. `epsilon_sigma_min`: minimum scale of exploration shocks `torch.tensor([scale_action1, scale_action2, ...])`
1. `explore_frac`: fraction of explorers (in DeepSimulate), array of length `par.T`

For **numerical integration**:

1. `use_quad`: do quadrature or Monte Carlo integration. Default is `True`. 
1. `redraw_mc`: re-draw monte carlo nodes each episode. Default is `False`. 
1. `update_numint_weights`: call `.numint_weights_update(states_pd_b)` when computing target in `DeepVPD`. Default is 'False'.

For **smoothing**:

1. `start_train_policy`: start training policy after this number of episodes
1. `tau`: target smoothing coefficient
1. `tau_final`: final tau value
1. `tau_schedule`: `{t_share:fac}` time share with `tau = (1-fac)*tau + fac*tau_final` or `None` (default)
1. `use_target_policy`: use target in update policy. Default is `True
1. `use_target_value`: use target in update value. Default is `True`
1. `target_value_in_policy`: use target networks for policy training. Default is `False`

For **termination of episodes**:

1. `K`: maximum number of episodes before termination
1. `K_time`: maximum number of minutes termination
1. `sim_R_freq`: episodes between each simulation of lifetime reward (`sim.R`)

1. `convergence_plot`: print convergence plot in convergence.png. Default is `False`
1. `transfer_grid`:grid for calculation of transfer
1. `Delta_transfer`: transfer tolerance for improvement
1. `Delta_time`: time tolerance for improvement
1. `K_time_min`: minimum number of minutues before termination

1. `terminate_on_policy_loss`: terminate if policy loss is below tolerance. Default is `False`
1. `tol_policy_loss`: tolerance for policy loss
1. `track_sim_policy_loss`: track policy loss on sim (expensive). Default is `False`
1. `N_sample_policy_loss` = 100 # how many samples batches when computing policy_loss on sim

For **termination of epochs**:

1. `epoch_termination`: terminate epochs early if no improvement. Default is `True`
1. `Nepochs_policy`: number of epochs in update to update policy
1. `Delta_epoch_policy`: number of epochs without improvement before termination
1. `epoch_policy_min`: minimum number of epochs
1. `Nepochs_value`: number of epochs in update to update value
1. `Delta_epoch_value`: number of epochs without improvement before termination
1. `epoch_value_min`: minimum number of epochs

For **clipping gradints:**

1. `clip_grad_policy`: limit value of gradient to this number
1. `clip_grad_value`: limit value of gradient to this number

For **using FOC in value-based algorithms**:

1. `use_FOC`: use analytical FOC. Default is `False`
1. `NFOC_targets` : number of FOC targets
1. `FOC_weight_pol`: weight on FOC when evaluating policy loss
1. `FOC_weight_val`: weight on FOC when evaluating value loss
1. `value_weight_pol`: weight on value of choice when evaluating policy loss
1. `value_weight_val`: weight on value of choice when evaluating value loss

for **backward induction**:

1. `backward`: do backwards induction. Default is `False` (but `True` for `*Backward` algorithms)
1. `use_simult_in_backward`: use simultaneous NN when training backward NN. Default is `True`
1. `Nneurons_policy_t`: number of neurons in policy NN at each time t, `np.array([Nneurons1,Nneurons2,...])`
1. `Nneurons_value_t`: number of neurons in value NN at each time t, `np.array([Nneurons1,Nneurons2,...])`
1. `NN_init_std`: standard deviation for initialization of weights and biases from normal distribution
1. `Nepochs_policy_t`: number of epochs for policy NN at each time t, `torch.tensor([epochs1, epochs2, ...])`
1. `learning_rate_policy_t`: learning rate for policy NN at each time t, `torch.tensor([lr1, lr2, ...])`
1. `learning_rate_policy_decay_t`: learning rate decay for policy NN at each time t, `torch.tensor([lr_decay1, lr_decay2, ...])`
1. `Nepochs_value_t`: number of epochs for value NN at each time t, list, `torch.tensor([epochs1, epochs2, ...])`
1. `learning_rate_value_t`: learning rate for value NN at each time t, `torch.tensor([lr1, lr2, ...])`
1. `learning_rate_value_decay_t`: learning rate decay for value NN at each time t, `torch.tensor([lr_decay1, lr_decay2, ...])`

**Misc.**:

1. `terminal_actions_known`: boolean for whether terminal actions are known
1. `min_actions`: minimum for each actions, `torch.tensor([min_action1, min_action2, ...])`
1. `max_actions`: maximum for each actions, `torch.tensor([max_action1, max_action2, ...])`
1. `NN_use_best`: load best NNs when terminating training. Default is `True`
1. `epoch_use_best`: load best NNs after each epoch. Default is `True`
1. `only_initial_states_and_shocks`: only train on initial states and shocks. Default is `False`
1. `Ngpus` : number of GPUs used
1. `N_target_batches` : number of batches when computing target 
1. `eq_w`: weight to put on each FOC if multiple in DeepFOC. Default is equal weights
1. `dtype`: datatype. Default is `torch.float32`

## Misc

1. The default time inputs to the neural nets are dummies for each period, `train.time_input_type = 'one_hot'`, so `train.Ninputs_time = par.T`.

1. Time embedding can be used with `train.time_input_type = 'embedding`. Must also choose `train.Ninputs_time`.

1. Manual time inputs can be set in `.time_inputs(time_indices)` with `train.time_input_type = 'manual`. Must also choose `train.Ninputs_time`.

1. The inputs to the neural nets can be transformed with `.input_transformation(x,t=None,t0=t0)` setting `train.input_transformation=True`. Must also choose `train.Ninputs_aux`.

1. Solution can be timed with:

    ``` 
    mymodel.time_solve()
    ```

1. Multiple GPUs can be used with:

    ```
    model = MyModelClass(algoname=algoname,device=0)
    model.solve_DPP(do_print=True)
    ```

1. Models can be saved and loaded with:

    ```
    model.save('test.pt')
    model_load = BufferStockModelClass(load='test.pt',device=device)
    ```

1. `train.do_sim_eps=True` produces `.sim_eps` with simulation noise.

## Getting started

Look at `0_SimpleConSav\` for a simple example.

## Replication for Druedahl and Røpke (2025)

The results for the buffer-stock model are reproduced as:

1. Run `BufferStock\generate_DP_files_DR25.py`
1. Run `BufferStock\generate_DL_files_DR25.py`
1. Run `BufferStock\02_Results_DR25.ipynb`
1. Run `BufferStock\02_Results_DR25_MoreShocks.ipynb`

The results for the model with multiple durable goods and convex adjustment costs are reproduced as:

1. Run `Durables\generate_DP_files_DR25.py`
1. Run `Durables\generate_DL_files_DR25.py`
1. Run `Durables\02_Results_DR25.ipynb`

All output is saved in `output/`.

## Replication for Druedahl, Huleux and Røpke (2025)

TBA

## Contributers

Main developers:

* Jeppe Druedahl
* Jacob Røpke
* Raphaël Huleux

Additional developers:

* Mikkel Østergaard Reich
* Martin Andreas Kildemark

**Want to contribute?** Please contact [jeppe.druedahl@econ.ku.dk](mailto:jeppe.druedahl@econ.ku.dk )
