import os
import json 
import datetime
from copy import deepcopy
from types import SimpleNamespace
import cProfile
import numpy as np
import pickle
import pstats
import time
import torch

# local
from . import replay_buffer
from . import DeepSimulate
from . import DeepQ
from . import DeepFOC
from . import DeepV
from . import DeepVPD
from . import DeepVPDDC

from . import neural_nets
from .DDP import solve_DDP, _solve_DDP_process
from .gpu import get_free_memory
from .simulate import simulate
from . import figs

algos = {
    'DeepSimulate': DeepSimulate,
    'DeepFOC': DeepFOC,
    'DeepV': DeepV,
    'DeepVPD': DeepVPD,
    'DeepQ': DeepQ,
    'DeepVPDDC': DeepVPDDC,
}

def solving_json(postfix):

    filename = "solving.json"

    # check if the file exists
    if not os.path.exists(filename):
        with open(filename, 'w') as file:
            json.dump([], file)

    # read the existing data from the file
    with open(filename, 'r') as file:
        data = json.load(file)

    # add a new row with the current timestamp and the value false
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_entry = {}
    if not postfix == '': new_entry['postfix'] = postfix
    new_entry['timestamp'] = timestamp
    new_entry['terminate'] = False
    data.append(new_entry)

    # write the updated data back to the file
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)

    return timestamp

def check_solving_json(timestamp):

    filename = "solving.json"

    try:

        with open(filename, 'r') as file:
            data = json.load(file)

        for entry in data:
            if entry.get('timestamp') == timestamp:
                terminate = entry.get('terminate', None)

        return terminate == True

    except Exception:

        return False
    
class DLSolverClass():
    """ generic solution algorithm """

    #########
    # setup #
    #########

    def __init__(self,
              algoname=None,device=None,dtype=None,par=None,sim=None,train=None,
              torch_rng_state=None,show_memory=False,load=None):
        """ initialize """

        if not torch.cuda.is_available(): show_memory = False

        assert device is not None, 'device must be specified'

        if load is not None:

            assert algoname is None, 'algoname must be None when loading'
            assert dtype is None, 'dtype must be None when loading'
            assert par is None, 'par must be None when loading'
            assert train is None, 'train must be None when loading'
            assert torch_rng_state is None, 'torch_rng_state must be None when loading'
            assert not show_memory, 'show_memory must be False when loading'
            
            with open(f'{load}', 'rb') as f:
                load_dict = pickle.load(f)

            algoname = load_dict['train'].algoname
            dtype = load_dict['train'].dtype

        if show_memory:
            free_GB_0 = get_free_memory(device)
            print(f'initial: {free_GB_0:.2f}GB free')

        assert algoname is not None, 'algoname must be specified'
        if dtype is None: dtype = torch.float32
        if par is None: par = {}
        if sim is None: sim = {}
        if train is None: train = {}

        # a. namespaces
        self.par = SimpleNamespace()
        self.sim = SimpleNamespace()
        self.train = SimpleNamespace()

        self.train.algoname = algoname
        self.train.device = device
        self.train.dtype = dtype

        self.policy_NN = None
        self.value_NN = None
        self.policy_NN_target = None
        self.value_NN_target = None		
        self.info = {}
        self.info_timing = {}

        # b. functions
        self._fetch_algo_functions(algoname)

        model_names = [
            'setup_train','allocate_train',
            'draw_initial_states','draw_shocks',
            'outcomes','reward','discount_factor','state_trans_pd','state_trans',
        ]	

        for name in model_names:
            assert hasattr(self,name) and callable(getattr(self,name)), f'The method {name} must be specified'

        # c. setup and allocate par and sim
        self._setup_default() # default parameters
        self.setup() # overwrite with model-specific parameters

        for k,v in par.items(): self.par.__dict__[k] = v # overwrite with call-specific parameters
        for k,v in sim.items(): self.sim.__dict__[k] = v # overwrite with call-specific parameters
        self.allocate()

        if show_memory: 
            free_GB_1 = get_free_memory(device) 
            print(f'setup()+allocate(): {free_GB_0-free_GB_1:.2f}GB allocated')

        # checks
        parnames = ['T','Nstates','Nstates_pd','Nshocks','Noutcomes','Nactions'] 
        for name in parnames: assert hasattr(self.par,name), f'Parameter par.{name} must be specified'
        
        simnames = ['N']  
        for name in simnames: assert hasattr(self.sim,name), f'Parameter sim.{name} must be specified'

        # d. setup and allocate train
        self._setup_train_default() # default parameters
        self.algo.setup(self) # overwrite with algo-specific parameters
        self.setup_train() # overwrite with model-specific parameters
        
        for k,v in train.items(): self.train.__dict__[k] = v # overwrite with call-specific parameters

        self.allocate_train()
        # get quad if self.quad exists
        if hasattr(self,'quad'): 
            self._get_quad()

        if show_memory: 
            free_GB_2 = get_free_memory(device) 
            print(f'setup_train()+allocate_train()+quad(): {free_GB_1-free_GB_2:.2f}GB allocated')

        # checks
        assert hasattr(self.train,'policy_activation_final'), 'train.policy_activation_final must be specified'
        assert hasattr(self.train,'min_actions'), 'train.min_actions must be specified'
        assert hasattr(self.train,'max_actions'), 'train.max_actions must be specified'
        assert len(self.train.policy_activation_final) in {self.par.Nactions, 1}, 'train.policy_activation_final must have the same length as Nactions or have length 1'
        assert len(self.train.min_actions) == self.par.Nactions, 'train.min_actions must have same length as Nactions'
        assert len(self.train.max_actions) == self.par.Nactions, 'train.max_actions must have same length as Nactions'
        if not self.train.epsilon_sigma is None:
            assert len(self.train.epsilon_sigma) == self.par.Nactions, 'train.epsilon_sigma must have same length as Nactions'
            assert len(self.train.epsilon_sigma_min) == self.par.Nactions, 'train.epsilon_sigma_min must have same length as Nactions'

        # e. prepate simulation
        if load is None: self.prepare_simulate_R(torch_rng_state=torch_rng_state)

        # f. allocate DL solver
        self.allocate_DLSolver()

        if show_memory: 
            free_GB_3 = get_free_memory(device) 
            print(f'allocate_DLSolver(): {free_GB_2-free_GB_3:.2f}GB allocated')
            print(f'final: {free_GB_3:.2f}GB free')

        # i. overwrite
        if load is not None:

            # o. par, sim and train
            nss = ['par','sim','train','sim_eps']
            for ns in nss:
                if ns in load_dict:
                    if not hasattr(self,ns): self.__dict__[ns] = SimpleNamespace()
                    for k,v in load_dict[ns].__dict__.items():					
                        if isinstance(v,np.ndarray): 
                            self.__dict__[ns].__dict__[k] = torch.tensor(v,device=self.train.device,dtype=self.train.dtype)
                        else: 
                            self.__dict__[ns].__dict__[k] = v

            # oo. info
            self.info = load_dict['info']
            self.info_timing = load_dict['info_timing']

            # ooo. nets
            self.policy_NN.load_state_dict(load_dict['policy_NN'])
            if 'value_NN' in load_dict: self.value_NN.load_state_dict(load_dict['value_NN'])

    def _fetch_algo_functions(self,algoname):
        """ fetch functions from algorithm"""

        self.algo = SimpleNamespace()

        # a. check if exists
        try:
            algomodule = algos[algoname]
        except Exception as e:
            print(f"Error: {e}")
            raise ValueError(f"Algorithm {algoname} not implemented (or error in import)")

        # b. required
        algo_names = ['setup','update_NN','create_NN','scheduler_step']
        for name in algo_names: 
            self.algo.__dict__[name] = eval(f'algomodule.{name}')
        
        # c. optional
        algo_names_optional = ['create_target_NN', 'transfer_nets_to_device','update_NN_DDP','compute_discrete_choice_prob_from_NN','policy_loss_f']
        for name in algo_names_optional: 
            try:
                self.algo.__dict__[name] = eval(f'algomodule.{name}')
            except:
                pass
                
    def _setup_default(self):
        """ setup default parameters """

        par = self.par
        par.NDC = 0 # number of discrete choices

        sim = self.sim
        sim.N = 100_000 # number of agents
        sim.reps = 0 # number of repetitions

    def _setup_train_default(self):
        """ setup default parameters """

        train = self.train

        # a. neural nets
        train.Nneurons_value = np.array([500,500]) # size and number of layers
        train.value_activation_intermediate = 'relu'
        train.N_value_NN = None # number of value networks
        train.Nneurons_policy = np.array([500,500]) # size and number of layers
        train.policy_activation_intermediate = 'relu'

        # b. learning
        train.learning_rate_policy = 1e-3  # learning rate
        train.learning_rate_policy_decay = 0.9999
        train.learning_rate_policy_min = 1e-5
        train.learning_rate_value = 1e-3 # learning rate
        train.learning_rate_value_decay = 0.9999
        train.learning_rate_value_min = 1e-5

        train.epoch_termination = True # terminate epochs early if no improvement
        
        train.Nepochs_policy = 15 # number of epochs in update
        train.epoch_policy_min = 5 # minimum number of epochs
        train.Delta_epoch_policy = 5 # number of epochs without improvement before termination
        
        train.Nepochs_value = 50 # number of epochs in update
        train.epoch_value_min = 10 # minimum number of epochs
        train.Delta_epoch_value = 10 # number of epochs without improvement before termination

        # c. sample and replay buffer
        train.N = 150 # sample size
        train.buffer_memory = 8 # buffer_size = buffer_memory*N
        train.batch_size = train.N # batch size

        train.only_initial_states_and_shocks = False # only initial states and shock (e.g. in DeepSimulate)
        train.store_actions = False # store actions in replay buffer
        train.store_reward = False # store reward in replay buffer
        train.store_pd = False # store pd states in replay buffer
        train.i_t_index = False # include t index in buffer

        # d. solver and exploration
        train.terminal_actions_known = False # terminal actions are known
        train.do_exo_actions_periods = -1 # number of periods with uniform sampling of actions
        train.start_train_policy = -1 # start training policy after this number of iterations

        train.K_time = 60 # maximum time in minutes
        train.K = np.inf # number of iterations
        train.sim_R_freq = 10 # frequency of simulation of lifetime reward
        train.convergence_plot = False # plot convergence, else txt-file is written

        train.K_time_min = np.inf # minimum time in minutes
        train.transfer_grid = np.arange(-10*1e-4,0+1e-4,1e-4) # transfer grid
        train.Delta_transfer = 1e-4 # size of transfer for convergence
        train.Delta_time = 20 # time without improvement before termination

        train.epsilon_sigma = None # std of exploration shocks
        train.epsilon_sigma_decay = 1.0 # decay of epsilon_sigma
        train.epsilon_sigma_min = None # minimum epsilon_sigma

        train.terminate_on_policy_loss = False # terminate if policy loss is below tolerance
        train.tol_policy_loss = 1e-4 # tolerance for policy loss
        
        # e. misc
        train.clip_grad_value = 100.0 # clip value
        train.clip_grad_policy = 100.0 # clip policy
        train.use_input_scaling = False # use input scaling
        train.N_sample_policy_loss = 100 # how many samples batches when computing policy_loss on sim

        if torch.cuda.is_available():
            train.Ngpus = torch.cuda.device_count() # number of GPUs
        else:
            train.Ngpus = 0

        train.k = -1 # initialization
        train.allow_synchronize = False # allow syncronize
        train.do_sim_eps = False # simulate with exploration

    def _get_quad(self):
        """ get quadrature points and weights """

        train = self.train
        dtype = train.dtype
        device = train.device

        quad,quad_w = self.quad() # shape = (Nquad,Nshocks), shape = (Nquad,)

        train.Nquad = quad.shape[0]
        assert quad.shape[0] == quad_w.shape[0], 'quad and quad_w must have same length'
        assert quad.shape[1] == self.par.Nshocks, 'quad must have same length as Nshocks'
        assert torch.isclose(torch.sum(quad_w),torch.tensor(1.0,device=quad_w.device,dtype=quad_w.dtype)), 'quad_w must sum to 1.0'

        train.quad = quad.to(dtype).to(device)
        train.quad_w = quad_w.to(dtype).to(device)

    ############
    # allocate #
    ############
    
    def _draw_initial_states(self,N,training=False):
        """ draw initial states """

        train = self.train

        dtype = train.dtype
        device = train.device

        initial_states = self.draw_initial_states(N,training=training) # len(tuple_initial_states) = Nstates, tuple_initial_states[i].shape = (N,)

        return initial_states.to(dtype).to(device)

    def _draw_shocks(self,N):
        """ draw shocks """

        par = self.par
        train = self.train

        dtype = train.dtype
        device = train.device

        if par.NDC == 0:
            shocks = self.draw_shocks(N) # shape = (T,N,Nshocks)
            shocks = shocks.to(dtype).to(device) 
            taste_shocks = None
        else:
            shocks,taste_shocks = self.draw_shocks(N) # shape = (T,N,Nshocks), # shape = (T,N,1)
            shocks = shocks.to(dtype).to(device) 
            taste_shocks = taste_shocks.to(dtype).to(device)

        return shocks,taste_shocks
    
    def _draw(self,sim,training=False):
        """ draw everything """

        par = self.par

        sim.states[0] = self._draw_initial_states(sim.N,training=training)
        if par.NDC == 0:
            sim.shocks[:,:,:],_ = self._draw_shocks(sim.N)
        else:
            sim.shocks[:,:,:],sim.taste_shocks[:,:] = self._draw_shocks(sim.N)	

    def prepare_simulate_R(self,torch_rng_state=None):
        """ prepare simulation """

        par = self.par
        sim = self.sim

        # a. set seed
        if torch_rng_state is None:
            torch.manual_seed(self.par.seed)
        else:
            torch.set_rng_state(torch_rng_state)

        # b. draw initial states and shocks
        self.torch_rng_state = {}
        for rep in range(sim.reps+1):

            # i. get state
            self.torch_rng_state[('sim',rep)] = torch.get_rng_state()

            # ii. draw
            self._draw(sim)

    def allocate_DLSolver(self):
        """ allocate algorithm variables and parameters """

        self.algo.create_NN(self)
        self._create_repbuffer()

    def _create_repbuffer(self):
        """ create replay buffer """

        # a. unpack
        par = self.par
        train = self.train

        # b. create replay buffer
        Nstates = par.Nstates
        Nstates_pd = par.Nstates_pd

        self.rep_buffer = replay_buffer.ReplayBuffer(
            Nstates=Nstates,
            Nstates_pd=Nstates_pd,
            Nactions=par.Nactions, 
            T=par.T, 
            N=train.N,
            dtype=train.dtype,
            memory=train.buffer_memory,
            device=train.device,
            store_actions=train.store_actions,
            store_reward=train.store_reward,
            store_pd=train.store_pd ,
            i_t_index=train.i_t_index,
        ) 

    ########################
    # evaluate neural nets #
    ########################

    eval_policy = neural_nets.eval_policy
    eval_value = neural_nets.eval_value
    eval_value_pd = neural_nets.eval_value_pd
    eval_actionvalue = neural_nets.eval_actionvalue

    ###################
    # training sample #
    ###################

    def _simulate_training_sample(self,epsilon_sigma,do_exo_actions,device=None):
        """ generate training sample"""

        if 'time._simulate_training_sample' in self.info: 
            t0 = time.perf_counter()

        # a. unpack
        par = self.par
        train = self.train

        if device is None: device = train.device
        dtype = train.dtype	

        if epsilon_sigma is None: epsilon_sigma = torch.zeros(par.Nactions,dtype=dtype,device=device)

        # b. draw initial states and shocks
        self._draw(train,training=True)

        if not train.only_initial_states_and_shocks:
            
            # i. draw exploration shocks
            eps = self.draw_exploration_shocks(epsilon_sigma,train.N).to(dtype).to(device) # shape = (T,N,Nactions)
            
            # ii. exogenous actions
            if do_exo_actions:
                exo_actions = self.draw_exo_actions(train.N).to(dtype).to(device) # shape = (T,N,Nactions)
            else:
                exo_actions = None

            # iii. simulate
            simulate(self,train,eps=eps,exo_actions=exo_actions)

            # iv. store in buffer
            self.rep_buffer.add(train)	

        if 'time._simulate_training_sample' in self.info: 
            self.info['time._simulate_training_sample'] += time.perf_counter() - t0	

    #########
    # solve #
    #########

    def convergence(self,folder='',postfix='',close_fig=True):
        """ check for convergence (and plot) """

        train = self.train

        if not hasattr(self,'add_transfer'): return False

        do_plot = train.convergence_plot
        return figs.convergence_plot(self,folder=folder,postfix=postfix,close_fig=close_fig,do_plot=do_plot)
    
    def simulate_R(self):
        """ simulate R """
        
        if 'time.simulate_R' in self.info: t0 = time.perf_counter()

        simulate(self,self.sim)
        
        if 'time.simulate_R' in self.info: self.info['time.simulate_R'] += time.perf_counter() - t0

    def simulate_Rs(self):
        """ simulate multiple Rs """

        sim = self.sim
        
        # a. remember
        old_rng_state = torch.get_rng_state()

        # b. loop
        Rs = np.zeros(self.sim.reps)
        for rep in range(sim.reps):

            # i. set rng
            torch.set_rng_state(self.torch_rng_state[('sim',rep)])
            
            # ii. draw
            sim = deepcopy(self.sim)
            self._draw(sim)

            # iii. simulate
            simulate(self,sim)
            Rs[rep] = sim.R.item()

        # c. reset
        torch.set_rng_state(old_rng_state)

        return Rs
    
    def _update_best(self):
        """ update best R and neural nets """

        t0 = time.perf_counter()

        sim = self.sim
        train = self.train
        info = self.info

        best = info['best']

        self.simulate_R()
        info[('R',train.k)] = sim.R.item()
        # if train.algoname == 'DeepFOC': # JR: code for computing policy loss on sim
        #     states = sim.states
        #     if train.terminal_actions_known:
        #         states = states[:-1]
        #     policy_loss = 0.0
        #     N_sample = round(sim.N/train.N_sample_policy_loss)
        #     assert sim.N % train.N_sample_policy_loss == 0, 'N_sample_policy_loss must divide N'
        #     for i in range(train.N_sample_policy_loss):
        #         states_sample = states[:,i*N_sample:(i+1)*N_sample]
        #         with torch.no_grad():
        #             policy_loss += self.algo.policy_loss_f(self,states_sample).item()
        #     policy_loss /= train.N_sample_policy_loss
        #     info[('policy_loss_sim',train.k)] = policy_loss
        
        if sim.R.item() > best.R: 

            best.k = train.k
            best.time = self.info[('k_time',train.k)]
            best.R = sim.R.item()
            
            best.policy_NN = deepcopy(self.policy_NN.state_dict())
            best.value_NN = deepcopy(self.value_NN.state_dict()) if not self.value_NN is None else None

            # compute transfer
            if hasattr(self,'add_transfer'):


                sim_base = sim

                R_transfer = self.info['R_transfer'] = np.zeros(train.transfer_grid.size)
                for i,transfer in enumerate(train.transfer_grid):

                    self.sim = deepcopy(sim)
                    self.add_transfer(transfer)
                    self.simulate_R()
                    R_transfer[i] = self.sim.R.item()

                self.sim = sim_base
        
        info['time.update_best'] += time.perf_counter() - t0
    
    def _print_progress(self,t0_k,t0_solve):
        """ print progress """

        train = self.train
        info = self.info

        best = info['best']

        k = train.k
        time_k = time.perf_counter()-t0_k			
        time_tot = (time.perf_counter()-t0_solve)/60 + info['time']/60		
        value_epochs = info[('value_epochs',k)]
        policy_epochs = info[('policy_epochs',k)]

        R = info[('R',k)]
        print(f'{k = :5d} of {train.K-1}: sim.R = {R:12.8f} [best: {best.R:12.8f}] [{time_k:.1f} secs] [{value_epochs = :3d}] [{policy_epochs = :3d}] [{time_tot:6.2f} mins]',end='')
        
        if train.terminate_on_policy_loss and k >= train.start_train_policy: 
            policy_loss = info[('policy_loss',k)]
            print(f' [{policy_loss = :.8f}]',end='')

        if not np.isnan(R):
            if best.k == k:
                print('')
            else:
                print(f' no improvements in {time_tot-best.time/60:.1f} mins')
        else:
            print('')

    def solve(self,do_print=False,do_print_all=False,show_memory=False,postfix=''):
        """ solve model """

        if not torch.cuda.is_available(): show_memory = False

        timestamp = solving_json(postfix)
        t0_solve = time.perf_counter()

        if do_print_all: do_print = True
        if do_print: print(f"started solving: {timestamp}")

        # a. unpack
        sim = self.sim
        train = self.train
        info = self.info
        
        if show_memory: 
            free_GB_ini = get_free_memory(train.device)
            print(f'initial: {free_GB_ini:.2f}GB free')

        if 'solve_in_progress' in self.info:
            continued_solve = True
        else:
            self.info['solve_in_progress'] = True
            continued_solve = False

        if not continued_solve:
            self.info['time'] = 0.0
            self.info['time.update_NN'] = 0.0
            self.info['time.update_NN.train_value'] = 0.0
            self.info['time.update_NN.train_policy'] = 0.0
            self.info['time.scheduler'] = 0.0
            self.info['time.convergence'] = 0.0
            self.info['time.update_best'] = 0.0
            self.info['time._simulate_training_sample'] = 0.0
            self.info['time.simulate_R'] = 0.0

        # b. initialize best
        if not continued_solve:

            best = info['best'] = SimpleNamespace()
            best.k = -1		
            best.time = 0.0
            best.R = -np.inf
            best.policy_NN = None
            best.value_NN = None

        else:

            best = info['best']
            
        # c. loop over iterations
        if not continued_solve:
            epsilon_sigma = info['epsilon_sigma'] = deepcopy(train.epsilon_sigma)
            k = 0
        else:
            epsilon_sigma = info['epsilon_sigma']
            k = info['iter']

        while True:
            
            t0_k = time.perf_counter()
            train.k = k

            info[('value_epochs',k)] = 0 # keep track of number of value epochs (updated in algo)
            info[('policy_epochs',k)] = 0 # keep track of number of policy epochs (updated in algo)

            # i. simulate training sample
            do_exo_actions = k < train.do_exo_actions_periods # exo actions defined in draw_exo_actions
            self._simulate_training_sample(epsilon_sigma,do_exo_actions)
            
            # update exploration
            if epsilon_sigma is not None:
                epsilon_sigma *= train.epsilon_sigma_decay
                epsilon_sigma = np.fmax(epsilon_sigma,train.epsilon_sigma_min)

            # ii. update neural nets
            t0 = time.perf_counter()
            self.algo.update_NN(self) # is different for each algorithm
            info['time.update_NN'] += time.perf_counter() - t0

            # iii. scheduler step
            t0 = time.perf_counter()
            self.algo.scheduler_step(self)
            info['time.scheduler'] += time.perf_counter() - t0

            # iv. print and termination
            info[('k_time',k)] = time.perf_counter()-t0_solve + info['time']
            info[('update_time',k)] = time.perf_counter()-t0_k
            if not train.sim_R_freq is None and k % train.sim_R_freq == 0:

                # o. update best
                self._update_best()

                # oo. print
                if do_print: self._print_progress(t0_k,t0_solve)

                # ooo. convergence
                t0 = time.perf_counter()
                
                terminate = self.convergence(postfix=postfix)
                info['time.convergence'] += time.perf_counter() - t0

                # oooo. termination
                if info[('k_time',k)]/60 > train.K_time_min: # above minimum time
                    if terminate: break # convergence criterion satisfied
            
            else:

                info[('R',k)] = np.nan		
                if do_print_all: self._print_progress(t0_k,t0_solve)
            
            # v. termination from policy loss
            if k > train.start_train_policy:
                if train.terminate_on_policy_loss and info[('policy_loss',k)] < train.tol_policy_loss:
                    self._update_best() # final update
                    k += 1
                    print(f'Terminating after {k} iter, policy loss lower than tolerance')
                    break
                
            # vi. termination from time
            time_tot = (time.perf_counter()-t0_solve)/60 + info['time']/60
            if time_tot > train.K_time:
                self._update_best() # final update
                k += 1
                print(f'Terminating after {k} iter, max time {train.K_time} mins reached')
                break
                
            # vii. check if solving.json has been updated for manual termination
            manuel_terminate = check_solving_json(timestamp)
            if manuel_terminate:
                self._update_best() # final update
                k += 1
                print(f'Terminating after {k} iter, manuel termination')
                break

            # vii. terminate from too many iterations
            k += 1
            if k >= train.K: 
                self._update_best() # final update
                print(f'Terminating after {k} iter, max number of iterations reached')
                break            

        # d. load best solution
        t0 = time.perf_counter()

        if self.policy_NN is not None: 
            if not best.policy_NN is None:
                self.policy_NN.load_state_dict(best.policy_NN)
        
        if self.value_NN is not None: 
            if not best.value_NN is None:
                self.value_NN.load_state_dict(best.value_NN)
                    
        info['time.update_best'] += time.perf_counter() - t0

        # e. final simulation
        t0 = time.perf_counter()
        self.simulate_R()
        info['time.update_best'] += time.perf_counter() - t0

        # f. store
        info['R'] = best.R
        info['time'] += time.perf_counter()-t0_solve
        info['iter'] = k
        info[('value_epochs','mean')] = np.mean([info[('value_epochs',k)] for k in range(info['iter'])])
        info[('policy_epochs','mean')] = np.mean([info[('policy_epochs',k)] for k in range(info['iter'])])

        if do_print: self.show_info()

        # g. extra: multiples simulations of R
        if do_print and self.sim.reps > 0: print('Simulating multiple Rs')
        Rs = self.simulate_Rs()
        info['Rs'] = Rs
    
        # h. extra: simulation with epsilon shocks
        if train.do_sim_eps and not epsilon_sigma is None:

            if do_print: print('Simulating with exploration')

            self.sim_eps = deepcopy(sim)
            eps = self.draw_exploration_shocks(epsilon_sigma,self.sim_eps.N).to(train.dtype).to(train.device)
            simulate(self,self.sim_eps,eps=eps) # note: same initial states and shocks as in .sim are used

        else:

            self.sim_eps = None
        
        # h. empty cache
        if show_memory: 
            free_GB_fin = get_free_memory(train.device)
            print(f'solve(): {free_GB_ini-free_GB_fin:.2f}GB allocated')

        if torch.cuda.is_available(): torch.cuda.empty_cache()

        if show_memory:
            free_GB_after = get_free_memory(train.device)
            print(f'empty_cache(): {free_GB_fin-free_GB_after:.2f}GB deallocated')
            print(f'final: {free_GB_after:.2f}GB free')

    def show_info(self):
        
        R = self.info['R']
        iter = self.info['iter']
        policy_epochs = self.info[('policy_epochs','mean')]
        value_epochs = self.info[('value_epochs','mean')]
        time_tot = self.info['time']

        print(f'{R = :.4f}, time = {time_tot/60:.1f} mins, iter = {iter}, policy epochs = {policy_epochs:.2f}, value epochs = {value_epochs:.2f}')

    def time_solve(self,Nstats=0):
        """ Time the solve function """

        # a. turn on synchronization
        train = self.train
        train.allow_synchronize = True

        # b. profile
        pr = cProfile.Profile()
        pr.enable()

        t0 = time.perf_counter()
        self.solve(do_print=False)
        t1 = time.perf_counter()
        time_solve = t1-t0

        pr.disable()

        if Nstats > 0:
            
            print('')

            (
                        pstats.Stats(pr)
                        .strip_dirs()
                        .sort_stats(pstats.SortKey.CUMULATIVE)
                        .print_stats(Nstats)
            )

        print('')

        # c. gather stats
        stats = pstats.Stats(pr)		

        def get_all_keys(d):
            for key, value in d.items():
                yield key
                if isinstance(value,dict):
                    yield from get_all_keys(value)

        def get_all_keys_in_levels(d,i=0,base=None):
            if base is None: base = ()
            for key, value in d.items():
                yield (i,base,key)
                if isinstance(value, dict):
                    if len(base) > 0:
                        yield from get_all_keys_in_levels(value,i+1,(*base,key))
                    else:
                        yield from get_all_keys_in_levels(value,i+1,(key,))

        if self.train.algoname == 'DeepSimulate':
            funcs_dict = {
                '_simulate_training_sample':None,
                'simulate_R':None,
                'update_NN':{
                    'policy_loss_f':None,
                    'NN_step':None
                    }
            }
        elif self.train.algoname == 'DeepFOC':
            funcs_dict = {
                '_simulate_training_sample':None,
                'simulate_R':None,
                'update_NN':{
                    'train_policy':{
                        'policy_loss_f':None,
                        'NN_step_policy':None
                        }  
                    }
            }
        elif self.train.algoname == 'DeepVPD':
            funcs_dict = {
                '_simulate_training_sample':None,
                'simulate_R':None,
                'update_NN':{
                    'train_value':{
                        'compute_target':None,
                        'value_loss_f':None,
                        'NN_step_value':None
                        },
                    'train_policy':{
                        'policy_loss_f':None,
                        'NN_step_policy':None
                        }  
                }
            }
        elif self.train.algoname == 'DeepQ':
            funcs_dict = {
                '_simulate_training_sample':None,
                'simulate_R':None,
                'update_NN':{
                    'train_value':{
                        'NN_step_value':None
                        },
                    'train_policy':{
                        'NN_step_policy':None
                        }  
                }
            }
        else:
            raise NotImplementedError(f"Algo {self.train.algoname} not implemented")
        
        func_names = list(get_all_keys(funcs_dict))

        self.info_timing = {}
        for func_info, func_stats in stats.stats.items():
            _, _, function_name = func_info # this assumes no duplicate function names across modules
            if function_name in func_names:
                self.info_timing[function_name] = func_stats

        # d. print
        print(f'Total time: {time_solve:.1f} secs')
        for f in get_all_keys_in_levels(funcs_dict):
            
            lvl = f[0]
            if lvl == 0:
                supname = 'total'
                suptime = time_solve
            else:
                supname = f[1][-1]
                suptime = self.info_timing[supname][3]
            
            name = f[2]
            _name = '  '*f[0] + name
            nowtime = self.info_timing[name][3]
            
            print(f'{_name:35s} {100*nowtime/suptime:5.1f}% of {supname} [{nowtime:.1f} secs]')		

        # e. turn off synchronization		
        train.allow_synchronize = False

    #############
    # save-load #
    #############


    def save(self,filename):
        """ save model """

        save_dict = {}

        # a. par, sim and train
        nss = ['par','sim','train','sim_eps']
        
        for ns in nss:
            if hasattr(self,ns) and not self.__dict__[ns] is None:

                save_dict[ns] = SimpleNamespace()
                for k,v in self.__dict__[ns].__dict__.items():
                    
                    if k in ['device']: continue
                    if ns == 'sim_eps' and not k in ['states']: continue
                    
                    if isinstance(v,torch.Tensor): 
                    
                        if ns == 'train': continue
                        save_dict[ns].__dict__[k] = v.cpu().numpy()
                    
                    else: 
                    
                        save_dict[ns].__dict__[k] = v
        
        # b. info
        save_dict['info'] = self.info
        for k, v in save_dict['info']['best'].policy_NN.items():
            save_dict['info']['best'].policy_NN[k] = v.cpu()
        if hasattr(self,'info_timing'): save_dict['info_timing'] = self.info_timing
        if not self.value_NN is None:
            for k, v in save_dict['info']['best'].value_NN.items():
                save_dict['info']['best'].value_NN[k] = v.cpu()


        # c. networks
        save_dict['policy_NN'] = deepcopy(self.policy_NN).to('cpu').state_dict()
        if not self.value_NN is None:
            save_dict['value_NN'] = deepcopy(self.value_NN).to('cpu').state_dict()
        
        save_dict['torch_rng_state'] = self.torch_rng_state
        save_dict['torch_rng_state_final'] = torch.get_rng_state()

        # d. save
        with open(f'{filename}', 'wb') as f:
            pickle.dump(save_dict, f)

    #########################
    # solve - multiple GPUs #
    #########################

    solve_DDP = solve_DDP
    _solve_DDP_process = _solve_DDP_process