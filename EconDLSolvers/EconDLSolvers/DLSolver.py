from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import time
import torch

# local
from . import neural_nets
from . import replay_buffer
from . import save_load
from . import timings
from . import figs
from .DDP import solve_DDP, _solve_DDP_process
from .simulate import simulate
from .manual_termination import solving_json, check_solving_json

from . import DeepSimulate
from . import DeepFOC
from . import DeepFOCBackward
from . import DeepVPD
from . import DeepVPDDC
from . import DeepVPDBackward
from . import DeepVPDDCBackward

algos = {
    'DeepSimulate': DeepSimulate,
    'DeepFOC': DeepFOC,
    'DeepFOCBackward': DeepFOCBackward,
    'DeepVPD': DeepVPD,
    'DeepVPDDC': DeepVPDDC,
    'DeepVPDBackward': DeepVPDBackward,
    'DeepVPDDCBackward': DeepVPDDCBackward
}
    
class DLSolverClass():
    """ generic solution algorithm """

    #########
    # setup #
    #########

    def __init__(self,
              algoname=None,device=None,dtype=None,par=None,sim=None,train=None,
              torch_rng_state=None,load=None):
        """ initialize """

        assert device is not None, 'device must be specified'            
        # check if device is a tuple with 2 elements
        if isinstance(device, tuple):
            device,device_name = device
        else:
            device_name = None

        if load is not None:

            assert algoname is None, 'algoname must be None when loading'
            assert dtype is None, 'dtype must be None when loading'
            assert par is None, 'par must be None when loading'
            assert train is None, 'train must be None when loading'
            assert torch_rng_state is None, 'torch_rng_state must be None when loading'
            
            with open(f'{load}', 'rb') as f:
                load_dict = torch.load(f,map_location=device,weights_only=False)

            algoname = load_dict['train'].algoname
            dtype = load_dict['train'].dtype
            par = load_dict['par'].__dict__
            train = load_dict['train'].__dict__

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
        self.train.device_name = device_name
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

        # checks
        parnames = ['T','Nstates','Nstates_pd','Nshocks','Noutcomes','Nactions','NDC'] 
        for name in parnames: assert hasattr(self.par,name), f'Parameter par.{name} must be specified'
        
        simnames = ['N','reps']  
        for name in simnames: assert hasattr(self.sim,name), f'Parameter sim.{name} must be specified'

        # d. setup and allocate train
        self._setup_train_default() # default parameters
        self.algo.setup(self) # overwrite with algo-specific parameters
        self.setup_train() # overwrite with model-specific parameters
        
        for k,v in train.items(): self.train.__dict__[k] = v # overwrite with call-specific parameters

        self.allocate_train()

        # checks
        assert hasattr(self.train,'policy_activation_final'), 'train.policy_activation_final must be specified'
        assert type(self.train.policy_activation_final) is list, 'train.policy_activation_final must be a list'
        assert len(self.train.policy_activation_final) == self.par.Nactions, 'train.policy_activation_final must have the same length as Nactions'
        
        assert hasattr(self.train,'min_actions'), 'train.min_actions must be specified'
        assert hasattr(self.train,'max_actions'), 'train.max_actions must be specified'
        
        assert len(self.train.min_actions) == self.par.Nactions, 'train.min_actions must have same length as Nactions'
        assert len(self.train.max_actions) == self.par.Nactions, 'train.max_actions must have same length as Nactions'

        if not self.train.epsilon_sigma is None:
            assert len(self.train.epsilon_sigma) == self.par.Nactions, 'train.epsilon_sigma must have same length as Nactions'
            assert len(self.train.epsilon_sigma_min) == self.par.Nactions, 'train.epsilon_sigma_min must have same length as Nactions'

        if algoname == 'DeepSimulate': assert not self.train.do_sim_eps, 'do_sim_eps = True is not implemented for DeepSimulate'
        if self.train.update_numint_weights: 
            assert not algoname == 'DeepFOC', 'update_numint_weights = True is not implemented for DeepFOC'

        # e. prepare simulation
        if load is None: self.prepare_simulate_R(torch_rng_state=torch_rng_state)

        # f. allocate DL solver

        # numerical integration
        if hasattr(self,'numerical_integration'): 
            self._numerical_integration()

        self.allocate_DLSolver()

        # g. overwrite
        if load is not None: save_load.load(self,load_dict)

        if not self.train.epsilon_sigma is None:
            self.train.epsilon_sigma = self.train.epsilon_sigma.to(device)
            self.train.epsilon_sigma_min = self.train.epsilon_sigma_min.to(device)        

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
        algo_names_optional = ['create_target_NN','update_NN_DDP','policy_loss_f', 'solve_backward']
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

        par = self.par
        train = self.train

        # a. neural networks and learning rates
        train.Nneurons_policy = np.array([500,500]) # size and number of layers
        train.policy_activation_intermediate = 'relu'

        train.value_activation_intermediate = 'relu'
        train.Nneurons_value = np.array([500,500]) # size and number of layers
        train.N_value_NN = None # number of value networks

        train.learning_rate_policy = 1e-3  # learning rate, lr_max
        train.learning_rate_policy_decay = 0.9999
        train.learning_rate_policy_min = 1e-5 # lr_min
        train.learning_rate_policy_schedule = None # {t_share:fac} time share with lr = fac*lr_max + (1-fac)*lr_min

        train.learning_rate_value = 1e-3 # learning rate
        train.learning_rate_value_decay = 0.9999
        train.learning_rate_value_min = 1e-5
        train.learning_rate_value_schedule = None # {t_share:fac} time share with lr = fac*lr_max + (1-fac)*lr_min

        train.manual_init_policy = False # whether to use manual initialization of policy NN

        # b. training sample
        train.N = 150 # sample
        train.buffer_memory = 8 # buffer_size = buffer_memory*N
        train.batch_size = train.N # batch size

        # c. replay buffer
        train.store_actions = False # store actions in replay buffer
        train.store_reward = False # store reward in replay buffer
        train.store_pd = False # store pd states in replay buffer
        train.i_t_index = False # include t index in buffer

        # d. exploration
        train.epsilon_sigma = None # std of exploration shocks
        train.epsilon_sigma_decay = 1.0 # decay of epsilon_sigma
        train.epsilon_sigma_min = None # minimum epsilon_sigma
        train.explore_frac = torch.zeros(par.T,dtype=train.dtype,device=train.device) # fraction of agents that explore in DeepSimulate

        # e. quadrature or monte carlo
        train.use_quad = True # whether to use quadrature or monte carlo for numerical integration (for DeepFOC and DeepVPD)
        train.redraw_mc = False # draw new mc shocks between iterations (when drawing new training sample)
        train.update_numint_weights = False # whether to update numerical integration weights with states_pd input        

        # f. smoothing
        train.start_train_policy = -1 # start training policy after this number of iterations
        train.tau = 0.2 # fixed tau / starting tau value
        train.tau_final = 1.0 # final tau value
        train.tau_schedule = None # {t_share:fac} time share with tau = (1-fac)*tau + fac*tau_final
        train.use_target_policy = True # use target in update policy
        train.use_target_value = True # use target in update value
        train.target_value_in_policy = False # use target value in policy
        
        # g. termination of iterations
        train.K = np.inf # number of iterations
        train.K_time = 60 # maximum time in minutes
        train.sim_R_freq = 10 # frequency of simulation of lifetime reward

        train.convergence_plot = False # plot convergence, else txt-file is written
        train.transfer_grid = np.arange(-10*1e-4,0+1e-4,1e-4) # transfer grid
        train.Delta_transfer = 1e-4 # size of transfer for convergence
        train.Delta_time = 20 # time without improvement before termination
        train.K_time_min = np.inf # minimum time in minutes

        train.terminate_on_policy_loss = False # terminate if policy loss is below tolerance
        train.tol_policy_loss = 1e-4 # tolerance for policy loss
        train.track_sim_policy_loss = False # track policy loss on sim (expensive)
        train.N_sample_policy_loss = None # how many samples batches when computing policy_loss on sim

        # h. termination of epochs
        train.epoch_termination = True # terminate epochs early if no improvement
        train.Nepochs_policy = 15 # number of epochs in update
        train.Delta_epoch_policy = 5 # number of epochs without improvement before termination
        train.epoch_policy_min = 5 # minimum number of epochs
        train.Nepochs_value = 50 # number of epochs in update
        train.Delta_epoch_value = 10 # number of epochs without improvement before termination
        train.epoch_value_min = 10 # minimum number of epochs

        # i. clipping gradient
        train.clip_grad_policy = 100.0 # clip policy
        train.clip_grad_value = 100.0 # clip value

        # j. backward algorithm
        train.backward = False # whether to use backward algorithm
        train.use_simult_in_backward = True # use simultaneous NN when training backward NN
        train.NN_init_std = 0.001 # std for low variance initialization of NN

        train.Nneurons_policy_t = None # number of neurons in policy NN at each time t
        train.learning_rate_policy_t = None # learning rate for policy NN at each time t
        train.learning_rate_policy_decay_t = None # learning rate decay for policy NN at each time t
        train.Nepochs_policy_t = None # number of epochs for policy NN at each time t

        train.Nneurons_value_t = None # number of neurons in value NN at each time t
        train.learning_rate_value_t = None # learning rate for value NN at each time t
        train.learning_rate_value_decay_t = None # learning rate decay for value NN at each time t
        train.Nepochs_value_t = None # number of epochs for value NN at each time t

        # k. using FOC (for DeepVPD and DeepVPDDC)
        train.use_FOC = False # use FOC in evaluation of policy
        train.NFOC_targets = 1 # number of FOC targets
        train.value_weight_val = 1.0 # weight on value loss	
        train.FOC_weight_val = 1.0 # weight on FOC loss
        train.value_weight_pol = 1.0 # weight on value loss	
        train.FOC_weight_pol = 1.0 # weight on FOC loss

        # l. time and transformations
        train.time_input_type = 'one_hot' # time as par.T dummies
        train.Ninputs_time = par.T # number of time inputs
        train.input_transformation = False # whether to use input transformation
        train.Ninputs_aux = 0 # number of auxiliary inputs
          
        # m. misc.
        train.terminal_actions_known = False # terminal actions are known
        train.NN_use_best = True # use best NN 
        train.epoch_use_best = True # use best epoch
        train.only_initial_states_and_shocks = False # only initial states and shock (e.g. in DeepSimulate)
        train.allow_synchronize = False # allow syncronize
        train.N_target_batches = 1 # number of batches when computing target
        train.do_sim_eps = False # simulate with exploration

        if torch.cuda.is_available():
            train.Ngpus = torch.cuda.device_count() # number of GPUs
        else:
            train.Ngpus = 0
        
        # n. other
        train.k = -1 # initialization

    def _numerical_integration(self):
        """ get quadrature points and weights """

        par = self.par
        train = self.train
        dtype = train.dtype
        device = train.device

        nodes,weigths = self.numerical_integration()

        # a. quadrature
        if train.use_quad:

            assert nodes.ndim == 2, 'nodes must be 2-dimensional'
            assert weigths.ndim == 1, 'weights must be 1-dimensional'

            train.Nnumint = nodes.shape[0]
            assert nodes.shape[0] == weigths.shape[0], 'nodes and weigths must have same length'
            assert nodes.shape[1] == self.par.Nshocks, 'nodes must have same length as Nshocks'
            
            if not train.update_numint_weights:
                assert torch.isclose(torch.sum(weigths),torch.tensor(1.0,device=weigths.device,dtype=weigths.dtype)), 'weigths must sum to 1.0'

        else: # for mc

            assert nodes.ndim == 4, 'nodes must be 2-dimensional'
            assert weigths.ndim == 1, 'weights must be 1-dimensional'

            train.Nnumint = nodes.shape[2]
            assert nodes.shape[2] == weigths.shape[0], 'mc draws and mc weights must have same length'

            assert nodes.shape[0] == self.par.T-1, 'mc draws must have same length as T-1'
            assert nodes.shape[1] == self.train.N, 'mc draws must have same length as N'
            assert nodes.shape[3] == self.par.Nshocks, 'mc draws must have same length as Nshocks'

            assert torch.isclose(torch.sum(weigths),torch.tensor(1.0,device=weigths.device,dtype=weigths.dtype)), 'weigths must sum to 1.0'

        train.numint_nodes = nodes.to(dtype).to(device)
        train.numint_weights = weigths.to(dtype).to(device)

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
        if not self.train.backward:
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
    eval_policy_t = neural_nets.eval_policy_t
    eval_value = neural_nets.eval_value
    eval_value_t = neural_nets.eval_value_t

    #################
    # evaluate mode #
    #################

    def _state_trans(self,states_pd):

        train = self.train

        # states_pd.shape = (T,N,Nstates_pd)

        T,N = states_pd.shape[:2]

        # a. add dimension for numerical integration
        if len(states_pd.shape) == 2:
            T,N = states_pd.shape
            states_pd = states_pd.reshape(T,N,1).repeat((1,1,train.Nnumint))
        else:
            T,N,X = states_pd.shape
            states_pd = states_pd.reshape(T,N,1,X).repeat((1,1,train.Nnumint,1))
            
        # b. add dimension T and N dimensions in quadrature
        if train.use_quad:
            shocks = train.numint_nodes.tile((T,N,1,1)) # (T-1,N,Nnumint,Nshocks)
        else:
            shocks = train.numint_nodes # (T-1,N,Nnumint,Nshocks)

        states_plus = self.state_trans(states_pd,shocks,t=None) # shape = (T,N,Nnumint,Nstates)
        
        return states_plus
    
    def _state_trans_t(self,states_pd,t):

        train = self.train

        # states_pd.shape = (N,Nstates_pd)

        N,X = states_pd.shape

        # a. add dimension for numerical integration
        states_pd = states_pd.reshape(N,1,X).repeat((1,train.Nnumint,1))

        # b. add dimension T and N dimensions in quadrature
        if train.use_quad:
            shocks = train.numint_nodes.tile((N,1,1))# (N,Nnumint,Nshocks)
        else:
            shocks = train.numint_nodes # (N,Nnumint,Nshocks)

        states_plus = self.state_trans(states_pd,shocks,t=t) # shape = (N,Nnumint,Nstates)
        
        return states_plus
        
    ###################
    # training sample #
    ###################

    def _simulate_training_sample(self,epsilon_sigma,device=None):
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
            
            # ii. simulate
            simulate(self,train,eps=eps)

            # iii. store in buffer
            if not train.backward:
                self.rep_buffer.add(train)

        if 'time._simulate_training_sample' in self.info: 
            self.info['time._simulate_training_sample'] += time.perf_counter() - t0	

        # c. update mc-shocks between iterations
        if train.redraw_mc:
            self._numerical_integration()

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
    
    def _update_best(self,final=False):
        """ update best R and neural nets """

        t0 = time.perf_counter()

        sim = self.sim
        train = self.train
        info = self.info

        best = info['best']

        # a. R
        self.simulate_R()
        info[('R',train.k)] = sim.R.item()
        if hasattr(self,'compute_model_moments'): self.compute_model_moments()

        # b. policy loss
        if train.algoname == 'DeepFOC' and train.track_sim_policy_loss:
            states = sim.states
            if train.terminal_actions_known:
                states = states[:-1]
            policy_loss = 0.0
            N_sample = round(sim.N/train.N_sample_policy_loss)
            assert sim.N % train.N_sample_policy_loss == 0, 'N_sample_policy_loss must divide N'
            for i in range(train.N_sample_policy_loss):
                states_sample = states[:,i*N_sample:(i+1)*N_sample]
                with torch.no_grad():
                    policy_loss += self.algo.policy_loss_f(self,states_sample).item()
            policy_loss /= train.N_sample_policy_loss
            info[('policy_loss_sim',train.k)] = policy_loss
        
        # b. transfer if best
        if sim.R.item() > best.R: 

            best.k = train.k
            best.time = self.info[('k_time',train.k)]
            best.R = sim.R.item()
            
            best.policy_NN = deepcopy(self.policy_NN.state_dict())
            best.value_NN = deepcopy(self.value_NN.state_dict()) if not self.value_NN is None else None
            if not self.train.N_value_NN is None:
                best.value_NNs = [deepcopy(self.value_NNs[i].state_dict()) for i in range(self.train.N_value_NN)]

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
        print(f'{k = :5d} of {train.K-1}: sim.R = {R:14.8f} [best: {best.R:14.8f}] [{time_k:.1f} secs] [{value_epochs = :3d}] [{policy_epochs = :3d}] [{time_tot:6.2f} mins]',end='')
        
        if train.terminate_on_policy_loss and k >= train.start_train_policy: 
            policy_loss = info[('policy_loss',k)]
            print(f' [{policy_loss = :.8f}]',end='')

        policy_lr = self.policy_opt.param_groups[-1]['lr'] # assuming no differential learning rates
        print (f' [{policy_lr = :.1e}]',end='')

        if hasattr(self,'value_opt') or not self.train.N_value_NN is None:
            if self.train.N_value_NN is None:
                value_lr = self.value_opt.param_groups[-1]['lr'] # assuming no differential learning rates
                print (f' [{value_lr = :.1e}]',end='')
            else:
                value_lrs = [self.value_opts[j].param_groups[-1]['lr'] for j in range(self.train.N_value_NN)]
                value_lr = sum(value_lrs)/self.train.N_value_NN
                print (f' [{value_lr = :.1e}]',end='')
        
        if not train.tau_schedule is None:
            tau = info[('tau',k)]
            print (f' [{tau = :.2f}]',end='')

        if not np.isnan(R):
            if best.k == k:
                print('')
            else:
                print(f' no improvements in {time_tot-best.time/60:.1f} mins')
        else:
            print('')

    def solve(self,do_print=False,do_print_all=False,do_print_terminate=True,model_simult=None,postfix=''):
        """ solve model """ 

        timestamp = solving_json(postfix)
        t0_solve = time.perf_counter()

        if do_print_all: do_print = True
        if do_print: print(f"started solving: {timestamp}")

        # a. unpack
        sim = self.sim
        train = self.train
        info = self.info
        info['t0_solve'] = t0_solve
        
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

        # b. backwars
        if train.backward:
            self._solve_backward(do_print=do_print,model_simult=model_simult)
            return
        
        # c. initialize best
        if not continued_solve:

            best = info['best'] = SimpleNamespace()
            best.k = -1		
            best.time = 0.0
            best.R = -np.inf
            best.policy_NN = None
            best.value_NN = None
            best.value_NNs = None

        else:

            best = info['best']
            
        # c. loop over iterations
        if not continued_solve:
            epsilon_sigma = info['epsilon_sigma'] = deepcopy(train.epsilon_sigma)
            train.k = info['iter'] = 0
        else:
            epsilon_sigma = info['epsilon_sigma']
            train.k = info['iter']

        while True:
            
            t0_k = time.perf_counter()
            k = train.k

            info[('value_epochs',k)] = 0 # keep track of number of value epochs (updated in algo)
            info[('policy_epochs',k)] = 0 # keep track of number of policy epochs (updated in algo)

            # i. simulate training sample
            self._simulate_training_sample(epsilon_sigma)
            
            # update exploration
            if epsilon_sigma is not None:
                epsilon_sigma *= train.epsilon_sigma_decay
                epsilon_sigma = torch.fmax(epsilon_sigma,train.epsilon_sigma_min)

            # ii. update neural nets
            t0 = time.perf_counter()
            self.algo.update_NN(self) # is different for each algorithm
            info['time.update_NN'] += time.perf_counter() - t0

            # iii. scheduler step
            t0 = time.perf_counter()
            self.algo.scheduler_step(self) # is different for each algorithm      
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
            
            info['iter'] = k + 1
            
            ###############
            # termination #
            ###############

            # v. policy loss
            if k >= train.start_train_policy:
                if train.terminate_on_policy_loss and info[('policy_loss',k)] < train.tol_policy_loss:
                    self._update_best(final=True) # final update
                    k += 1
                    if do_print_terminate: print(f'Terminating after {k} episodes, policy loss lower than tolerance')
                    break
                
            # vi. time
            time_tot = (time.perf_counter()-t0_solve)/60 + info['time']/60
            if time_tot > train.K_time:
                self._update_best(final=True) # final update
                k += 1
                if do_print_terminate: print(f'Terminating after {k} episodes, max time {train.K_time} mins reached')
                break
                
            # vii. manual
            manual_termination = check_solving_json(timestamp)
            if manual_termination:
                self._update_best(final=True) # final update
                k += 1
                if do_print_terminate: print(f'Terminating after {k} episodes, manual termination')
                break

            # vii. episodes
            if train.k+1 >= train.K: 
                self._update_best(final=True) # final update
                if do_print_terminate: print(f'Terminating after {train.k+1} episodes, max number of episodes reached')
                break            

            train.k += 1

        # e. load best solution
        t0 = time.perf_counter()

        if train.NN_use_best:

            if self.policy_NN is not None: 
                if not best.policy_NN is None:
                    self.policy_NN.load_state_dict(best.policy_NN)
            
            if self.value_NN is not None: 
                if not best.value_NN is None:
                    self.value_NN.load_state_dict(best.value_NN)

            if self.train.N_value_NN is not None:
                if not best.value_NNs is None:
                    for i in range(self.train.N_value_NN):
                        self.value_NNs[i].load_state_dict(best.value_NNs[i])
                    
        info['time.update_best'] += time.perf_counter() - t0

        # f. final simulation
        t0 = time.perf_counter()
        self.simulate_R()
        info['time.update_best'] += time.perf_counter() - t0

        # g. store
        info['R'] = sim.R.item() # best.R if train.NN_use_best else info[('R',train.k)]
        info['best_R'] = best.R
    
        info['time'] += time.perf_counter()-t0_solve
        info[('value_epochs','sum')] = np.sum([info[('value_epochs',k)] for k in range(info['iter'])])
        info[('policy_epochs','sum')] = np.sum([info[('policy_epochs',k)] for k in range(info['iter'])])
        info[('value_epochs','mean')] = np.mean([info[('value_epochs',k)] for k in range(info['iter'])])
        info[('policy_epochs','mean')] = np.mean([info[('policy_epochs',k)] for k in range(info['iter'])])

        if do_print: self.show_info()

        # h. extra: multiples simulations of R
        if do_print and self.sim.reps > 0: print('Simulating multiple Rs')
        Rs = self.simulate_Rs()
        info['Rs'] = Rs
    
        # i. extra: simulation with epsilon shocks
        if train.do_sim_eps and not epsilon_sigma is None:

            if do_print: print('Simulating with exploration')

            self.sim_eps = deepcopy(sim)
            eps = self.draw_exploration_shocks(epsilon_sigma,self.sim_eps.N).to(train.dtype).to(train.device)
            simulate(self,self.sim_eps,eps=eps) # note: same initial states and shocks as in .sim are used

        else:

            self.sim_eps = None
        
        # j. empty cache
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    def set_time_per_t(self,value=False):

        train = self.train
        info = self.info

        time_left = 60*train.K_time-info['time']
        T = self.par.T-1 if train.terminal_actions_known else self.par.T
        if value: T *= 2
        info['time_per_t'] = time_left / T

    def _solve_backward(self,do_print=False,model_simult=None):
        """ solve backward """

        train = self.train
        info = self.info

        if self.train.use_simult_in_backward and model_simult is None:
            raise ValueError('model_simult must be provided when use_simult_in_backward = True')

        # a. solve
        self.algo.solve_backward(self,do_print=do_print,model_simult=model_simult)

        # b. simulate
        self.simulate_R()
        info['R'] = self.sim.R.item()
        self.simulate_Rs()

        # hc extra: multiples simulations of R
        if do_print and self.sim.reps > 0: print('Simulating multiple Rs')
        Rs = self.simulate_Rs()
        info['Rs'] = Rs

        info['time'] += time.perf_counter()-info['t0_solve']    

    def show_info(self):
        
        R = self.info['R']
        iter = self.info['iter']
        policy_epochs = self.info[('policy_epochs','mean')]
        value_epochs = self.info[('value_epochs','mean')]
        time_tot = self.info['time']
    
        print(f'{R = :.4f}, time = {time_tot/60:.1f} mins, iter = {iter}, policy epochs = {policy_epochs:.2f}, value epochs = {value_epochs:.2f}')

    time_solve = timings.time_solve
    
    #############
    # save-load #
    #############

    save = save_load.save

    #########################
    # solve - multiple GPUs #
    #########################

    solve_DDP = solve_DDP
    _solve_DDP_process = _solve_DDP_process