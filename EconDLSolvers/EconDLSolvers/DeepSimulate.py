import torch

# for distributed training
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# local
from . import auxilliary as aux
from .simulate import simulate_DeepSimulate

scheduler_step = aux.scheduler_step

#########
# setup #
######### 

def setup(model):
	""" setup training parameters"""

	# unpack
	train = model.train

	# a. learning
	train.learning_rate_policy = 1e-3  # learning rate
	train.learning_rate_policy_decay = 1.0 # learning rate decay
	train.learning_rate_policy_min = 1e-5 # minimum learning rate

	# b. sample 
	train.only_initial_states_and_shocks = True # whether to only simulate initial states and shocks
	train.N = 5000 # number of samples
	train.buffer_memory = 1 # buffer_size = buffer_memory*N

	# c. solver and exploration
	train.sim_R_freq = 50 # simulation frequency
	train.k_min = 500 # minimum number of iterations
	train.Delta_k = 200 # number of iterations without improvement before termination


	train.start_train_policy = -1

	# d. not used
	train.batch_size  = None
	train.clip_grad_value = None
	train.Delta_epoch_policy = None
	train.Delta_epoch_value = None
	train.epoch_policy_min = None
	train.epoch_termination = None
	train.epoch_value_min = None
	train.epsilon_sigma = None
	train.epsilon_sigma_decay = None
	train.epsilon_sigma_min = None
	train.learning_rate_value = None
	train.learning_rate_value_decay = None
	train.learning_rate_value_min = None
	train.Nepochs_policy = None
	train.Nepochs_value = None
	train.Nneurons_value = None
	train.tau = None
	train.use_target_policy = None
	train.use_target_value = None	

def create_NN(model):
	""" create neural nets """

	aux.create_NN(model,value=False)

#########
# solve #
######### 

def update_NN(model):
	""" update neural net parameters """
	
	# a. unpack
	par = model.par
	train = model.train
	info = model.info

	# b. get initial conditions and shocks
	initial_states = train.states[0]
	shocks = train.shocks

	# c. update policy
	policy_loss = aux.policy_update_step(model,policy_loss_f,initial_states,shocks)
	
	# d. store
	info[('policy_loss',train.k)] = policy_loss
	info[('value_epochs',train.k)] = 0	
	info[('policy_epochs',train.k)] = 1

def policy_loss_f(model,initial_states,shocks):
	""" policy loss """
	
	policy_NN = model.policy_NN

	policy_loss = simulate_DeepSimulate(model,policy_NN,initial_states,shocks)
	
	return policy_loss

###################################
# solve with distributed training #
###################################

def update_NN_DDP(model,rank):
	""" update neural net parameters  - distributed version"""

	# a. unpack
	train = model.train
	device = train.device = rank
	dtype = train.dtype

	info = model.info

	# b. setup policy with DDP
	old_policy_NN = model.policy_NN.to(rank)
	model.policy_NN = DDP(old_policy_NN,device_ids=[rank]) # setup DDP model

	# c. transfer data to all other processes
	if rank == 0:
	
		initial_states_tensor = train.states.to(rank)
		shocks_tensor = train.shocks.to(rank)
	
	else:

		initial_states_tensor = torch.empty(train.states.shape,dtype=dtype,device=device)
		shocks_tensor = torch.empty(train.shocks.shape,dtype=dtype,device=device)
	
	dist.broadcast(initial_states_tensor,src=0)
	dist.broadcast(shocks_tensor,src=0)

	# d. split up batch
	batch_size = train.N//train.Ngpus
	
	initial_states_tensor_ = initial_states_tensor[0,rank*batch_size:(rank+1)*batch_size,:]
	shocks_tensor_ = shocks_tensor[:,rank*batch_size:(rank+1)*batch_size,:]

	# e. train model
	policy_loss = aux.policy_update_step(model,policy_loss_f,initial_states_tensor_,shocks_tensor_)

	if rank == 0:
		info[('policy_loss',train.k)] = policy_loss
		info[('value_epochs',train.k)] = 0	
		info[('policy_epochs',train.k)] = 1		

	# f. finalize
	model.policy_NN = old_policy_NN