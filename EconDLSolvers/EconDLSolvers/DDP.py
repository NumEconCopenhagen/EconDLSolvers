from copy import deepcopy
import numpy as np
import os
import pickle
import time
import torch
import warnings

# for distributed training
import torch.multiprocessing
import torch.distributed
from torch.distributed import init_process_group, destroy_process_group

from .simulate import simulate

def solve_DDP(model,do_print=False,do_print_all=False,port=None):
	""" Solve model using Distributed Data Parallel"""

	t0_solve = time.perf_counter()

	# a. unpack
	train = model.train
	info = model.info	

	assert train.device in [0,'cuda:0'], 'the main device must be cuda:0'

	# disable epoch termination as it is complicated to implement in DDP
	if train.epoch_termination:
		print('Epoch termination is not implemented in DDP, setting to False')
		train.epoch_termination = False
	

	if train.terminate_on_policy_loss:
		print('Terminate on policy loss is not implemented in DDP, setting to False')
		model.terminate_on_policy_loss = False

	# c. call DDP solver function
	torch.multiprocessing.spawn(
		model._solve_DDP_process,args=(t0_solve,port,do_print,do_print_all),
		nprocs=train.Ngpus,join=True)

	# d. pickle load info
	info = pickle.load(open(f'info_pickle.pkl', 'rb'))
	model.info = info
	os.remove('info_pickle.pkl')

	# e. final simulation
	model.simulate_R()

	info['R'] = model.sim.R
	info['time'] = time.perf_counter() - t0_solve
	info[('value_epochs','mean')] = np.mean([info[('value_epochs',k)] for k in range(info['iter'])])
	info[('policy_epochs','mean')] = np.mean([info[('policy_epochs',k)] for k in range(info['iter'])])

	if do_print: model.show_info()

	# f. simulate with epsilon shocks
	if not train.epsilon_sigma is None:

		model.sim_eps = deepcopy(model.sim)
		eps = model.draw_exploration_shocks(train.epsilon_sigma,model.sim_eps.N).to(train.dtype).to(train.device)
		simulate(model,model.sim_eps,eps=eps)

def _solve_DDP_process(model,rank,t0_solve,port=None,do_print=False,do_print_all=False):
	""" Solving operation on a single process """

	# a. unpack and DDP setup
	par = model.par
	train = model.train
	info = model.info
	sim = model.sim
	policy_NN = model.policy_NN
	value_NN = model.value_NN
	
	ddp_setup(rank,train.Ngpus,port=port)
	model.info['time'] = 0.0
	model.info['time.update_NN'] = 0.0
	model.info['time.update_NN.train_value'] = 0.0
	model.info['time.update_NN.train_policy'] = 0.0
	model.info['time.scheduler'] = 0.0
	model.info['time.convergence'] = 0.0
	model.info['time.update_best'] = 0.0
	model.info['time._simulate_training_sample'] = 0.0
	model.info['time.simulate_R'] = 0.0
	# move tensors to rank device
	for k,v in par.__dict__.items():
		if isinstance(v,torch.Tensor): par.__dict__[k] = v.to(rank)

	for k,v in train.__dict__.items():
		if isinstance(v,torch.Tensor): train.__dict__[k] = v.to(rank)

	# b. preparation - only rank 0
	if rank == 0:

		model.prepare_simulate_R()

		# i. initialize best and 	
		best_k = -1
		best_R = -np.inf
		best_policy_NN = None 
		best_value_NN = None
		
	epsilon_sigma = deepcopy(train.epsilon_sigma)
	torch.distributed.barrier()
	
	# b. loop over iterations
	t0_loop = time.perf_counter()
	for k in range(train.K):

		train.k = k
		
		if rank == 0:
			t0_k = time.perf_counter()
			info[('value_epochs',k)] = 0
			info[('policy_epochs',k)] = 0			

		# i. create training sample
		do_exo_actions = False # k < train.do_exo_actions_periods

		if rank == 0:
			model._simulate_training_sample(epsilon_sigma,do_exo_actions)

		torch.distributed.barrier()

		# ii. update neural network
		model.algo.update_NN_DDP(model,rank)

		# iii. update exploration
		if epsilon_sigma is not None:
			epsilon_sigma *= train.epsilon_sigma_decay
			epsilon_sigma = np.fmax(epsilon_sigma,train.epsilon_sigma_min)

		# iv. scheduler step
		with warnings.catch_warnings(action="ignore"):
			model.algo.scheduler_step(model)	

		torch.distributed.barrier()

		# v. calculate R and save best - only rank 0
		if rank == 0:

			# o. compute sim_R
			if k % train.sim_R_freq == 0:

				model.simulate_R()

				info[('R',k)] = model.sim.R
				info[('k_time',k)] = time.perf_counter() - t0_solve
				
				# o. save best and update improvement counter
				if model.sim.R > best_R:
				
					best_k = k
					best_R = sim.R
					best_policy_NN = deepcopy(policy_NN.state_dict())
					best_value_NN = deepcopy(value_NN.state_dict()) if not value_NN is None else None

				# oo. print
				if do_print:
					
					time_k = time.perf_counter() - t0_k			
					time_tot = (time.perf_counter() - t0_solve)/60		
					print(f'{k = :5d} of {train.K-1}: {sim.R = :12.8f} [best: {best_R:12.8f}] [{time_k:.1f} secs] [{time_tot:4.1f} mins]',end='')

					if best_k == k:
						print('')
					else:
						print(f' no improvements in {k-best_k} iter')


			# oo. no compute sim_R
			else:

				info[('R',k)] = np.nan

				if do_print_all:
					
					time_k = time.perf_counter() - t0_k					
					time_tot = (time.perf_counter() - t0_solve)/60

					print(f'{k = :5d} of {train.K-1}: sim.R = {np.nan:12.8f} [best: {best_R:12.8f}] [{time_k:.1f} secs] [{time_tot:6.2f} mins]')

			# ooo. termination from time
			time_tot = (time.perf_counter() - t0_solve)/60
			if time_tot > train.K_time:
				model.simulate_R()

				if sim.R > best_R: 
					info[('R',k)] = sim.R
					best_policy_NN = deepcopy(policy_NN.state_dict())
					best_value_NN = deepcopy(value_NN.state_dict()) if not value_NN is None else None

				print(f'Terminating after {k+1} iter, max time {train.K_time} mins reached')
				break_statement = torch.tensor([1],dtype=torch.int32,device=rank)
			else:
				break_statement = torch.tensor([0],dtype=torch.int32,device=rank)
		else:

			break_statement = torch.tensor([0],dtype=torch.int32,device=rank)
		
		torch.distributed.barrier()
		torch.distributed.broadcast(tensor=break_statement,src=0) # broadcast break statement to all processes
		
		if break_statement:
			break

	# vi. load best solution - only rank 0
	if rank == 0:

		model.policy_NN.load_state_dict(best_policy_NN)
		if model.value_NN is not None: model.value_NN.load_state_dict(best_value_NN)
	
		# dump info to pickle file for storing info from rank 0
		model.info['time.loop'] = time.perf_counter() - t0_loop
		model.info['iter'] = k+1
		pickle.dump(model.info, open(f'info_pickle.pkl', 'wb'))

	# vii. cleanup
	destroy_process_group()

def ddp_setup(rank,world_size,port):
	"""
	Initializes the distributed environment for DDP

	Args:
		rank: Unique identifier of each process
		world_size: Total number of processes
	"""

	if port is None: port = 12345

	os.environ["MASTER_ADDR"] = "localhost" # set adress of master
	os.environ["MASTER_PORT"] = f"{port}" # set port of master

	init_process_group(backend="gloo",rank=rank,world_size=world_size) # initialize process group