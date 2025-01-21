from types import SimpleNamespace
import numpy as np
import torch
from torch.utils.data import DataLoader



class ReplayBuffer():
	""" Buffer to store observations."""

	def __init__(self,
			  Nstates,Nstates_pd,Nactions,T,N,
			  dtype,memory=1,device=None,
			  store_actions=True,store_reward=True,store_pd=True,i_t_index=False):
		
		""" Initialize """

		assert device is not None, 'device must be specified'
		assert memory >= 1, 'memory must be at least 1'

		# a. general attributes
		self.ptr = 0
		self.curr_size = 0
		self.T = T
		self.N = N
		self.buffer_size = N*memory
		self.dtype = dtype
		self.device = device
		self.store_actions = store_actions
		self.store_reward = store_reward
		self.store_pd = store_pd
		self.i_t_index = i_t_index

		# b. elements of the buffer
		self.Nstates = Nstates
		self.Nactions = Nactions
		self.states = torch.zeros((T,self.buffer_size,Nstates),dtype=dtype,device=device) # state buffer
		if self.store_pd: self.states_pd = torch.zeros((T,self.buffer_size,Nstates_pd),dtype=dtype,device=device) # state buffer
		if self.store_actions: self.actions = torch.zeros((T,self.buffer_size,Nactions),dtype=dtype,device=device) # action buffer
		if self.store_reward: self.reward = torch.zeros((T,self.buffer_size),dtype=dtype,device=device) # reward buffer
		
	def add(self,train):
		""" Add observations """

		# a. insert
		i0 = self.ptr
		iend = i0 + self.N

		self.states[:,i0:iend] = train.states
		if self.store_pd: self.states_pd[:,i0:iend] = train.states_pd
		if self.store_actions: self.actions[:,i0:iend] = train.actions
		if self.store_reward: self.reward[:,i0:iend] = train.reward

		# b. update pointer and size
		self.ptr = (self.ptr + self.N) % self.buffer_size
		self.curr_size = min(self.curr_size + self.N, self.buffer_size)

	def sample(self,batch_size):
		""" Sample a batch of transitions """

		device = self.device

		# a. return the sampled transitions as torch tensors
		batch = SimpleNamespace()

		# b. get dimensions for slicing
		Nrows = self.T
		Ncols = self.curr_size

		# c. create row index
		row_indices = torch.arange(Nrows).reshape(-1,1).to(device) # (T,1)

		# d. create column index
		if self.i_t_index:
			col_indices = torch.zeros((Nrows,batch_size),dtype=torch.long,device=device)
			for i in range(Nrows):
				col_indices[i,:] = torch.multinomial(torch.ones(Ncols),batch_size,replacement=False)
		else:
			col_indices = torch.multinomial(torch.ones(Ncols),batch_size,replacement=False).reshape(1,-1).to(device) # (1,N)

		# e. select
		batch.states = self.states[row_indices,col_indices]
		
		if self.store_pd: batch.states_pd = self.states_pd[row_indices,col_indices]
		if self.store_actions: batch.actions = self.actions[row_indices,col_indices]
		if self.store_reward: batch.reward = self.reward[row_indices,col_indices]

		return batch