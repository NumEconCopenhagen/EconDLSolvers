import os
import numpy as np
import torch

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from EconDLSolvers import choose_gpu
from BufferStockModel import BufferStockModelClass

############
# settings #
############

algonames = ['DeepSimulate','DeepFOC','DeepVPD']

Ds = [0]

DO_MAIN = True
DO_FOC = True
DO_AVG = True
DO_AVG_FOC = True
DO_BACKWARD = True
DO_BACKWARD_PURE = True

K_time = 60
K_time_backward = 60
K = 100_000

#######
# run #
#######

def run_DL(D,algoname,folder=None,device=0,dtype=torch.float32,par=None,train=None,postfix='',model_simult=None):

    assert os.path.isdir(folder), f'Folder {folder} does not exist'
    if par is None: par = {}
    if train is None: train = {}

    print(f'Algorithm: {algoname}')
    print(f'Number of fixed states: {D}')
    if not postfix == '': print(f'Postfix: {postfix}')

    # a. setup
    model = BufferStockModelClass(algoname=algoname,device=device,dtype=dtype,
                                  par={'Nstates_fixed':D,**par},train=train)

    # b. solving model
    print('Solving')
    model.solve(do_print=False,model_simult=model_simult)

    if algoname not in ['DeepFOCBackward','DeepVPDBackward']:

        model.show_info()

        # c. computing MPC
        print('Computing MPC')
        model.compute_MPC()

        # d. computing Euler error
        print('Computing Euler error')
        model.compute_euler_errors()

        # e. compute policy on grids
        from BufferStockModelEGM import BufferStockModelEGMClass
        model_DP = BufferStockModelEGMClass(par={'Nstates_fixed':0})
        c_func = lambda state,action: state[...,0]*(1-action[...,0]) 
        model.info['sol_con_grid'] = model.compute_policy_on_grids(model_DP.egm,c_func)

    else:

        R = model.sim.R.item()
        print(f'{R = :12.8f}')

    # f. save model
    print('Saving')

    # save less data for backward methods
    if model.train.backward:
        for train_key in ['states','states_pd','shocks','outcomes','actions','reward']:
            model.train.__dict__[train_key] = None

    filename = f'{folder}/BufferStockModelVPD_{algoname}_{D}D{postfix}.pt'
    model.save(filename)

############################
# find best GPU and folder #
############################

print('')
print('#############################################')
print('')

device = choose_gpu()

folder = '../output'
if not os.path.isdir(folder): os.mkdir(folder)

print(f'Folder: {folder}')

########
# main #
########

if DO_MAIN:

    for algoname in algonames:
        for D in Ds:

            print('')
            print('#############################################')
            print('')

            do_sim_eps = True if D == 0 and not algoname == 'DeepSimulate' else False            
            train = {'K':K,'K_time':K_time,'do_sim_eps':do_sim_eps}
            
            try:
                run_DL(D,algoname,folder=folder,device=device,train=train)
                print('')
            except Exception as e:
                print(e)
                print(f'Failed for {algoname} and {D = }')                           
         
#########
# extra #
#########

if DO_FOC:

    print('')
    print('#############################################')
    print('')    

    algoname = 'DeepVPD'
    postfix = '_FOC'
    train = {'K':K,'K_time':K_time,'use_FOC':True}

    for D in Ds:
        try:
            run_DL(D,algoname,folder=folder,device=device,train=train,postfix=postfix)
            print('')
        except Exception as e:
            print(e)
            print(f'Failed for {algoname} {postfix} and {D = }')

if DO_AVG:

    print('')
    print('#############################################')
    print('')

    algoname = 'DeepVPD'
    N_value_NN_vec = [3,5]

    for D in Ds:
        for N_value_NN in N_value_NN_vec:

            postfix = f'_NNs{N_value_NN}'
            train = {'K':K,'K_time':K_time,'N_value_NN':N_value_NN}
            try:
                run_DL(D,algoname,folder=folder,device=device,train=train,postfix=postfix)
                print('')
            except Exception as e:
                print(e)
                print(f'Failed for {algoname} {postfix} and {D = }')               

if DO_AVG_FOC:

    print('')
    print('#############################################')
    print('')

    algoname = 'DeepVPD'
    N_value_NN = 3

    for D in Ds:
        postfix = f'_NNs{N_value_NN}FOC'
        train = {'K':K,'K_time':K_time,'N_value_NN':N_value_NN,'use_FOC':True}
        try:
            run_DL(D,algoname,folder=folder,device=device,train=train,postfix=postfix)
            print('')
        except Exception as e:
            print(e)
            print(f'Failed for {algoname} {postfix} and {D = }')

############
# backward #
############

algoname = 'DeepVPDBackward'

if DO_BACKWARD:

    for D in Ds:

        print('')
        print('#############################################')
        print('')

        train = {
            'K':K,
            'K_time':K_time_backward,            
            'Neurons_policy_t':np.array([50,50]),
            'N':1_000_000,
            'N_target_batches':10,
            'batch_size':100_000,
        }

        filename = f'../output/BufferStockModelVPD_DeepVPD_{D}D.pt'
        model_simult = BufferStockModelClass(load=filename,device=device)
        
        try:
            run_DL(D,algoname,folder=folder,device=device,train=train,model_simult=model_simult)
            print('')
        except Exception as e:
            print(e)
            print(f'Failed for {algoname} and {D = }')    

        del model_simult

if DO_BACKWARD_PURE:

    for D in Ds:

        algoname = 'DeepVPDBackward'

        print('')
        print('#############################################')
        print('')

        train = {
            'K':K,
            'K_time':K_time_backward,
            'use_simult_in_backward': False,
            'Neurons_policy_t':np.array([50,50]),
            'N_target_batches':10,
            'N':1_000_000,
            'batch_size':100_000,
        }

        try:
            run_DL(D,algoname,folder=folder,device=device,train=train,postfix='_pure')
            print('')
        except Exception as e:
            print(e)
            print(f'Failed for {algoname} and {D = }')    