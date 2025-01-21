import os
import numpy as np
import torch

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from EconDLSolvers import choose_gpu
from BufferStockModel import BufferStockModelClass

############
# settings #
############

algonames = ['DeepSimulate','DeepFOC','DeepVPD','DeepV','DeepQ']

Ds = [0,3]

DO_MAIN = True
DO_EXPLORE = True
DO_FLOAT64 = True
DO_NEURONS = True
DO_FOC = True
DO_AVG = True
DO_AVG_FOC = True

K_time = 60
K = 100_000

#######
# run #
#######

def run_DL(D,algoname,folder=None,device=0,dtype=torch.float32,par=None,train=None,postfix=''):

    assert os.path.isdir(folder), f'Folder {folder} does not exist'
    if par is None: par = {}
    if train is None: train = {}

    print(f'Algorithm: {algoname}')
    print(f'Number of fixed states: {D}')
    if not postfix == '': print(f'Postfix: {postfix}')

    # a. setup
    model = BufferStockModelClass(algoname=algoname,device=device,dtype=dtype,par={'Nstates_fixed':D,**par},train=train)

    # b. solving model
    print('Solving')
    model.solve(do_print=False)
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

    # f. save model
    print('Saving')
    filename = f'{folder}/BufferStockModel_{algoname}_{D}D{postfix}.pkl'
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
# test #
########

D = 0
algoname = 'DeepVPD'
postfix = '_test'

train = {'K':K,'K_time':0.5}

print('')
print('#############################################')
print('')

run_DL(D,algoname,folder=folder,device=device,train=train,postfix=postfix)

########
# main #
########


if DO_MAIN:

    for D in Ds:
        for algoname in algonames:

            if algoname == 'DeepQ' and D > 0: continue
            if algoname == 'DeepV' and D > 0: continue

            print('')
            print('#############################################')
            print('')

            train = {'K':K,'K_time':K_time,'do_sim_eps':D==0}
            
            try:
                run_DL(D,algoname,folder=folder,device=device,train=train)
                print('')
            except Exception as e:
                print(e)
                print(f'Failed for {algoname} and {D = }')                           

###########
# explore #
###########

print('')
print('#############################################')
print('')

if DO_EXPLORE:

    D = 0
    epsilon_sigma = 0.4

    for algoname in ['DeepVPD','DeepFOC']:
        postfix = f'_moreexplore'
        train = {'K':K,'K_time':K_time,'epsilon_sigma':np.array([epsilon_sigma]),'do_sim_eps':True}

        try:
            run_DL(D,algoname,folder=folder,device=device,train=train,postfix=postfix)
            print('')
        except Exception as e:
            print(e)
            print(f'Failed for {algoname} {postfix} and {D = }')           

#########
# extra #
#########

if DO_FLOAT64:

    algoname = 'DeepVPD'
    D = 3
    postfix = '_float64'
    train = {'K':K,'K_time':K_time}

    try:
        run_DL(D,algoname,folder=folder,device=device,dtype=torch.float64,train=train,postfix=postfix)
        print('')
    except Exception as e:
        print(e)
        print(f'Failed for {algoname} {postfix} and {D = }')     

if DO_NEURONS:

    algoname = 'DeepVPD'
    Nneurons_vec = [250,750]

    for D in Ds:
        for i,Nneurons in enumerate(Nneurons_vec):

            postfix = f'_Nneurons{i}'
            train = {'K':K,'K_time':K_time,'Nneurons_policy':np.array([Nneurons,Nneurons]),'Nneurons_value':np.array([Nneurons,Nneurons])}
            try:
                run_DL(D,algoname,folder=folder,device=device,train=train,postfix=postfix)
                print('')
            except Exception as e:
                print(e)
                print(f'Failed for {algoname} {postfix} and {D = }')       

if DO_FOC:

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