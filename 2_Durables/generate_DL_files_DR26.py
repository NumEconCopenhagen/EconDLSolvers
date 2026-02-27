import os
import numpy as np
import torch

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from EconDLSolvers import choose_gpu
from DurablesModel import DurablesModelClass

def run_DL(D,algoname,folder=None,device=None,par=None,train=None,postfix='',model_simult=None):
    """ Run the model for a given algorithm and number of durables """ 

    assert os.path.isdir(folder), f'Folder {folder} does not exist'

    print(f'Algorithm: {algoname}')
    print(f'Number of durables: {D}')
    if not postfix == '': print(f'Postfix: {postfix}')
    
    # a. setup
    KKT = (algoname in ['DeepFOC','DeepFOCBackward']) or (algoname == 'DeepVPD' and 'use_FOC' in train and train['use_FOC'])
    if par is None: par = {}
    par = {**par,'D':D,'KKT':KKT}
    if train is None: train = {}
    model = DurablesModelClass(algoname=algoname,device=device,par=par,train=train)

    # b. solving model
    print('Solving')
    model.solve(do_print=False,model_simult=model_simult)

    if not model.train.backward:

        model.show_info()

        # c. Euler
        print('Euler errors')
        model.compute_euler_errors()

        # d. compute MPC and MPX
        print('Computing MPC and MPX')
        model.compute_MPC_MPX()

    # e. save
    print('Saving')

    # save less data for backward methods
    if model.train.backward:
        for train_key in ['states','states_pd','shocks','outcomes','actions','reward']:
            model.train.__dict__[train_key] = None

    filename = f'{folder}/DurablesModel_{algoname}_{D}D{postfix}.pt'
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

K = 100_000

K_time_base = 120
K_time_backward = 2*120

DO_MAIN = True
DO_SOFTMAX = True
DO_BACKWARD = True

if DO_MAIN:
    for D in [1,2,3,8]:

        for algoname in ['DeepSimulate','DeepFOC']:

            if D in [1,2,3]:
                K_time = K_time_base
            elif D == 8:
                K_time = K_time_base*2

            train = {'K':K,'K_time':K_time,}

            print('')
            print('#############################################')
            print('')

            try:
                run_DL(D,algoname,folder=folder,device=device,train=train)
                print('')
            except Exception as e:
                print(e)
                print(f'Failed for {algoname} and {D = }')

###########
# softmax #
###########

if DO_SOFTMAX:

    D = 8
    for algoname in ['DeepSimulate','DeepFOC']:

        if D in [1,2,3]:
            K_time = K_time_base
        elif D == 8:
            K_time = K_time_base*2

        par = {'use_softmax':True}
        train = {'K':K,'K_time':K_time,}

        print('')
        print('#############################################')
        print('')

        try:
            run_DL(D,algoname,folder=folder,device=device,par=par,train=train,postfix='_softmax')
            print('')
        except Exception as e:
            print(e)
            print(f'Failed for {algoname} and {D = }')

############
# backward #
############

filename = f'{folder}/DurablesModel_DeepFOC_8D.pt'
if DO_BACKWARD and os.path.exists(filename):

    print('')
    print('#############################################')
    print('')

    algoname = 'DeepFOCBackward'
    D = 8

    # a. load previous model
    model_simult = DurablesModelClass(load=filename,device=device)

    train = {
        'K':K,
        'K_time':K_time_backward,
        'Nneurons_policy_t':np.array([100,100]),
        'N':1_000_000,
        'batch_size':100_000,
    }
   
    try:
        run_DL(D,algoname,folder=folder,device=device,train=train,model_simult=model_simult)
        print('')
    except Exception as e:
        print(e)
        print(f'Failed for {algoname} and {D = }')    

    del model_simult