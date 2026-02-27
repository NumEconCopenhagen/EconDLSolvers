import os
import numpy as np
import torch

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from EconDLSolvers import choose_gpu
from BufferStockModel import BufferStockModelClass, BufferStockMoreShocksModelClass

############
# settings #
############

algonames = ['DeepSimulate','DeepFOC']

Ds = [0,3]

DO_TEST = True
DO_MAIN = True
DO_EXPLORE = True
DO_KKT = True
DO_MC = True
DO_HYPERPAR = True
DO_MC_MORESHOCKS = True
DO_BACKWARD_SHORT = True
DO_BACKWARD_PURE = True
DO_CONTINUOUS_TIME = True

K_time_main = 60
K_time_rest = K_time_main
K_time_backward_short = 10
K_time_backward_refine = 2*K_time_main
K_time_backward_pure = 4*K_time_main

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

    if not model.train.backward:

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

    filename = f'{folder}/BufferStockModel_{algoname}_{D}D{postfix}.pt'
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

if DO_TEST:

    D = 0
    algoname = 'DeepSimulate'
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

            print('')
            print('#############################################')
            print('')

            do_sim_eps = D == 0 and not algoname == 'DeepSimulate'
            train = {'K':K,'K_time':K_time_main,'do_sim_eps':do_sim_eps}
            
            try:
                run_DL(D,algoname,folder=folder,device=device,train=train)
                print('')
            except Exception as e:
                print(e)
                print(f'Failed for {algoname} and {D = }')                           

###########
# explore #
###########

if DO_EXPLORE:

    print('')
    print('#############################################')
    print('')

    D = 0
    epsilon_sigma = 0.4

    for algoname in algonames:

        print('')
        print('#############################################')
        print('')

        postfix = f'_moreexplore'
        do_sim_eps = not algoname == 'DeepSimulate'
        train = {'K':K,'K_time':K_time_rest,'epsilon_sigma':torch.tensor([epsilon_sigma]),'do_sim_eps':do_sim_eps}

        try:
            run_DL(D,algoname,folder=folder,device=device,train=train,postfix=postfix)
            print('')
        except Exception as e:
            print(e)
            print(f'Failed for {algoname} {postfix} and {D = }')

#######
# KKT #
#######

if DO_KKT:

    print('')
    print('#############################################')
    print('')

    algoname = 'DeepFOC'
    for D in Ds:

        postfix = f'_KKT'
        par = {'KKT':True}
        train = {'K':K,'K_time':K_time_rest}

        print('')
        print('#############################################')
        print('')

        try:
            run_DL(D,algoname,folder=folder,device=device,par=par,train=train,postfix=postfix)
            print('')
        except Exception as e:
            print(e)
            print(f'Failed for {algoname} {postfix} and {D = }')            

######
# MC #
######

if DO_MC:

    print('')
    print('#############################################')
    print('')


    algoname = 'DeepFOC'
    for D in Ds:

        print('')
        print('#############################################')
        print('')

        postfix = f'_MC'
        train = {'K':K,'K_time':K_time_rest,'use_quad':False}

        try:
            run_DL(D,algoname,folder=folder,device=device,train=train,postfix=postfix)
            print('')
        except Exception as e:
            print(e)
            print(f'Failed for {algoname} {postfix} and {D = }')               

if DO_MC_MORESHOCKS:

    print('')
    print('#############################################')
    print('')

    def run_DL_MoreShocks(algoname,folder=None,device=0,dtype=torch.float32,par=None,train=None,postfix=''):

        assert os.path.isdir(folder), f'Folder {folder} does not exist'
        if par is None: par = {}
        if train is None: train = {}

        print(f'Algorithm: {algoname}')
        if not postfix == '': print(f'Postfix: {postfix}')

        # a. setup
        model = BufferStockMoreShocksModelClass(algoname=algoname,device=device,dtype=dtype,par={'Nstates_fixed':0,**par},train=train)

        # b. solving model
        print('Solving')
        model.solve(do_print=False)
        model.show_info()

        # c. save model
        print('Saving')
        filename = f'{folder}/BufferStockMoreShocksModel_{algoname}_{postfix}.pt'
        model.save(filename)

    # run quadrature
    print('')
    print('Quadrature with complicated shock process')
    par = {'full':True}
    train = {'use_quad':True,'K_time':K_time_rest}
    run_DL_MoreShocks('DeepFOC',folder=folder,device=device,par=par, train=train,postfix='quad')
    
    # run basic MC
    print('')
    print('basic MC with complicated shock process')

    par = {'full':True,'do_antithetic_mc':False,'Nmc': 200}
    train = {'use_quad':False,'K_time':K_time_rest,'redraw_mc':False}
    run_DL_MoreShocks('DeepFOC',folder=folder,device=device,par=par, train=train,postfix='mc')

    # run basic MC with redraw
    print('')
    print('basic MC with redraw with complicated shock process')

    par = {'full':True, 'do_antithetic_mc':False,'Nmc': 200}
    train = {'use_quad':False,'K_time':K_time_rest,'redraw_mc':True}
    run_DL_MoreShocks('DeepFOC',folder=folder,device=device,par=par, train=train,postfix='mcre')
    
    # run mc with antithetic
    print('')
    print('antithetic MC with redraw with complicated shock process')

    par = {'full':True,'do_antithetic_mc':True,'Nmc': 200}
    train = {'use_quad':False,'K_time':K_time_rest,'redraw_mc':True}
    run_DL_MoreShocks('DeepFOC',folder=folder,device=device,par=par, train=train,postfix='mcanti')

    # run mc with antithetic and QMC
    print('')
    print('antithetic QMC with complicated shock process')

    par = {'full':True,'do_antithetic_mc':True,'Nmc': 200,'use_sobol':True}
    train = {'use_quad':False,'K_time':K_time_rest,'redraw_mc':False}
    run_DL_MoreShocks('DeepFOC',folder=folder,device=device,par=par, train=train,postfix='qmcanti')

######
# HP #
######

# varying hyperparameters

if DO_HYPERPAR:

    for D in Ds:
        for algoname in algonames:
            for Nneurons in [400,600]:

                print('')
                print('#############################################')
                print('')
                
                postfix = f'_HyperParNneurons{Nneurons}'
                train = {'K':K,'K_time':K_time_rest,'Nneurons_policy':np.array([Nneurons,Nneurons])}
                
                try:
                    run_DL(D,algoname,folder=folder,device=device,train=train,postfix=postfix)
                    print('')
                except Exception as e:
                    print(e)
                    print(f'Failed for {algoname} and {D = }') 

############
# backward #
############

if DO_BACKWARD_SHORT:

    D = 3

    print('')
    print('#############################################')
    print('')

    # a. short simultanous
    algoname = 'DeepFOC'
    train = {'K':K,'K_time':K_time_backward_short}
    
    try:
        run_DL(D,algoname,folder=folder,device=device,train=train,postfix='_short')
        print('')
    except Exception as e:
        print(e)
        print(f'Failed for {algoname} and {D = }')    

    # b. backward
    algoname = 'DeepFOCBackward'
    filename = f'../output/BufferStockModel_DeepFOC_{D}D_short.pt'
    model_simult = BufferStockModelClass(load=filename,device=device)
    
    train = {
        'K':K,
        'K_time':K_time_backward_refine,
        'Neurons_policy_t':np.array([50,50]),
        'N':1_000_000,
        'batch_size':100_000,
    }

    try:
        run_DL(D,algoname,folder=folder,device=device,train=train,model_simult=model_simult,postfix='_short')
        print('')
    except Exception as e:
        print(e)
        print(f'Failed for {algoname} and {D = }')    

if DO_BACKWARD_PURE:

    D = 0
    algoname = 'DeepFOCBackward'

    print('')
    print('#############################################')
    print('')

    train = {
        'K':K,
        'K_time':K_time_backward_pure,
        'use_simult_in_backward': False,
        'Neurons_policy_t':np.array([50,50]),
        'N':1_000_000,
        'batch_size':100_000,
    }

    try:
        run_DL(D,algoname,folder=folder,device=device,train=train,postfix='_pure')
        print('')
    except Exception as e:
        print(e)
        print(f'Failed for {algoname} and {D = }')

if DO_CONTINUOUS_TIME:

    print('')
    print('#############################################')
    print('')

    D = 0
    for algoname in algonames:

        print('')
        print('#############################################')
        print('')

        postfix = f'_conttime'
        train = {'K':K,'K_time':K_time_rest,'time_input_type':'manual','Ninputs_time':2}

        try:
            run_DL(D,algoname,folder=folder,device=device,train=train,postfix=postfix)
            print('')
        except Exception as e:
            print(e)
            print(f'Failed for {algoname} {postfix} and {D = }')

        if algoname == 'DeepFOC':

            postfix = f'_conttimelowlr'
            train['learning_rate_policy'] = 1e-4

            try:
                run_DL(D,algoname,folder=folder,device=device,train=train,postfix=postfix)
                print('')
            except Exception as e:
                print(e)
                print(f'Failed for {algoname} {postfix} and {D = }')                        