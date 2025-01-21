import gc
import os
import time
import numpy as np

from DurablesModelEGM import DurablesModelEGMClass

def get_egms():

    # main
    egm_main = {}

    # rough
    egm_rough = {}
    egm_rough['Np'] = 50
    egm_rough['Nm'] = 50
    egm_rough['Nn'] = 50

    # fine
    egm_fine = {}
    egm_fine['Np'] = 100
    egm_fine['Nm'] = 150
    egm_fine['Nn'] = 150

    return egm_main,egm_rough,egm_fine

def get_memory_usage(model):

    nss = ['par','egm','sim']
    nbytes = 0
    for ns in nss:
        for key,val in model.__dict__[ns].__dict__.items():
            if type(val) is np.ndarray:
                nbytes += val.nbytes
    
    return nbytes/(10**9) # in gb

def solve_and_simulate(model):
    
    par = model.par
    egm = model.egm
    sim = model.sim

    # a. solve
    t0 = time.perf_counter()
    model.solve()
    t1 = time.perf_counter()
    model.info['time'] = t1-t0
    print(f'time: {t1-t0:.1f} secs')

    func_evals = np.mean(egm.sol_func_evals)
    print(f' func_evals: {func_evals:.1f}')
    
    for i in np.unique(egm.sol_flag):
        flag_share = np.mean(egm.sol_flag==i)
        print(f' flag {i}: {flag_share:5.1%}')

    # b. simulate
    model.simulate_R()
    R = sim.R
    print(f'{R = :12.8f}')  

    m = sim.states[...,0]
    p = sim.states[...,1]
    n = sim.states[...,2:]
    print(f' m in [{m.min():.2f}, {m.max():.2f}]')
    print(f' p in [{p.min():.2f}, {p.max():.2f}]')
    print(f' n in [{n.min():.2f}, {n.max():.2f}]')

    a = sim.states_pd[...,0]
    d = sim.states_pd[...,2:]
    print(f' a in [{a.min():.2f}, {a.max():.2f}]')
    print(f' d in [{d.min():.2f}, {d.max():.2f}]')

def run_DP(D,folder=None,postfix='',par=None,egm=None):

    assert os.path.isdir(folder), f'Folder {folder} does not exist'

    if par is None: par = {}
    if egm is None: egm = {}

    algoname = 'DP'
    print(f'Algorithm: {algoname}')
    print(f'Number of durables: {D}')

    # a. setup
    par['D'] = D
    model = DurablesModelEGMClass(par=par,egm=egm)    

    # b. compile
    print('Compiling')
    model.link_to_cpp(do_print=False)
    
    # c. solve and simulate model
    print('Solving and simulating')
    solve_and_simulate(model)

    model.cpp.delink()

    # d. compute R-transfer function
    print('Computing R-transfer function')
    model.compute_transfer_func()     

    # e. compute euler error
    print('Computing euler error')
    model.compute_euler_errors_DP()

    # f. save model
    print('Saving model')

    model.info['memory_usage'] = get_memory_usage(model)

    if D > 1:
        
        # remove large objects
        model.egm.sol_q = None
        model.egm.sol_w = None

        # not saving policy functions
        model.egm.sol_v = None
        model.egm.sol_vm = None
        model.egm.sol_m_pd_fac = None
        model.egm.sol_d1_fac = None
        model.egm.sol_d2_fac = None
        model.egm.sol_d3_fac = None
        model.egm.sol_func_evals = None
        model.egm.sol_flag = None
        model.egm.sol_c_keep = None
        model.egm.sol_w = None
        model.egm.sol_q = None
    
    filename = f'{folder}/DurablesModel_DP_{D}D{postfix}.pkl'
    model.save(filename)

if __name__ == '__main__':

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    ##########
    # folder #
    ##########

    print('')
    print('#############################################')
    print('')

    folder = '../output'
    print(f'Folder: {folder}')
    print('')

    ########
    # main #
    ########

    egm_main,egm_rough,egm_fine = get_egms()

    for D,egm,postfix in [
        (1,egm_main,''),
        (2,egm_main,''),
        (1,egm_fine,'_fine'),
        (2,egm_fine,'_fine'),
        (1,egm_rough,'_rough'),
        (2,egm_rough,'_rough'),
        (3,egm_rough,'_rough'),
        ]:

        gc.collect() # garbage collection

        print('#############################################')
        print('')
        print(f'D = {D}')
        for key in ['Np','Na','Nm','Nn','solver']:
            if key in egm:
                val = egm[key]
                print(f'{key} = {val}')
        if not postfix == '': print(f'Postfix: {postfix}')
        print('')

        try:
            run_DP(D,folder=folder,postfix=postfix,egm=egm)
        except Exception as e:
            print(f'Failed, {e}')

        print('')