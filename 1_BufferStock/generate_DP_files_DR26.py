import os
import time
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from BufferStockModelEGM import BufferStockModelEGMClass, BufferStockMoreShocksModelEGMClass

#######
# run #
#######

def get_memory_usage(model):

    nss = ['par','egm','sim']
    nbytes = 0
    for ns in nss:
        for key,val in model.__dict__[ns].__dict__.items():
            if type(val) is np.ndarray:
                nbytes += val.nbytes
    
    return nbytes/(10**9) # in gb

def run_DP(D,folder=None,save_sol=False,more_shocks=False):

    assert os.path.isdir(folder), f'Folder {folder} does not exist'

    algoname = 'DP'
    print(f'Algorithm: {algoname}')
    print(f'Number of fixed states: {D}')

    print(f"Running with {D} fixed states")

    # a. setup
    if more_shocks:
        assert D==0, "MoreShocks model only implemented for D=0"
        model = BufferStockMoreShocksModelEGMClass(par={'Nstates_fixed':D})
    else:
        model = BufferStockModelEGMClass(par={'Nstates_fixed':D})
    model.savefolder = folder

    model.info['memory'] = get_memory_usage(model)
    print(f'Memory: {model.info["memory"]:.1f} gb')

    # b. compile
    print('Compiling')
    model.link_to_cpp(do_print=False)

    # c. solve
    print('Solving')
    t0 = time.perf_counter()
    model.cpp.solve_all(model.par,model.egm)
    model.info['time'] = time.perf_counter()-t0
    print(f'Time: {model.info["time"]:.2f} sec')

    model.cpp.delink()

    # c. simulate
    print('Simulating')
    model.simulate_R(final=True)
    print(f'R = {model.sim.R:12.8f}')

    m = model.sim.states[...,0]
    p = model.sim.states[...,1]
    print(f' m in [{m.min():.2f}, {m.max():.2f}]')
    print(f' p in [{p.min():.2f}, {p.max():.2f}]')

    m_pd = model.sim.states_pd[...,0]
    print(f' m_pd in [{m_pd.min():.2f}, {m_pd.max():.2f}]')

    # d. compute R-transfer function
    print('Computing R-transfer function')
    model.compute_transfer_func()
        
    # e. euler error
    print('Computing Euler-errors')
    model.compute_euler_errors()

    # f. multiple Rs and transfer functions
    print('Computing multiple R and transfer functions')
    model.compute_transfer_funcs()
    
    # f. save model
    print('Saving model')

    if not save_sol: # remove large objects
        model.egm.sol_con = None
        model.egm.sol_w = None

    if more_shocks:
        filename = f'{folder}/BufferStockMoreShocksModel_DP_{D}D.pt'
    else:
        filename = f'{folder}/BufferStockModel_DP_{D}D.pt'
    
    model.save(filename)
    
    print('')
    
##########
# folder #
##########

print('')
print('#############################################')
print('')

folder = '../output'
print(f'Folder: {folder}')

########
# main #
########

for D in [0,3]:

    print('')
    print('#############################################')
    print('')

    run_DP(D,folder=folder,save_sol=D==0)

    if D==0:
        run_DP(D,folder=folder,save_sol=False,more_shocks=True)
