
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from EconDLSolvers import choose_gpu
from DurablesModel import DurablesModelClass

def run_DL(D,algoname,folder=None,device=None,train=None,postfix=''):
    """ Run the model for a given algorithm and number of durables """ 

    assert os.path.isdir(folder), f'Folder {folder} does not exist'

    print(f'Algorithm: {algoname}')
    print(f'Number of durables: {D}')

    # a. setup
    KKT = (algoname == 'DeepFOC') or (algoname == 'DeepVPD' and 'use_FOC' in train and train['use_FOC'])
    par = {'D':D,'KKT':KKT}
    if train is None: train = {}
    model = DurablesModelClass(algoname=algoname,device=device,par=par,train=train)

    # b. solving model
    print('Solving')
    model.solve(do_print=False)
    model.show_info()

    # c. Euler
    print('Euler errors')
    model.compute_euler_errors()

    # c. save
    print('Saving')
    filename = f'{folder}/DurablesModel_{algoname}_{D}D{postfix}.pkl'
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

for D in [1,2,3,8]:

    for algoname in ['DeepSimulate','DeepFOC','DeepVPD']:

        if D in [1,2,3]:
            K_time = 2*60
        elif D == 8:
            K_time = 4*60

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

#########
# altLR #
#########

K = 100_000
algoname = 'DeepVPD'
K_time = 2*60

for D in [1,2,3]:

    train = {'K':K,'K_time':K_time,'learning_rate_value_decay':0.999,'learning_rate_policy_decay':0.999}
    postfix = '_altLR'  

    print('')
    print('#############################################')
    print('')

    try:
        run_DL(D,algoname,folder=folder,device=device,train=train,postfix=postfix)
        print('')
    except Exception as e:
        print(e)
        print(f'Failed for {algoname} and {D = }')   

#########
# extra #
#########

D = 8
algoname = 'DeepVPD'
K_time = 4*60

train = {'K':K,'K_time':K_time,'use_FOC':True}

postfix = '_FOC'

print('')
print('#############################################')
print('')

try:
    run_DL(D,algoname,folder=folder,device=device,train=train,postfix=postfix)
    print('')
except Exception as e:
    print(e)
    print(f'Failed for {algoname} and {D = }')