import time
import numpy as np
import cProfile
import pstats

def time_solve(model,Nstats=0):
    """ Time the solve function """

    # a. turn on synchronization
    train = model.train
    train.allow_synchronize = True

    # b. profile
    pr = cProfile.Profile()
    pr.enable()

    t0 = time.perf_counter()
    model.solve(do_print=False)
    t1 = time.perf_counter()
    time_solve = t1-t0

    pr.disable()

    if Nstats > 0:
        
        print('')

        (
                    pstats.Stats(pr)
                    .strip_dirs()
                    .sort_stats(pstats.SortKey.CUMULATIVE)
                    .print_stats(Nstats)
        )

    print('')

    # c. gather stats
    stats = pstats.Stats(pr)		

    def get_all_keys(d):
        for key, value in d.items():
            yield key
            if isinstance(value,dict):
                yield from get_all_keys(value)

    def get_all_keys_in_levels(d,i=0,base=None):
        if base is None: base = ()
        for key, value in d.items():
            yield (i,base,key)
            if isinstance(value, dict):
                if len(base) > 0:
                    yield from get_all_keys_in_levels(value,i+1,(*base,key))
                else:
                    yield from get_all_keys_in_levels(value,i+1,(key,))

    if model.train.algoname == 'DeepSimulate':
        funcs_dict = {
            '_simulate_training_sample':None,
            'update_NN':{
                'policy_loss_f':None,
                'NN_step':None
                },
            '_update_best':None,
        }
    elif model.train.algoname == 'DeepFOC':
        funcs_dict = {
            '_simulate_training_sample':None,
            'update_NN':{
                'train_policy':{
                    'policy_loss_f':None,
                    'NN_step_policy':None
                    }  
                },
            '_update_best':None,
        }
    elif model.train.algoname == 'DeepVPD':
        funcs_dict = {
            '_simulate_training_sample':None,
            'update_NN':{
                'train_value':{
                    'compute_target':None,
                    'value_loss_f':None,
                    'NN_step_value':None
                    },
                'train_policy':{
                    'policy_loss_f':None,
                    'NN_step_policy':None
                    },
            },
            '_update_best':None,
        }
    else:
        raise NotImplementedError(f"Algo {model.train.algoname} not implemented")
    
    func_names = list(get_all_keys(funcs_dict))

    model.info_timing = {}
    for func_info, func_stats in stats.stats.items():
        _, _, function_name = func_info # this assumes no duplicate function names across modules
        if function_name in func_names:
            model.info_timing[function_name] = func_stats

    # d. print
    print(f'Total time: {time_solve:.1f} secs')
    for f in get_all_keys_in_levels(funcs_dict):
        
        lvl = f[0]
        if lvl == 0:
            supname = 'total'
            suptime = time_solve
        else:
            supname = f[1][-1]
            suptime = model.info_timing[supname][3]
        
        name = f[2]
        _name = '  '*f[0] + name
        nowtime = model.info_timing[name][3]
        
        print(f'{_name:35s} {100*nowtime/suptime:5.1f}% of {supname} [{nowtime:.1f} secs]')		

    # e. turn off synchronization		
    train.allow_synchronize = False