from types import SimpleNamespace
import torch

def get_objs(model):
    
    NNs = ['policy_NN','value_NN','policy_NN_target','value_NN_target']
    opts = ['policy_opt','value_opt']
    schedulers = ['policy_scheduler','value_scheduler']

    return NNs+opts+schedulers

def save(model,filename):
    """ save model """

    save_dict = {}

    # a. namespaces
    nss = ['par','sim','train','sim_eps','rep_buffer']
    
    for ns in nss:

        if hasattr(model,ns) and not model.__dict__[ns] is None:

            save_dict[ns] = SimpleNamespace()

            for k,v in model.__dict__[ns].__dict__.items():                
                if k in ['device']: continue
                save_dict[ns].__dict__[k] = v
    
    # b. info
    save_dict['info'] = model.info
    if hasattr(model,'info_timing'): save_dict['info_timing'] = model.info_timing

    if not model.train.backward:

        for k,v in save_dict['info']['best'].policy_NN.items():
            save_dict['info']['best'].policy_NN[k] = v
        
        if not model.value_NN is None:
            for k,v in save_dict['info']['best'].value_NN.items():
                save_dict['info']['best'].value_NN[k] = v

        if not model.train.N_value_NN is None:
            for i in range(model.train.N_value_NN):
                for k,v in save_dict['info']['best'].value_NNs[i].items():
                    save_dict['info']['best'].value_NNs[i][k] = v

    # c. objs
    for k in get_objs(model):
        if hasattr(model,k) and not (v:=model.__dict__[k]) is None:
            save_dict[k] = v.state_dict()

    if not model.train.N_value_NN is None:

        save_dict['value_NNs'] = []
        if hasattr(model,'value_NN_targets'): save_dict['value_NN_targets'] = [] 
        save_dict['value_opts'] = []
        save_dict['value_schedulers'] = []
        
        for i in range(model.train.N_value_NN):

            v = model.value_NNs[i]
            save_dict['value_NNs'].append(v.state_dict())

            if hasattr(model,'value_NN_targets'): 
                v = model.value_NN_targets[i]
                save_dict['value_NN_targets'].append(v.state_dict())

            v = model.value_opts[i]
            save_dict['value_opts'].append(v.state_dict())
            
            v = model.value_schedulers[i]
            save_dict['value_schedulers'].append(v.state_dict())

    if model.train.backward:

        save_dict['policy_NN_t'] = []
        save_dict['policy_opt_t'] = []
        save_dict['policy_scheduler_t'] = []
        
        save_dict['value_NN_t'] = []
        save_dict['value_opt_t'] = []
        save_dict['value_scheduler_t'] = []

        for i in range(model.par.T):

            v = model.policy_NN_t[i]
            save_dict['policy_NN_t'].append(v.state_dict())
            
            v = model.policy_opt_t[i]
            save_dict['policy_opt_t'].append(v.state_dict())
            
            v = model.policy_scheduler_t[i]
            save_dict['policy_scheduler_t'].append(v.state_dict())

            if hasattr(model,'value_NN_t'):

                v = model.value_NN_t[i]
                save_dict['value_NN_t'].append(v.state_dict())

                v = model.value_opt_t[i]
                save_dict['value_opt_t'].append(v.state_dict())
                
                v = model.value_scheduler_t[i]
                save_dict['value_scheduler_t'].append(v.state_dict())

    save_dict['torch_rng_state'] = model.torch_rng_state
    save_dict['torch_rng_state_final'] = torch.get_rng_state()

    # d. save
    with open(f'{filename}', 'wb') as f:
        torch.save(save_dict, f)

def load(model,load_dict):

    device = model.train.device

    # a. namespaces
    nss = ['par','sim','train','sim_eps','rep_buffer','best']

    for ns in nss:
        if ns in load_dict:
            if not hasattr(model,ns): model.__dict__[ns] = SimpleNamespace()            
            for k,v in load_dict[ns].__dict__.items():			
                model.__dict__[ns].__dict__[k] = v
                if hasattr(v,'to'): v = v.to(device)

    # b. info
    model.info = load_dict['info']
    model.info_timing = load_dict['info_timing']
    
    if not model.train.backward:
        for k,v in model.info['best'].policy_NN.items():
            model.info['best'].policy_NN[k] = v.to(device)
        
        if not model.value_NN is None:
            for k,v in model.info['best'].value_NN.items():
                model.info['best'].value_NN[k] = v.to(device)

        if not model.train.N_value_NN is None:
            for i in range(model.train.N_value_NN):
                for k,v in model.info['best'].value_NNs[i].items():
                    model.info['best'].value_NNs[i][k] = v.to(device)

    # c. objs
    for obj in get_objs(model):
        if obj in load_dict:
            v = getattr(model,obj)
            v.load_state_dict(load_dict[obj]) 
            if hasattr(v,'to'): v = v.to(device)

    if not model.train.N_value_NN is None:

        for i in range(model.train.N_value_NN):
        
            model.value_NNs[i].load_state_dict(load_dict['value_NNs'][i]) 
            model.value_NNs[i] = model.value_NNs[i].to(device)

            if hasattr(model,'value_NN_targets'): 
                model.value_NN_targets[i].load_state_dict(load_dict['value_NN_targets'][i]) 
                model.value_NN_targets[i] = model.value_NN_targets[i].to(device)
                
            model.value_opts[i].load_state_dict(load_dict['value_opts'][i]) 
            model.value_schedulers[i].load_state_dict(load_dict['value_schedulers'][i]) 

    if model.train.backward:

        for i in range(model.par.T):

            model.policy_NN_t[i].load_state_dict(load_dict['policy_NN_t'][i]) 
            model.policy_NN_t[i] = model.policy_NN_t[i].to(device)
            
            model.policy_opt_t[i].load_state_dict(load_dict['policy_opt_t'][i]) 
            model.policy_scheduler_t[i].load_state_dict(load_dict['policy_scheduler_t'][i]) 

            if hasattr(model,'value_NN_t'):
            
                model.value_NN_t[i].load_state_dict(load_dict['value_NN_t'][i]) 
                model.value_NN_t[i] = model.value_NN_t[i].to(device)

                model.value_opt_t[i].load_state_dict(load_dict['value_opt_t'][i]) 
                model.value_scheduler_t[i].load_state_dict(load_dict['value_scheduler_t'][i])

    model.torch_rng_state = load_dict['torch_rng_state']
    for k,v in model.torch_rng_state.items(): 
        model.torch_rng_state[k] = v.to('cpu')
    torch.set_rng_state(load_dict['torch_rng_state_final'].to('cpu'))