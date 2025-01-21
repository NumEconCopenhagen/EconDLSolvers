import os
import glob
import pickle
from types import SimpleNamespace
import numpy as np
import pandas as pd
pd.set_option('display.max_colwidth',None) # stop the truncation of long strings

from IPython.display import display, HTML, Latex

import matplotlib.pyplot as plt
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

########
# load #
########

def load(filename):

    with open(filename, 'rb') as f:
        load_dict = pickle.load(f)

    return SimpleNamespace(**load_dict)

def load_all(folder,basename,no_DP=False):

    models = {}
    for filename in sorted(glob.glob(f'{folder}/{basename}*.pkl')):

        # a. specs
        specs = os.path.basename(filename).split('.')[0].split('_')
        assert len(specs) in [3,4]
        
        algoname = specs[1]
        if algoname == 'DP' and no_DP: continue
        D = specs[2]
        if len(specs) == 4: extra = specs[3]

        # b. print and store
        if len(specs) == 3:
            print(algoname,D)
            models[(algoname,D)] = load(filename)
        else:
            print(algoname,D,extra)
            models[(algoname,D,extra)] = load(filename)

    return models

def train_specs(models,do_display=True,folder='../output',filename=None):
    """ create a table with training specifications for each model."""

    rows = [
    (r'Nneurons_policy', r'Neurons in the policy network'),
    (r'policy_activation_intermediate', r'Activation function for policy network intermediate layers'),
    (r'policy_activation_final', r'Activation function for policy network final layer'),
    (r'Nneurons_value', r'Neurons in the value network'),
    (r'value_activation_intermediate', r'Activation function for value network intermediate layers'),
    (r'N_value_NN', r'Number of value networks'),

    (r'learning_rate_policy', r'Initial learning rate for the policy network'),
    (r'learning_rate_policy_decay', r'Decay rate for policy learning rate'),
    (r'learning_rate_policy_min', r'Minimum learning rate for the policy network'),
    (r'learning_rate_value', r'Initial learning rate for the value network'),
    (r'learning_rate_value_decay', r'Decay rate for value learning rate'),
    (r'learning_rate_value_min', r'Minimum learning rate for the value network'),

    (r'epsilon_sigma', r'Initial exploration noise, $\sigma_{\epsilon}$'),
    (r'epsilon_sigma_decay', r'Decay rate for exploration noise'),
    (r'epsilon_sigma_min', r'Minimum exploration noise'),
    (r'do_exo_actions_periods', r'Periods with exogenous actions'),

    (r'K', r'Maximum number of iterations before termination, $K$'),
    (r'K_time', r'Maximum number of minutes before termination'),

    (r'sim_R_freq', r'Simulation frequency, $\Delta_R$'),
    #(r'transfer_grid',r'Grid for calculation of transfer'),
    (r'Delta_transfer',r'Transfer tolerance for improvement'),
    (r'Delta_time',r'Time tolerance for improvement'),
    (r'K_time_min', r'Minimum number of minutes before termination'),

    (r'terminate_on_policy_loss', r'Terminate if policy loss is below tolerance'),
    (r'tol_policy_loss', r'Tolerance for policy loss'),

    (r'N', r'Sample size, $N^{\text{train }}$'),
    (r'buffer_memory', r'Size of replay buffer'),
    (r'batch_size', r'Batch size for training'),

    (r'epoch_termination', r'Epoch termination condition'),
    (r'Nepochs_policy', r'Number of epochs for policy network, $\#_{\pi}$'),
    (r'Delta_epoch_policy', r'Epoch increment for policy network, $\Delta_{\pi}$'),
    (r'epoch_policy_min', r'Minimum epochs for policy network'),
    (r'Nepochs_value', r'Number of epochs for value network, $\#_{\overline{v}}$'),
    (r'Delta_epoch_value', r'Number of epochs with no improvement before termination'),
    (r'epoch_value_min', r'Minimum epochs for value network'),

    (r'start_train_policy', r'Epoch to start training policy network'),
    (r'tau', r'Target smoothing coefficient, $\tau$'),
    (r'use_target_policy', r'Use target policy network'),
    (r'use_target_value', r'Use target value network'),

    (r'clip_grad_policy', r'Gradient clipping for policy network'),
    (r'clip_grad_value', r'Gradient clipping for value network'),
    (r'use_input_scaling', r'Use input scaling'),
    (r'scale_vec_states', r'State scaling vector for states'),
    (r'scale_vec_states_pd', r'State scaling vector for post decesion states'),
    (r'min_actions', r'Minimum action values'),
    (r'max_actions', r'Maximum action values'),

    (r'dtype', r'Data type'),
    (r'use_FOC', r'Use analytical First Order Conditions'),
    (r'Nquad', r'Number of quadrature points'),
    (r'Ngpus', r'Number of GPUs used'),
    ]

    # extract parameter names and create a mapping for descriptions
    param_names = [param for param, desc in rows]
    descriptions = {param: desc for param, desc in rows}

    # initialize the DataFrame 
    columns = []
    for key in models.keys():
        if key[0] == 'DP':
            continue
        if len(key) == 2:
            columns.append(f'{key[0]}')
        else:
            columns.append(f'{key[0]}_{key[1]}')

    df = pd.DataFrame(index=param_names, columns=['Description'] + columns).fillna('-')

    # fill in the 'Description' column
    df['Description'] = df.index.map(descriptions)

    # fill in the DataFrame with model parameters
    for key, model in models.items():

        if key[0] == 'DP':
            continue

        for k in sorted(model.train.__dict__.keys(), key=str.casefold):
            v = model.train.__dict__[k]
            if v is None: continue
            if k in param_names:
                col_name = f'{key[0]}' if len(key) == 2 else f'{key[0]}_{key[1]}'

                if isinstance(v, (list, tuple, np.ndarray)):
                    
                    seen = set()
                    unique_v = [x for x in v if not (x in seen or seen.add(x))]

                    if k in ['policy_activation_intermediate', 'policy_activation_final', 'value_activation_intermediate']:
                        v_str = '/ '.join(map(str, unique_v))
                    elif len(unique_v) == 1:
                        v_str = f'{unique_v[0]} for all elements'
                    else:
                        v_str = ', '.join(map(str, v))
                
                else:
                
                    v_str = str(v)            

                df.loc[k, col_name] = v_str
                        
     # reset the index to turn the index into a column
    df = df.reset_index()
    df = df.rename(columns={'index': 'Variable Name'})

    # add \ in front of underscores in the 'variable Name' column and headers so latex does not see them as subscripts
    df['Variable Name'] = df['Variable Name'].str.replace('_', r'\_', regex=False)
    df.columns = [col.replace('_', r'\_') if isinstance(col, str) else col for col in df.columns]

    if do_display: display(df)

    if filename is not None:
        
        filepath = f'{folder}/{filename}.tex'
        display(Latex(f'<a href="{filepath}">{filepath}</a>'))
        
        # generate LaTeX code
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=FutureWarning)
            latex_table = df.to_latex(escape=False,index=False,na_rep='-',longtable=False)
        
        # Bold column headers
        #####################

        # split LaTeX code into lines
        lines = latex_table.split('\n')
        
        # find the line with the column headers (after \toprule)
        for i, line in enumerate(lines):
            if '\\toprule' in line:
                header_line_index = i + 1  # The header is usually the line after \toprule
                break
        
        # extract the header line
        header_line = lines[header_line_index]
        
        # remove the trailing '\\' from the header line
        if header_line.endswith('\\\\'):
            header_line = header_line[:-2]
            end_of_line = ' \\\\'
        else:
            end_of_line = ''
        
        # split headers
        headers = header_line.split('&')
        headers = [header.strip() for header in headers]
        
        # wrap headers in \textbf{}
        headers = ['\\textbf{' + h + '}' for h in headers]
        
        # reconstruct the header line and add the end '\\'
        lines[header_line_index] = ' & '.join(headers) + end_of_line
        
        # reconstruct the LaTeX code
        latex_table = '\n'.join(lines)
        ############################

        # wrap the table in \resizebox{\textwidth}{!}{...}
        latex_table = '\\resizebox{\\textwidth}{!}{%\n' + latex_table + '\n}'

        # write the LaTeX code to the file
        with open(filepath, 'w') as f:
            f.write(latex_table)

############
# transfer #
############

def compute_transfer(R_transfer,transfer_grid,R,do_extrap=False):

    # print(R - R_transfer[0],R_transfer[-1] - R)

    if R < R_transfer[0]:
        if not do_extrap: return np.nan
        fac = (R_transfer[0]-R) / (R_transfer[1]-R_transfer[0])
        transfer = transfer_grid[0] - fac * (transfer_grid[1]-transfer_grid[0])
    elif R > R_transfer[-1]:
        if not do_extrap: return np.nan
        fac = (R-R_transfer[-1]) / (R_transfer[-1]-R_transfer[-2])
        transfer = transfer_grid[-1] + fac * (transfer_grid[-1]-transfer_grid[-2])
    else:
        transfer = np.interp(R,R_transfer,transfer_grid)

    return transfer

def transfer_plot(name,models,algonames,D,folder='../output'):

    fig = plt.figure(figsize=(9,6))
    ax = fig.add_subplot(1,1,1)

    model_DP = models[('DP',f'{D}D')]
    if hasattr(model_DP.sim,'R_transfer'):
        R_transfer = model_DP.sim.R_transfer
    else:
        R_transfer = model_DP.egm.R_transfer
    ax.plot(R_transfer,100**2*model_DP.egm.transfer_grid,'-o',color='black',lw=2,ms=3,label='EGM')
    ax.axhline(y=0.0,lw=2,color='black',ls='-')

    for i,(algoname,ls) in enumerate(zip(algonames,('-','--',':','-.',(0, (3, 5, 1, 5, 1, 5))))):
        key = (algoname,f'{D}D')
        if not key in models: continue
        models[key].info['transfer'] = transfer = compute_transfer(R_transfer,model_DP.egm.transfer_grid,models[key].sim.R)
        ax.axvline(x=models[key].sim.R,lw=2,color=colors[i],ls=ls,label=algoname)
        ax.axhline(y=100**2*transfer,lw=2,color=colors[i],ls=ls)

    ax.set_xlabel('average expected life-time reward, $R$')
    ax.set_ylabel('transfer, bp. of initial cash-on-hand')
    ax.legend(loc='upper left',ncol=2)
    ax.set_yscale('symlog')

    fig.savefig(f'{folder}/{name}_transfer_{D}D.svg') 
   
###############
# convergence #
###############

def convergence_plot(modelname,models,specs,DP=None,do_transfer=False,DP_name='EGM',
                     xlim=None,ylim=None,legend_ncol=2,
                     folder='../output',postfix='',do_display=True):

    fig, ax = plt.subplots(1,1,figsize=(12,6))

    if isinstance(DP,dict): 
        assert len(specs) == len(DP)
        DPs = DP
    else:
        DPs = None

    for i,(key,label) in enumerate(specs.items()):
        
        model = models[key]
        if isinstance(DPs,dict): DP = models[('DP',key[1])]
            
        x = []
        y = []
        best = -np.inf
        for k in range(model.train.K):
            
            if not ('R',k) in model.info: continue

            R = model.info[('R',k)]
            if np.isnan(R): continue

            if do_transfer:
                R_transfer = DP.sim.R_transfer
                transfer = compute_transfer(R_transfer,DP.egm.transfer_grid,R)
                if transfer > best: 
                    best = transfer
                    y.append(100**2*transfer)
                    x.append(model.info[('k_time',k)]/60)
            else:
                if R > best: 
                    best = R
                    y.append(R)
                    x.append(model.info[('k_time',k)]/60)

        ax.plot(np.log10(x),np.array(y),label=label,marker='o',ms=4,color=colors[i],lw=2)

    # DP
    if not DP is None:

        if not isinstance(DPs,dict):

            if do_transfer:
                ax.axhline(y=0,color='black',ls=':',lw=2)
            else:
                ax.axhline(y=DP.sim.R,color='black',ls=':',lw=2)

            if hasattr(DP,'info') and 'time' in DP.info:
                ax.axvline(x=np.log10(DP.info['time']/60),label=DP_name,color='black',ls=':',lw=2)

        else:

            for i,(key,label) in enumerate(DPs.items()):

                DP = models[key]

                if do_transfer:
                    ax.axhline(y=0,color=colors[i],ls=':',lw=2,label=label)
                else:
                    ax.axhline(y=DP.sim.R,color=colors[i],ls=':',lw=2,label=label)

                ax.axvline(x=np.log10(DP.info['time']/60),color=colors[i],ls=':',lw=2,label='')

    # x-axis
    mins = [0.001,0.01,0.1,1,10,100,1000]
    mins_minor = [
        0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,
        0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,
        0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,
        2,3,4,5,6,7,8,9,
        20,30,40,50,60,70,80,90,
        200,300,400,500,600,700,800,900
    ]
    log_ticks = np.log10(np.array(mins))
    ax.set_xticks(log_ticks)
    # add minor ticks
    ax.set_xticks(np.log10(mins_minor),minor=True)

    # old labels:
    # ax.set_xticklabels([f"$10^{{{int(tick)}}}$" for tick in log_ticks])    

    # new labels:
    labels = []
    for m in mins:
        
        if m < 0.01:
            labels.append(f'{m:.3f}')
        elif m < 0.1:
            labels.append(f'{m:.2f}')
        elif m < 1:
            labels.append(f'{m:.1f}')
        else:
            labels.append(f'{int(m)}')

    ax.set_xticklabels(labels)    
    
    if xlim is not None: ax.set_xlim([np.log10(xlim[0]),np.log10(xlim[1])]) 
    ax.set_xlabel('time (mins)')
    
    # y-axis
    if do_transfer: 
        ax.set_yscale('symlog',linscale=0.5)
        ax.set_yticks([-1000,-100,-10,-1,0,1,10,100])
        ax.set_yticklabels([-1000,-100,-10,-1,0,1,10,100])  

    if ylim is not None: ax.set_ylim(ylim)
    
    if do_transfer:
        ax.set_ylabel('transfer, bp. of initial cash-on-hand')
    else:
        ax.set_ylabel('average expected life-time reward, $R$')

    # legend
    ax.legend(loc='upper left',ncol=legend_ncol)

    # save
    fig.tight_layout()
    filepath = f'{folder}/{modelname}_convergence{postfix}.svg'
    fig.savefig(filepath)

    if do_display:
        plt.show()
    else:
        plt.close(fig)
        display(HTML(f'<a href="{filepath}">{filepath}</a>'))