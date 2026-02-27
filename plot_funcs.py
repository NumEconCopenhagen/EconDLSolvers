import os
import glob
import pickle
from turtle import backward
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
pd.set_option('display.max_colwidth',None) # stop the truncation of long strings

from IPython.display import display, HTML, Latex

import matplotlib.pyplot as plt
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

########
# load #
########

def load(filename,device='cpu'):

    try:
        with open(filename, 'rb') as f:
            load_dict = torch.load(f,map_location=torch.device('cpu'),weights_only=False)
    except:
        with open(filename, 'rb') as f:
            load_dict = pickle.load(f)

    return SimpleNamespace(**load_dict)

def load_all(folder,basename,no_DP=False):

    models = {}
    for filename in sorted(glob.glob(f'{folder}/{basename}_*.pt')):

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

def get_train_specs_rows(VPD=False):

    rows = [

        # neural networks
        (r'Nneurons_policy', r'Neurons in the policy network'),
        (r'policy_activation_intermediate', r'Activation function for policy network intermediate layers'),
        (r'policy_activation_final', r'Activation function for policy network final layer'),
        (r'time_input_type', r'Type of time input'),

        # learning rates
        (r'learning_rate_policy', r'Initial learning rate for the policy network'),
        (r'learning_rate_policy_decay', r'Decay rate for policy learning rate'),
        (r'learning_rate_policy_min', r'Minimum learning rate for the policy network'),

        # exploration
        (r'epsilon_sigma', r'Initial exploration noise, $\sigma_{\epsilon}$'),
        (r'epsilon_sigma_decay', r'Decay rate for exploration noise'),
        (r'epsilon_sigma_min', r'Minimum exploration noise'),
        (r'explore_frac', r'Initial fraction of explorers'),

        # termination criteria
        (r'K', r'Maximum number of iterations before termination, $K$'),
        (r'K_time', r'Maximum number of minutes before termination'),

        # simulation and data
        (r'sim_R_freq', r'Simulation frequency, $\Delta_R$'),
        (r'N', r'Sample size, $N^{\text{train }}$'),
        (r'buffer_memory', r'Size of replay buffer'),
        (r'batch_size', r'Batch size for training'),

        # epochs
        (r'epoch_termination', r'Epoch termination condition'),
        (r'Nepochs_policy', r'Number of epochs for policy network, $\#_{\pi}$'),
        (r'Delta_epoch_policy', r'Epoch increment for policy network, $\Delta_{\pi}$'),
        (r'epoch_policy_min', r'Minimum epochs for policy network'),

        # constraints and clipping
        (r'clip_grad_policy', r'Gradient clipping for policy network'),
        (r'min_actions', r'Minimum action values'),
        (r'max_actions', r'Maximum action values'),

        # misc
        (r'dtype', r'Data type'),
        (r'Nnumint', r'Number of quadrature points'),
        (r'Ngpus', r'Number of GPUs used'),

        ]


    if VPD:

        extra_list = [

            # Termination of epochs (value)
            (r'Delta_epoch_value', r'Number of epochs without improvement before termination'),
            (r'Nepochs_value', r'number of epochs in update to update value'),

            # Termination of episodes
            (r'Delta_transfer', r'Transfer tolerance for improvement'),
            (r'Delta_time', r'Time tolerance for improvement'),
            (r'K_time_min', r'Minimum number of minutes before termination'),

            # Using FOC in value-based algorithms
            (r'use_FOC', r'Use analytical First Order Conditions'),
            (r'NFOC_targets', r'Number of FOC targets'),
            (r'FOC_weight_pol', r'Weight on FOC when evaluating policy loss'),
            (r'FOC_weight_val', r'Weight on FOC when evaluating value loss'),
            (r'eq_w', r'Weight to put on each FOC if multiple in DeepFOC'),
            (r'value_weight_pol', r'Weight on value of choice when evaluating policy loss'),
            (r'value_weight_val', r'Weight on value of choice when evaluating value loss'),

            # Backward induction
            (r'backward', r'Do backwards induction'),
            (r'use_simult_in_backward', r'Use simultaneous neural networks when training backward neural networks'),
            (r'Nneurons_policy_t', r'Number of neurons in policy neural network at each time t'),
            (r'Nneurons_value_t', r'Number of neurons in value neural network at each time t'),
            (r'NN_init_std', r'Standard deviation for initialization of weights and biases from normal distribution'),
            (r'Nepochs_policy_t', r'Number of epochs for policy neural network at each time t'),
            (r'learning_rate_policy_t', r'Learning rate for policy neural network at each time t'),
            (r'learning_rate_policy_decay_t', r'Learning rate decay for policy neural network at each time t'),
            (r'Nepochs_value_t', r'Number of epochs for value neural network at each time t'),
            (r'learning_rate_value_t', r'Learning rate for value neural network at each time t'),
            (r'learning_rate_value_decay_t', r'Learning rate decay for value neural network at each time t'),

            # Neural network and learning rates (value networks)
            (r'Nneurons_value', r'Neurons for value network'),
            (r'N_value_NN', r'Number of value networks'),
            (r'value_activation_intermediate', r'Activation functions for value network intermediate layers'),
            (r'learning_rate_value', r'Learning rate maximum for value functions'),
            (r'learning_rate_value_decay', r'Decay in learning rate for value functions'),
            (r'learning_rate_value_min', r'Minimum learning rate for value functions'),
            (r'learning_rate_value_schedule', r'Learning rate schedule for value network'),

            # Policy learning extensions
            (r'learning_rate_policy_schedule', r'Learning rate schedule for policy network'),
            (r'manual_init_policy', r'Manual initialization of policy neural network'),

            # Training samples and targets
            (r'N_target_batches', r'Number of batches when computing target'),
            (r'N_sample_policy_loss', r'Number of sample batches when computing policy loss on simulation'),

            # Replay buffer extensions
            (r'store_actions', r'Store actions in replay buffer'),
            (r'store_pd', r'Store post-decision states in replay buffer'),
            (r'store_reward', r'Store rewards in replay buffer'),
            (r'i_t_index', r'Include t index in replay buffer'),

            # Numerical integration
            (r'use_quad', r'Do quadrature or Monte Carlo integration'),
            (r'redraw_mc', r'Re-draw Monte Carlo nodes each episode'),
            (r'update_numint_weights', r'Update numerical integration weights when computing target'),

            # Inputs and transformations
            (r'Ninputs_time', r'Number of time inputs to the neural networks'),
            (r'Ninputs_aux', r'Number of auxiliary inputs to the neural networks'),
            (r'input_transformation', r'Transform inputs to the neural networks'),

            # Smoothing / target networks
            (r'start_train_policy', r'Start training policy after this number of episodes'),
            (r'tau', r'Target smoothing coefficient'),
            (r'tau_final', r'Final tau value'),
            (r'tau_schedule', r'Tau schedule over time'),
            (r'use_target_policy', r'Use target in update policy'),
            (r'use_target_value', r'Use target in update value'),
            (r'target_value_in_policy', r'Use target networks for policy training'),

            # Termination based on policy loss
            (r'terminate_on_policy_loss', r'Terminate if policy loss is below tolerance'),
            (r'tol_policy_loss', r'Tolerance for policy loss'),
            (r'track_sim_policy_loss', r'Track policy loss on simulation'),

            # Misc
            (r'clip_grad_value', r'Limit value of gradient for value network'),
            (r'epoch_use_best', r'Load best neural networks after each epoch'),
            (r'epoch_value_min', r'Minimum number of epochs'),
            (r'NN_use_best', r'Load best neural networks when terminating training'),
            (r'only_initial_states_and_shocks', r'Only train on initial states and shocks'),
            (r'terminal_actions_known', r'Whether terminal actions are known'),
            (r'convergence_plot', r'Print convergence plot in convergence.png'),
            (r'do_sim_eps', r'Produce sim_eps with simulation noise'),

            # Undocumented / model-specific
            (r'budget_shares', r'Do budget shares with softmax or sequential sigmoids'),
        ]

        rows = rows + extra_list

    return rows

def train_specs(models,do_display=True,folder='../output',filename=None):
    """ create a table with training specifications for each model."""

    VPD = False
    if ('DeepVPD' in [m.train.algoname for _,m in models.items()]) or ('DeepVPDDC' in [m.train.algoname for _,m in models.items()]):
        VPD = True
    
    rows = get_train_specs_rows(VPD=VPD)

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
            columns.append(f'{key[0]}_{key[2]}')

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

                if isinstance(v,(list,tuple,np.ndarray)):
                    
                    seen = set()
                    unique_v = [x for x in v if not (x in seen or seen.add(x))]

                    if k in ['policy_activation_intermediate','policy_activation_final','value_activation_intermediate']:
                        v_str = '/ '.join(map(str,unique_v))
                    elif len(unique_v) == 1:
                        v_str = f'{unique_v[0]} for all elements'
                    else:
                        v_str = ', '.join(map(str, v))
                
                elif v == 'one_hot':

                    v_str = 'one\\_hot'
                
                elif k == 'explore_frac':

                    v_str = str(v[0].item())

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
    if hasattr(model_DP.egm,'transfer_grid'):
        transfer_grid = model_DP.egm.transfer_grid
    else:
        transfer_grid = model_DP.vfi.transfer_grid
    ax.plot(R_transfer,100**2*transfer_grid,'-o',color='black',lw=2,ms=3,label='EGM')
    ax.axhline(y=0.0,lw=2,color='black',ls='-')

    for i,(algoname,ls) in enumerate(zip(algonames,('-','--',':','-.',(0, (3, 5, 1, 5, 1, 5))))):
        key = (algoname,f'{D}D')
        if not key in models: continue
        models[key].info['transfer'] = transfer = compute_transfer(R_transfer,transfer_grid,models[key].sim.R)
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

def convergence_plot(modelname,models,specs,DP=None,backward=None,do_transfer=False,DP_name='EGM',
                     xlim=None,ylim=None,legend_ncol=2,
                     folder='../output',postfix='',do_display=True,show_all=False):

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
        for k in range(model.train.k):
            
            if not ('R',k) in model.info: continue

            R = model.info[('R',k)]
            if np.isnan(R): continue

            if do_transfer:
                R_transfer = DP.sim.R_transfer
                if hasattr(DP,'egm'):
                    transfer_grid = DP.egm.transfer_grid
                else:
                    transfer_grid = DP.vfi.transfer_grid
                transfer = compute_transfer(R_transfer,transfer_grid,R)
                if transfer > best or show_all: 
                    best = transfer
                    y.append(100**2*transfer)
                    x.append(model.info[('k_time',k)]/60)
            else:
                if R > best or show_all: 
                    best = R
                    y.append(R)
                    x.append(model.info[('k_time',k)]/60)

        ax.plot(np.log10(x),np.array(y),label=label,marker='o',ms=4,color=colors[i],lw=2)
        # ... inside your for-loop, right after ax.plot(...)
        x_last = np.log10(x[-1])
        y_last = y[-1]

        # what to show: final value; optionally show gap to DP target
        label_text = f"{y_last:.3g}"
        if (DP is not None) and (not do_transfer) and (not isinstance(DPs, dict)):
            gap = y_last - DP.sim.R
            label_text += f"  (Δ={gap:+.2g})"  # precision vs. DP target

        ax.annotate(
            label_text,
            xy=(x_last,y_last),
            xytext=(4,4), # small offset so it doesn't sit on the marker
            textcoords='offset points',
            fontsize=9,
            ha='left',va='bottom',
            color=colors[i],
            bbox=dict(boxstyle='round,pad=0.15',fc='white',ec='none',alpha=0.7),
            clip_on=False
        )

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
    
    # backward
    if not backward is None:

        if isinstance(backward, dict):
            backward_list = [backward]
        elif isinstance(backward, list):
            backward_list = backward
        else:
            raise TypeError(f"'backward' must be dict or list of dicts, got {type(backward)}")

        for bwd in backward_list:

            t_beg = bwd['t_beg']
            t_end = bwd['t_end']
            mins = t_end - t_beg

            if do_transfer:

                transfer_beg = bwd['transfer_beg']
                transfer_end = bwd['transfer_end']

                ax.plot(
                    [np.log10(t_beg), np.log10(t_end)],
                    [100**2 * transfer_beg, 100**2 * transfer_end],
                    color='gray', ls='--', marker='o', lw=2,
                    label=f'backward induction {mins:.0f} mins'
                )

                ax.annotate(
                    f'{100**2 * transfer_end:.3g}',
                    xy=(np.log10(t_end), 100**2 * transfer_end),
                    xytext=(4, 4),
                    textcoords='offset points',
                    fontsize=9,
                    ha='left', va='bottom',
                    color='gray',
                    bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.7),
                    clip_on=False
                )

            else:

                R_beg = bwd['R_beg']
                R_end = bwd['R_end']

                ax.plot(
                    [np.log10(t_beg), np.log10(t_end)],
                    [R_beg, R_end],
                    color='gray', ls='--', marker='o', lw=2,
                    label=f'backward induction {mins:.0f} mins'
                )

                ax.annotate(
                    f'{R_end:.3g}',
                    xy=(np.log10(t_end), R_end),
                    xytext=(4, 4),
                    textcoords='offset points',
                    fontsize=9,
                    ha='left', va='bottom',
                    color='gray',
                    bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.7),
                    clip_on=False
                )


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
    ax.legend(loc='upper left',ncol=legend_ncol,framealpha=0.95)

    # save
    fig.tight_layout()
    filepath = f'{folder}/{modelname}_convergence{postfix}.svg'
    fig.savefig(filepath)

    if do_display:
        plt.show()
    else:
        plt.close(fig)
        display(HTML(f'<a href="{filepath}">{filepath}</a>'))
