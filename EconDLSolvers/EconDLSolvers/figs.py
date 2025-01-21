import numpy as np
import matplotlib.pyplot as plt

from .auxilliary import compute_transfer

def convergence_plot(model,
                     y_fac=1.0,ylabel='transfer',
                     folder='',postfix='',close_fig=True,filetype='svg',do_plot=True):
    """ check for convergence (and plot) """

    train = model.train

    # a. gather info
    x = []
    y = []
    Delta_time_within_transfer = None
    best_R = -np.inf
    for k in range(train.k+1):

        R = model.info[('R',k)]
        if np.isnan(R): continue
        if R > best_R: 
            best_R = R
        else:
            continue

        x.append(model.info[('k_time',k)]/60)
        transfer = compute_transfer(model.info['R_transfer'],train.transfer_grid,R)

        if np.isnan(transfer): 
            y.append(train.transfer_grid[0]-1) # just something below the grid
        else:
            y.append(transfer)

        if Delta_time_within_transfer is None and k < train.k:
            if transfer > -train.Delta_transfer:
                Delta_time_within_transfer = model.info[('k_time',train.k)]/60-x[-1]

    # b. plot
    if do_plot:
    
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)

        # i. convergence square
        y0 = -y_fac*train.Delta_transfer
        y1 = 0.0
        x0 = model.info[('k_time',train.k)]/60-train.Delta_time
        x1 = model.info[('k_time',k)]/60

        if x0 > 0:
            square = plt.Rectangle((x0,y0),x1-x0,y1-y0, edgecolor=None,facecolor='grey',alpha=0.5)
            ax.add_patch(square)          

        # ii. transfers
        ax.plot(np.array(x),y_fac*np.array(y),'-o')

        # iii. details
        ax.set_xlim([0,train.K_time])
        ax.set_ylim([y_fac*train.transfer_grid[0],y_fac*(-train.transfer_grid[-2])])
        ax.set_xlabel('time (m)')
        ax.set_ylabel(ylabel)

        # iv. save
        fig.tight_layout()
        if folder == '':
            fig.savefig(f'convergence{postfix}.png')
        else:
            fig.savefig(f'{folder}/convergence{postfix}.{filetype}')
        
        if close_fig: plt.close(fig)

    else:
        
        filename = f'convergence{postfix}.txt'
        with open(filename,'w') as f:
            for x,y in zip(x,y):
                f.write(f'{x:6.2f}: {y_fac*y:4.1f}\n')

    # c. termination check
    if Delta_time_within_transfer is not None:
        return Delta_time_within_transfer > train.Delta_time
    else:
        return False  