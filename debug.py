import matplotlib.pyplot as plt
import numpy as np

FIGURE_X = 6.0
FIGURE_Y = 4.0

def probabilities_vs_value_plot(snapshot, img_path):
    
    # Training 2-axes plot of episode length and vs episilon.
    V = np.array(snapshot['V(12)'])
    P, e  = zip(*[(p[12], p[15]) for p in snapshot['PI(12)']])
    Y = np.array([P, e])
    label = snapshot['label']
    X = np.linspace(1, V.shape[0], V.shape[0])

    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    suptitle = 'Prob. vs. V(12)' 
    if label is not None:
        suptitle += f': {label}'

    y1_label = 'Probabilities.'
    y2_label = 'Value'
    x_label = 'Timesteps'

    #define colors to use
    c1 = ['b', 'r']
    c2 = 'g'


    # define subplots
    fig, ax = plt.subplots()

    # add first line to plot
    labels = ('Pr.(s=12, a=12)','Pr.(s=12, a=15)')
    pl1 = ax.plot(X, Y.T, label=labels)

    # add x-axis label
    ax.set_xlabel(x_label)

    # add y-axis label
    ax.set_ylabel(y1_label)

    # define second y-axis that shares x-axis with current plot
    ax2 = ax.twinx()

    # add second line to plot
    pl2 = ax2.plot(X, V, color=c2, label='V(12)')

    # add second y-axis label
    ax2.set_ylabel(y2_label, color=c2)

    lns = pl1 + pl2
    labels = [l.get_label() for l in lns]
    plt.suptitle(suptitle)
    plt.legend(lns, labels, loc=0)
    file_name = img_path / 'probabilities_vs_value.pdf'
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
    file_name = img_path / 'probabilities_vs_value.png'
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
    plt.close()

# After an update in the right direction V(12; w) should grow.
def critic_update_error_plot(snapshot, img_path):
    
    V12 = np.array(snapshot['V(12)'])
    X = np.linspace(1, len(V12), len(V12))
    
    # indexes: taking action 12 on state 12 --> go to goal.
    index = np.array(snapshot['state']) == 12 
    optindex = np.array(snapshot['action']) == 12
    subindex = np.array(snapshot['action']) != 12
    mu = np.array(snapshot['mu'])


    label = snapshot['label']
    y_labels = ('optimal', 'suboptimal')
    indexes = (optindex, subindex) 

    
    i = 0
    for y_label, update_index in zip(y_labels, indexes):

        fig, ax = plt.subplots()
        fig.set_size_inches(FIGURE_X, FIGURE_Y)

        suptitle = 'Critic Updates on V(12)' 
        if label is not None:
            suptitle += f': {label}'
        
        filter_index  = index & update_index
        Yi = V12[filter_index] - V12[np.roll(filter_index, shift=1)]
        delta_mu = mu[filter_index] - mu[np.roll(filter_index, shift=1)]
        Xi = np.linspace(1, len(Yi), len(Yi))

        y1_label = f'delta_V[{y_label}]'
        y2_label = 'delta_mu'
        x_label = 'Updates'

        # define colors to use
        c1 = ['b', 'r']
        c2 = 'g'

        # add first line to plot
        ax.plot(Xi, Yi, c=c1[i], label=y1_label)

        # add x-axis label
        ax.set_xlabel(x_label)

        # add y-axis label
        ax.set_ylabel(y1_label)

        # define second y-axis that shares x-axis with current plot
        ax2 = ax.twinx()

        # add second line to plot
        ax2.plot(Xi, delta_mu, color=c2)

        # add second y-axis label
        ax2.set_ylabel(y2_label, color=c2)

        i += 1
        
        plt.suptitle(suptitle)
        plt.legend(loc=4)
        file_name = img_path / f'critic_errors-{y_label}.pdf'
        plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
        file_name = img_path / f'critic_errors-{y_label}.png'
        plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
        plt.close()

def _diff(arr, val=0):
    return np.insert(np.diff(arr, axis=0), 0, val, axis=0)

def actor_update_gains_plot(snapshot, img_path):
    # Are the 'bad' updates happening when other states are visited?

    # Are those updates in the wrong direction happening because other
    # states are visited?
    PI12 = np.array(snapshot['PI(12)'])
    state = np.array(snapshot['state'])
    delta_success = np.zeros_like(state)
    n_steps = len(state)

    pi12_12 = PI12[:, 12]
    delta_success  = _diff(pi12_12)
    X = np.linspace(1, n_steps, n_steps)
    
    Y = np.cumsum(np.where(state == 12, delta_success, np.zeros_like(state)))

    label = snapshot['label']
    fig, ax = plt.subplots()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    suptitle = 'Actor Update-Gains (st=12)' 
    if label is not None:
        suptitle += f': {label}'
    x_label = 'Training timesteps'
    y_label = 'Update Gains Pr(st=12, at=12)'

    # add first line to plot
    ax.plot(X, Y, c='b')

    # add x-axis label
    ax.set_xlabel(x_label)

    # add y-axis label
    ax.set_ylabel(y_label, c='b')

    # define second y-axis that shares x-axis with current plot
    ax2 = ax.twinx()

    # add second line to plot
    ax2.plot(X, pi12_12, c='g')

    # add second y-axis label
    y_label = 'Pr(s.=12, a=(up, right))'
    ax2.set_ylabel(y_label, c='g')

    plt.suptitle(suptitle)
    plt.legend(loc=4)
    file_name = img_path / 'actor_update_gains_vs_probability.pdf'
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
    file_name = img_path / 'actor_update_gains_vs_probability.png'
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
    plt.close()

def actor_generalization_gains_plot(snapshot, img_path):
    # Are the 'bad' updates happening when other states are visited?

    # Are those updates in the wrong direction happening because other
    # states are visited?

    PI12 = np.array(snapshot['PI(12)'])
    # We want only the delta
    Y = PI12[:, 12]
    gY = _diff(PI12[:, 12])
    state = np.array(snapshot['state'])
    n_steps = PI12.shape[0]
    X = np.linspace(1, n_steps, n_steps)
    
    gY = np.cumsum(np.where(state != 12, gY, np.zeros_like(state)))

    label = snapshot['label']
    fig, ax = plt.subplots()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    suptitle = 'Actor Gen.-Gains (st!= 12)' 
    if label is not None:
        suptitle += f': {label}'
    x_label = 'Training timesteps'
    y_label = 'Gen.-Gains Pr(s=12, at=12) when st!=12'

    # add first line to plot
    ax.plot(X, gY, c='b')

    # add x-axis label
    ax.set_xlabel(x_label)

    # add y-axis label
    ax.set_ylabel(y_label, c='b')

    # define second y-axis that shares x-axis with current plot
    ax2 = ax.twinx()

    # add second line to plot
    ax2.plot(X, Y, c='g')

    # add second y-axis label
    y_label = 'Pr(s.=12, a=(up, right))'
    ax2.set_ylabel(y_label, color='g')

    plt.suptitle(suptitle)
    plt.legend(loc=4)
    file_name = img_path / 'actor_generalization_gains_vs_probability.pdf'
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
    file_name = img_path / 'actor_generalization_gains_vs_probability.png'
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
    plt.close()

if __name__ == '__main__':


    import json
    from pathlib import Path

    path = Path('data/AC-CHALLENGE/known_to_work/onehot')
    # path = Path('data/AC-CHALLENGE/puzzle')

    # path = Path('data/20220214112104/')
    snapshot_path = path / 'snapshot.json'
    with snapshot_path.open('r') as f:
        snapshot = json.load(f)

    probabilities_vs_value_plot(snapshot, path)
    # critic_update_error_plot(snapshot, path)
    actor_update_gains_plot(snapshot, path)
    actor_generalization_gains_plot(snapshot, path)
