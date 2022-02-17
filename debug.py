import matplotlib.pyplot as plt
import numpy as np

FIGURE_X = 6.0
FIGURE_Y = 4.0

def probabilities_vs_value_plot(snapshot, img_path):
    
    # Training 2-axes plot of episode length and vs episilon.
    V = np.array(snapshot['V(3)'])
    P, e  = zip(*[(p[-1], p[5]) for p in snapshot['PI(3)']])
    Y = np.array([P, e])
    label = snapshot['label']
    X = np.linspace(1, V.shape[0], V.shape[0])

    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    suptitle = 'Prob. vs. V(3)' 
    if label is not None:
        suptitle += f': {label}'

    y1_label = 'Probabilities.'
    y2_label = 'Value'
    x_label = 'Timesteps'

    #define colors to use
    c1 = ['b', 'r']
    c2 = 'g'

    #define subplots
    fig, ax = plt.subplots()

    #add first line to plot
    ax.plot(X, Y[0, :], c=c1[0], label='Pr.(a=3)')
    ax.plot(X, Y[1, :], c=c1[1], label='Pr.(a=5)')

    #add x-axis label
    ax.set_xlabel(x_label)

    #add y-axis label
    ax.set_ylabel(y1_label)

    #define second y-axis that shares x-axis with current plot
    ax2 = ax.twinx()

    #add second line to plot
    ax2.plot(X, V, color=c2)

    #add second y-axis label
    ax2.set_ylabel(y2_label, color=c2)
    

    plt.suptitle(suptitle)
    plt.legend(loc=4)
    file_name = img_path / 'probabilities_vs_value.pdf'
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
    file_name = img_path / 'probabilities_vs_value.png'
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
    plt.close()

# After an update in the right direction V(15; w) should grow.
# def critic_update_error_plot(snapshot, img_path):
#     
#     V15 = np.array(snapshot['V(15)'])
#     X = np.linspace(1, len(V15), len(V15))
#     
#     # indexes: taking action 15 on state 15 --> go to goal.
#     index = np.array(snapshot['state']) == 15 
#     optindex = np.array(snapshot['action']) == 15
#     subindex = np.array(snapshot['action']) != 15
#     mu = np.array(snapshot['mu'])
# 
# 
#     label = snapshot['label']
#     y_labels = ('optimal', 'suboptimal')
#     indexes = (optindex, subindex) 
# 
#     
#     i = 0
#     for y_label, update_index in zip(y_labels, indexes):
# 
#         fig, ax = plt.subplots()
#         fig.set_size_inches(FIGURE_X, FIGURE_Y)
# 
#         suptitle = 'Critic Updates on V(15)' 
#         if label is not None:
#             suptitle += f': {label}'
#         
#         filter_index  = index & update_index
#         Yi = V15[filter_index] - V15[np.roll(filter_index, shift=1)]
#         delta_mu = mu[filter_index] - mu[np.roll(filter_index, shift=1)]
#         Xi = np.linspace(1, len(Yi), len(Yi))
# 
#         y1_label = f'delta_V[{y_label}]'
#         y2_label = 'delta_mu'
#         x_label = 'Updates'
# 
#         # define colors to use
#         c1 = ['b', 'r']
#         c2 = 'g'
# 
#         # add first line to plot
#         ax.plot(Xi, Yi, c=c1[i], label=y1_label)
# 
#         # add x-axis label
#         ax.set_xlabel(x_label)
# 
#         # add y-axis label
#         ax.set_ylabel(y1_label)
# 
#         # define second y-axis that shares x-axis with current plot
#         ax2 = ax.twinx()
# 
#         # add second line to plot
#         ax2.plot(Xi, delta_mu, color=c2)
# 
#         # add second y-axis label
#         ax2.set_ylabel(y2_label, color=c2)
# 
#         i += 1
#         
#         plt.suptitle(suptitle)
#         plt.legend(loc=4)
#         file_name = img_path / f'critic_errors-{y_label}.pdf'
#         plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
#         file_name = img_path / f'critic_errors-{y_label}.png'
#         plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
#         plt.close()

def _diff(arr, val=0):
    return np.insert(np.diff(arr, axis=0), 0, val, axis=0)

def actor_updates_plot(snapshot, img_path):
    # Are the 'bad' updates happening when other states are visited?

    # Are those updates in the wrong direction happening because other
    # states are visited?
    PI3 = np.array(snapshot['PI(3)'])
    state = np.array(snapshot['state'])
    n_steps = len(state)
    X = np.linspace(1, n_steps, n_steps)

    Y1 = PI3[:, -1]
    delta  = _diff(Y1)
    
    cumsum_updates = np.cumsum(np.where(state == 3, delta, np.zeros_like(state)))
    cumsum_generalization = np.cumsum(np.where(state != 3, delta, np.zeros_like(state)))
    Y2 = np.array([cumsum_updates, cumsum_generalization]).T

    label = snapshot['label']
    fig, ax = plt.subplots()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    suptitle = 'Actor Updates on (s=3)' 
    if label is not None:
        suptitle += f': {label}'
    x_label = 'Training timesteps'
    y_label = 'Pr(s=3, a=(right, up))'

    # add first line to plot
    ax.plot(X, Y1, c='b')

    # add x-axis label
    ax.set_xlabel(x_label)

    # add y-axis label
    ax.set_ylabel(y_label, c='b')

    # define second y-axis that shares x-axis with current plot
    ax2 = ax.twinx()

    # add second line to plot
    ax2.plot(X, Y2[:, 0], color='g', label='s=3')
    ax2.plot(X, Y2[:, 1], color='r', label='s!=3')

    # add second y-axis label
    ax2.set_ylabel('Updates', c='g')

    plt.suptitle(suptitle)
    plt.legend(loc=4)
    file_name = img_path / 'actor_updates.pdf'
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
    file_name = img_path / 'actor_updates.png'
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
    plt.close()

def actor_generalization_error_plot(snapshot, img_path):
    # Are the 'bad' updates happening when other states are visited?

    # Are those updates in the wrong direction happening because other
    # states are visited?
    PI3 = np.array(snapshot['PI(3)'])

    # We want only the delta
    PI3_3 = _diff(PI3[:, 3])
    state = np.array(snapshot['state'])
    n_steps = PI3.shape[0]
    X = np.linspace(1, n_steps, n_steps)

    success = np.cumsum(np.where(state != 3, PI3_3, np.zeros_like(state)))
    label = snapshot['label']
    fig, ax = plt.subplots()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    suptitle = 'Actor updates on s_t != 3' 
    if label is not None:
        suptitle += f': {label}'
    x_label = 'Training timesteps'
    y_label = 'Probabilities on s=3'

    # add first line to plot
    ax.plot(X, success, c='b', label='success')

    # add x-axis label
    ax.set_xlabel(x_label)

    # add y-axis label
    ax.set_ylabel(y_label)

    plt.suptitle(suptitle)
    plt.legend(loc=4)
    file_name = img_path / 'actor_generalization_errors.pdf'
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
    file_name = img_path / 'actor_generalization_errors.png'
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
    plt.close()

if __name__ == '__main__':


    import json
    from pathlib import Path
    path = Path('data/AC-CHALLENGE/with_exploration/')
    # path = Path('data/20220214112104/')
    snapshot_path = path / 'snapshot.json'
    with snapshot_path.open('r') as f:
        snapshot = json.load(f)

    probabilities_vs_value_plot(snapshot, path)
    # critic_update_error_plot(snapshot, path)
    actor_updates_plot(snapshot, path)
    # actor_generalization_error_plot(snapshot, path)
