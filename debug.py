import matplotlib.pyplot as plt
import numpy as np

FIGURE_X = 6.0
FIGURE_Y = 4.0

def probabilities_vs_value_plot(snapshot, img_path):
    
    # Training 2-axes plot of episode length and vs episilon.
    V = np.array(snapshot['V(15)'])
    P, e  = zip(*[(p[-1], p[5]) for p in snapshot['PI(15)']])
    Y = np.array([P, e])
    label = snapshot['label']
    X = np.linspace(1, V.shape[0], V.shape[0])

    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    suptitle = 'Prob. vs. V(15)' 
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
    ax.plot(X, Y[0, :], c=c1[0], label='Pr.(a=15)')
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
def critic_update_error_plot(snapshot, img_path):
    
    V15 = np.array(snapshot['V(15)'])
    X = np.linspace(1, len(V15), len(V15))
    
    # indexes: taking action 15 on state 15 --> go to goal.
    index = np.array(snapshot['state']) == 15 
    optindex = np.array(snapshot['action']) == 15
    subindex = np.array(snapshot['action']) != 15
    mu = np.array(snapshot['mu'])


    label = snapshot['label']
    y_labels = ('optimal', 'suboptimal')
    indexes = (optindex, subindex) 

    
    i = 0
    for y_label, update_index in zip(y_labels, indexes):

        fig, ax = plt.subplots()
        fig.set_size_inches(FIGURE_X, FIGURE_Y)

        suptitle = 'Critic Updates on V(15)' 
        if label is not None:
            suptitle += f': {label}'
        
        filter_index  = index & update_index
        Yi = V15[filter_index] - V15[np.roll(filter_index, shift=1)]
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

# After an update in the right direction PI(15; w) should grow.
# def actor_update_error_plot(snapshot, img_path):
#     
#     PI15 = np.array(snapshot['PI(15)'])
#     X = np.linspace(1, PI15.shape[0], PI15.shape[0])
#     
#     # indexes: taking action 15 on state 15 --> go to goal.
#     index = np.array(snapshot['state']) == 15 
#     optindex = np.arange(16) == 15
#     subindex = np.arange(16) != 15
#     delta = np.array(snapshot['delta'])
# 
# 
#     label = snapshot['label']
#     y_labels = ('success', 'fail')
#     indexes = (optindex, subindex) 
# 
#     i = 0
#     for y_label, col_index in zip(y_labels, indexes):
# 
#         fig, ax = plt.subplots()
#         fig.set_size_inches(FIGURE_X, FIGURE_Y)
# 
#         suptitle = 'Critic Updates on V(15)' 
#         if label is not None:
#             suptitle += f': {label}'
#         
#         lag_index = np.roll(index, shift=1)
#         Yi = PI15[index] - PI15[lag_index]
#         Yi = Yi[:, col_index]
#         if len(Yi.shape) > 1: Yi = np.sum(Yi, axis=1)
#         delta_t = delta[index] - delta[lag_index]
#         Xi = np.linspace(1, len(Yi), len(Yi))
# 
#         y1_label = f'delta_PI[{y_label}]'
#         y2_label = 'delta'
#         x_label = 'Updates'
# 
#         #define colors to use
#         c1 = ['b', 'r']
#         c2 = 'g'
# 
#         #add first line to plot
#         ax.plot(Xi, Yi, c=c1[i], label=y1_label)
# 
#         #add x-axis label
#         ax.set_xlabel(x_label)
# 
#         #add y-axis label
#         ax.set_ylabel(y1_label)
# 
#         #define second y-axis that shares x-axis with current plot
#         ax2 = ax.twinx()
# 
#         #add second line to plot
#         ax2.plot(Xi, delta_t, color=c2)
# 
#         #add second y-axis label
#         ax2.set_ylabel(y2_label, color=c2)
# 
#         i += 1
#         
# 
#         plt.suptitle(suptitle)
#         plt.legend(loc=4)
#         file_name = img_path / f'actor_errors-{y_label}.pdf'
#         plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
#         file_name = img_path / f'actor_errors-{y_label}.png'
#         plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
#         plt.close()


def _diff(arr, val=0):
    return np.insert(np.diff(arr, axis=0), 0, val, axis=0)

def actor_update_error_plot(snapshot, img_path):
    # Are the 'bad' updates happening when other states are visited?

    # Are those updates in the wrong direction happening because other
    # states are visited?
    PI15 = np.array(snapshot['PI(15)'])
    state = np.array(snapshot['state'])
    delta_success = np.zeros_like(state)
    n_steps = len(state)

    pi15_15 = PI15[:, -1]
    delta_success  = _diff(pi15_15)
    X = np.linspace(1, n_steps, n_steps)
    
    # indicator: state != 15
    # get first and last values from the interval.
    # prev_pi, prev_state = PI15[0][-1], snapshot['state']
    # pi_deltas = 0
    # gen_delta_list = []
    # for step, state in enumerate(snapshot['state']):
    #     if prev_state != 15: pi_delta += PI15[step][-1]
    cumsum_success = np.cumsum(np.where(state == 15, delta_success, np.zeros_like(state)))

    label = snapshot['label']
    fig, ax = plt.subplots()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    suptitle = 'Actor Updates on s_t == 15' 
    if label is not None:
        suptitle += f': {label}'
    x_label = 'Training timesteps'
    y_label = 'Prob. success on s_t == 15'

    # define colors to use
    # c1 = ['b', 'r']
    # c2 = 'g'

    # add first line to plot
    ax.plot(X, pi15_15, c='b', label='Pr(success)')

    # add x-axis label
    ax.set_xlabel(x_label)

    # add y-axis label
    ax.set_ylabel(y_label, c='b')

    # define second y-axis that shares x-axis with current plot
    ax2 = ax.twinx()

    # add second line to plot
    ax2.plot(X, cumsum_success, color='g', label='Updates Pr(success)')

    # add second y-axis label
    ax2.set_ylabel('Updates Pr(success)', c='g')

    plt.suptitle(suptitle)
    plt.legend(loc=4)
    file_name = img_path / 'actor_update_errors.pdf'
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
    file_name = img_path / 'actor_update_errors.png'
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
    plt.close()

def actor_generalization_error_plot(snapshot, img_path):
    # Are the 'bad' updates happening when other states are visited?

    # Are those updates in the wrong direction happening because other
    # states are visited?
    PI15 = np.array(snapshot['PI(15)'])
    # We want only the delta
    PI15_15 = _diff(PI15[:, -1])
    PI15_err = _diff(np.sum(PI15[:, :-1], axis=1))
    state = np.array(snapshot['state'])
    n_steps = PI15.shape[0]
    X = np.linspace(1, n_steps, n_steps)
    
    # indicator: state != 15
    # get first and last values from the interval.
    # prev_pi, prev_state = PI15[0][-1], snapshot['state']
    # pi_deltas = 0
    # gen_delta_list = []
    # for step, state in enumerate(snapshot['state']):
    #     if prev_state != 15: pi_delta += PI15[step][-1]
    fail = np.cumsum(np.where(state != 15, PI15_err, np.zeros_like(state)))
    success = np.cumsum(np.where(state != 15, PI15_15, np.zeros_like(state)))

    label = snapshot['label']
    fig, ax = plt.subplots()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    suptitle = 'Critic Updates on s_t != 15' 
    if label is not None:
        suptitle += f': {label}'
    x_label = 'Probabilities on s=15'
    y_label = 'Training timesteps'

    # define colors to use
    # c1 = ['b', 'r']
    # c2 = 'g'

    # add first line to plot
    ax.plot(X, success, c='b', label='success')
    ax.plot(X, fail, c='r', label='fail')

    # add x-axis label
    ax.set_xlabel(x_label)

    # add y-axis label
    ax.set_ylabel(y_label)

    # define second y-axis that shares x-axis with current plot
    # ax2 = ax.twinx()

    # add second line to plot
    # ax2.plot(Xi, delta_t, color=c2)

    # add second y-axis label
    # ax2.set_ylabel(y2_label, color=c2)

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
    path = Path('data/AC-CHALLENGE/very_long_log/')
    # path = Path('data/20220214112104/')
    snapshot_path = path / 'snapshot.json'
    with snapshot_path.open('r') as f:
        snapshot = json.load(f)

    probabilities_vs_value_plot(snapshot, path)
    critic_update_error_plot(snapshot, path)
    actor_update_error_plot(snapshot, path)
    actor_generalization_error_plot(snapshot, path)
