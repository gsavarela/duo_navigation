from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
plt.style.use('ggplot')

FIGURE_X = 6.0
FIGURE_Y = 4.0
CENTRALIZED_AGENT_COLOR = (0.2, 1.0, 0.2)

def globally_averaged_plot(mus, img_path):
    
    globally_averaged_return = np.array(mus)
    n_steps = globally_averaged_return.shape[0]


    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)


    X = np.linspace(1, n_steps, n_steps)
    Y = globally_averaged_return

    plt.plot(X, Y, c=CENTRALIZED_AGENT_COLOR, label='Centralized')
    plt.xlabel('Time')
    plt.ylabel('Globally Averaged Return J')
    plt.legend(loc=4)

    file_name = img_path / 'globally_averaged_return.pdf'
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
    file_name = img_path / 'globally_averaged_return.png'
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
