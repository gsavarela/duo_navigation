import argparse
from collections import defaultdict
from functools import reduce
import json
import os
from pathlib import Path

from utils import pi2str, pos2str, act2str, act2str2, best_actions, q2str
from utils import str2bool

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


plt.style.use("ggplot")

FIGURE_X = 6.0
FIGURE_Y = 4.0
CENTRALIZED_AGENT_COLOR = (0.2, 1.0, 0.2)
SMOOTHING_CURVE_COLOR = (0.33, 0.33, 0.33)


def snapshot_plot(snapshot_log, img_path):

    episodes = snapshot_log["episode"]
    rewards = snapshot_log["reward"]

    # TODO: remove get and make default __itemgetter__.
    label = snapshot_log.get("label", None)
    task = snapshot_log.get("task", "episodic")

    if task == "episodic":
        epsilons = snapshot_log["epsilon"]
        # cumulative_rewards_plot(rewards, img_path, label)
        episode_duration_plot(episodes, epsilons, img_path, label=label)
        episode_rewards_plot(episodes, rewards, img_path, label=label)
    else:
        # For continous tasks
        globally_averaged_plot(snapshot_log["mu"], img_path, episodes)

    snapshot_path = img_path / "snapshot.json"
    with snapshot_path.open("w") as f:
        json.dump(snapshot_log, f)


# use this only for continuing tasks.
# episodes is a series with the episode numbers
def globally_averaged_plot(mus, img_path, episodes):

    globally_averaged_return = np.array(mus)

    episodes = np.array(episodes)
    globally_averaged_episodes = []
    episodes_to_plot = (
        int(np.min(episodes)),
        int(np.mean(episodes)),
        int(np.max(episodes)),
    )
    for episode in episodes_to_plot:
        globally_averaged_episodes.append(globally_averaged_return[episodes == episode])
    Y = np.vstack(globally_averaged_episodes).T

    n_steps = Y.shape[0]
    X = np.linspace(1, n_steps, n_steps)

    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    labels = [f"episode {epis}" for epis in episodes_to_plot]
    plt.plot(X, Y, label=labels)
    plt.xlabel("Time")
    plt.ylabel("Globally Averaged Return J")
    plt.legend(loc=4)

    file_name = img_path / "globally_averaged_return_per_episode.pdf"
    plt.savefig(file_name, bbox_inches="tight", pad_inches=0)
    file_name = img_path / "globally_averaged_return_per_episode.png"
    plt.savefig(file_name, bbox_inches="tight", pad_inches=0)
    plt.close()


def cumulative_rewards_plot(rewards, img_path, label=None):

    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    Y = np.cumsum(rewards) / np.arange(1, len(rewards) + 1)
    X = np.linspace(1, len(rewards), len(rewards))
    # Y_smooth = sm.nonparametric.lowess(Y, X, frac=0.10)

    suptitle = "Team Return"
    y_label = "Cumulative Averaged Reward"
    x_label = "Timestep"

    plt.suptitle(suptitle)
    plt.plot(X, Y, c=CENTRALIZED_AGENT_COLOR, label=label)
    # plt.plot(X, Y_smooth[:, 1], label=f'Smoothed {label}')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(loc=4)

    file_name = img_path / "cumulative_averaged_rewards.pdf"
    plt.savefig(file_name, bbox_inches="tight", pad_inches=0)
    file_name = img_path / "cumulative_averaged_rewards.png"
    plt.savefig(file_name, bbox_inches="tight", pad_inches=0)
    plt.close()


def episode_rewards_plot(episodes, rewards, img_path, label=None):
    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    target = 0.1

    rewards = np.array(rewards)
    episodes = np.array(episodes)
    rewards_episodes = []
    episodes_to_plot = np.arange(np.max(episodes))
    X = np.linspace(1, len(episodes_to_plot), len(episodes_to_plot))
    for episode in episodes_to_plot:
        rewards_episodes.append(np.sum(rewards[episodes == episode]))

    Y = np.array(rewards_episodes)
    # Y_smooth = sm.nonparametric.lowess(Y, X, frac=0.10)

    suptitle = "Episode Return vs Target"
    y_label = "Cum. Average Return Per Episode"
    x_label = "Episodes"

    plt.suptitle(suptitle)
    plt.axhline(y=target, c="red", label="target")
    # plt.plot(X, Y_smooth[:, 1], c=SMOOTHING_CURVE_COLOR, label=label)
    plt.plot(X, Y, c=CENTRALIZED_AGENT_COLOR, label=label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(loc=4)
    file_name = img_path / "return_per_episode.pdf"
    plt.savefig(file_name, bbox_inches="tight", pad_inches=0)
    file_name = img_path / "return_per_episode.png"
    plt.savefig(file_name, bbox_inches="tight", pad_inches=0)
    plt.close()


# Training 2-axes plot of episode length and vs episilon.
def episode_duration_plot(episodes, epsilons, img_path, label=None):

    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    epsilons = np.array(epsilons)
    episodes = np.array(episodes)
    episodes_to_plot = np.arange(np.max(episodes))
    X = np.linspace(1, len(episodes_to_plot), len(episodes_to_plot))

    episodes_duration = []
    episodes_epsilon = []
    for episode in episodes_to_plot:
        episodes_duration.append(np.sum(episodes == episode))
        episodes_epsilon.append(np.mean(epsilons[episodes == episode]))

    Y1 = np.array(episodes_duration)
    # Y1_smooth = sm.nonparametric.lowess(Y1, X, frac=0.10)
    Y2 = np.array(episodes_epsilon)

    suptitle = "Duration vs. Epsilon"
    if label is not None:
        suptitle += f": {label}"

    y1_label = "Duration"
    y2_label = "Epsilon"
    x_label = "Episodes"

    # define colors to use
    c1 = "steelblue"
    c2 = "red"

    # define subplots
    fig, ax = plt.subplots()

    # add first line to plot
    # ax.plot(X, Y1_smooth[:, 1], color=SMOOTHING_CURVE_COLOR)
    ax.plot(X, Y1, color=CENTRALIZED_AGENT_COLOR)

    # add x-axis label
    ax.set_xlabel(x_label)

    # add y-axis label
    # ax.set_ylabel(f'Smoothed {y1_label}', color=SMOOTHING_CURVE_COLOR)
    ax.set_ylabel(f"{y1_label}", color=CENTRALIZED_AGENT_COLOR)

    # define second y-axis that shares x-axis with current plot
    ax2 = ax.twinx()

    # add second line to plot
    ax2.plot(X, Y2, color=c2)

    # add second y-axis label
    ax2.set_ylabel(y2_label, color=c2)

    plt.suptitle(suptitle)
    file_name = img_path / "duration_vs_epsilon.pdf"
    plt.savefig(file_name, bbox_inches="tight", pad_inches=0)
    file_name = img_path / "duration_vs_epsilon.png"
    plt.savefig(file_name, bbox_inches="tight", pad_inches=0)
    plt.close()


def advantages_plot(
    advantages, results_path, state_actions=[(0, [0]), (1, [0, 3]), (3, [3])]
):

    n_steps = len(advantages)
    # Makes a list of dicts.
    ld = [dict(adv) for adv in advantages if adv[-1] is not None]
    # Converts a list of dicts into dictionary of lists.
    dl = {k: [d[k] for d in ld] for k in ld[0]}
    for x, ys in state_actions:
        if x in dl:
            fig = plt.figure()
            fig.set_size_inches(FIGURE_X, FIGURE_Y)
            X = np.linspace(1, n_steps, n_steps)
            Y = np.array(dl[x])
            labels = tuple([f"Best action {act2str(y)}" for y in ys])

            plt.suptitle(f"Advantages State {x}")
            plt.plot(X, Y, label=labels)
            plt.xlabel("Timesteps")
            plt.ylabel("Advantages")
            plt.legend(loc="center right")

            file_name = (results_path / f"advantages_state_{x}.pdf").as_posix()
            plt.savefig(file_name, bbox_inches="tight", pad_inches=0)
            file_name = (results_path / f"advantages_state_{x}.png").as_posix()
            plt.savefig(file_name, bbox_inches="tight", pad_inches=0)
            plt.close()


def q_values_plot(
    q_values, results_path, state_actions=[(0, [0]), (1, [0, 3]), (3, [3])]
):

    n_steps = len(q_values)
    # Makes a list of dicts.
    ld = [dict(qval) for qval in q_values if qval[-1] is not None]
    # Converts a list of dicts into dictionary of lists.
    dl = {k: [d[k] for d in ld] for k in ld[0]}
    for x, ys in state_actions:
        if x in dl:
            fig = plt.figure()
            fig.set_size_inches(FIGURE_X, FIGURE_Y)
            X = np.linspace(1, n_steps, n_steps)
            Y = np.array(dl[x])
            labels = tuple([f"Best action {act2str(y)}" for y in ys])

            plt.suptitle(f"Q-values State {x}")
            plt.plot(X, Y, label=labels)
            plt.xlabel("Timesteps")
            plt.ylabel("Relative Q-values")
            plt.legend(loc="center right")

            file_name = (results_path / f"q_values_state_{x}.pdf").as_posix()
            plt.savefig(file_name, bbox_inches="tight", pad_inches=0)
            file_name = (results_path / f"q_values_state_{x}.png").as_posix()
            plt.savefig(file_name, bbox_inches="tight", pad_inches=0)
            plt.close()


# TODO: port display_ac
def display_policy(env, agent):
    return display_ac(env, agent)


def display_ac(env, agent):
    goal_pos = env.goal_pos

    def bact(x):
        return best_actions(
            x, np.array(list(goal_pos)), width=env.width - 2, height=env.height - 2
        )

    print(env)

    states_positions_gen = env.next_states()
    margin = "#" * 30
    data = defaultdict(list)
    print(f"{margin} GOAL {margin}")
    print(f"GOAL: {goal_pos}")
    print(f"{margin} ACTOR {margin}")
    while True:
        try:
            state, pos = next(states_positions_gen)
            pi_log = pi2str(agent.PI[state])
            max_action = np.argmax(agent.PI[state])
            actions_optimal = bact(pos)

            actions_log = act2str2([max_action])
            best_log = act2str2(actions_optimal)
            pos_log = ", ".join([pos2str(p) for p in pos])
            data["state"].append(state)
            data["Coord 1"].append(tuple(pos[0]))
            data["Coord 2"].append(tuple(pos[1]))
            data["V"].append(np.round(agent.V[state], 2))
            data["move_most_likely"].append(actions_log)
            data["move_optimal"].append(best_log)

            pr_success = 0
            advantages = agent.A[state, :]
            for i, pi in enumerate(agent.PI[state]):
                data[f"A(state, {i})"].append(f"{agent.A[state, i]:0.2f}")
                data[f"PI(state, {i})"].append(np.round(pi, 2))
                if i in actions_optimal:
                    pr_success += pi
            data[f"PI(state, success)"].append(np.round(pr_success, 2))

            advantage_log = ",".join([f"{a:0.2f}" for a in advantages])

            msg = (
                f"\t{state}\t{pos_log}"
                f"\tV({state})={agent.V[state]:0.2f}"
                f"\t{pi_log}\n"
                f"\t{state}\t{pos_log}"
                f"\tP(success={pr_success:0.2f})"
                f"\t{actions_log}\tin\t{best_log}: {max_action in actions_optimal}\n"
                f"\t{state}\t{pos_log}"
                f"\tP(success={pr_success:0.2f})\t({advantage_log})\n"
                f'{"-" * 150}'
            )
            print(msg)
        except StopIteration:
            break

    if hasattr(agent, "Q"):
        print(f"{margin} CRITIC {margin}")
        states_positions_gen = env.next_states()
        while True:
            try:
                state, pos = next(states_positions_gen)

                max_action = np.argmax(agent.Q[state, :])

                actions_log = act2str2([max_action])
                actions_optimal = bact(pos)
                best_log = act2str2(actions_optimal)
                pos_log = ", ".join([pos2str(p) for p in pos])
                msg = (
                    f"\t{state}"
                    f"\t{agent.V[state]:0.2f}"
                    f"\t{pos_log}"
                    f"\t{q2str(agent.Q[state, :])}"
                    f"\t{actions_log}"
                    f"\t{best_log}"
                )
                print(msg)

                for i, q in enumerate(agent.Q[state, :]):
                    data[f"Q(state, {i})"].append(np.round(q, 2))
            except StopIteration:
                break

    df = pd.DataFrame.from_dict(data).set_index("state")

    return df


def validation_plot(rewards):

    Y = np.cumsum(rewards)
    n_steps = len(rewards)
    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)
    X = np.linspace(1, n_steps, n_steps)
    plt.suptitle(f"Validation Round")
    plt.plot(X, Y)
    plt.xlabel("Timesteps")
    plt.ylabel("Accumulated Averaged Rewards.")
    plt.show()


def get_arguments():
    parser = argparse.ArgumentParser(
        description="""
            This script creates average reward plots for episodes.
            
        """
    )

    parser.add_argument(
        "paths", type=str, nargs="+", help="List of paths to experiments."
    )
    parser.add_argument(
        "-l", "--labels", nargs="+", help="List of experiments' labels.", required=True
    )
    parser.add_argument(
        "-t",
        "--subtitle",
        nargs=1,
        default="",
        type=str,
        help="Subtitle to graphs",
        required=False,
    )
    parser.add_argument(
        "-o",
        "--use_parent_output",
        nargs=1,
        default=True,
        type=str2bool,
        help="Uses parent directory (common ancestor) as output folder.",
        required=False,
    )

    return parser.parse_args()


def print_arguments(args, output_folder_path):

    print("Arguments (analysis/compare.py):")
    print("\tExperiments: {0}\n".format(args.paths))
    print("\tExperiments labels: {0}\n".format(args.labels))
    print("\tExperiments subtitle: {0}\n".format(args.subtitle))
    print("\tOutput folder: {0}\n".format(output_folder_path))


def get_common_path(paths):
    """returns the common ancestor path

    Parameters:
    ----------
    * paths: list<pathlib.Path>
    experiment paths

    Returns:
    --------
    * pathlib.Path object
    output folder
    """
    path_parts = map(lambda x: Path(x).parts, paths)

    def fn(x, y):
        return tuple([xx for xx, yy in zip(x, y) if xx == yy])

    common_folder_parts = reduce(fn, path_parts)
    common_folder_path = Path.cwd().joinpath(*common_folder_parts)
    return common_folder_path


def main():

    print("\nRUNNING plots benchmark\n")

    args = get_arguments()

    output_folder_path = Path("data/benchmark/")
    if args.use_parent_output:
        output_folder_path = get_common_path(args.paths)

    print_arguments(args, output_folder_path)

    # Prepare output folder.
    os.makedirs(output_folder_path.as_posix(), exist_ok=True)

    dataframes = []
    for label, exp_path in zip(args.labels, args.paths):

        file_path = Path(exp_path) / "snapshot.json"
        with file_path.open("r") as f:
            snapshot = json.load(f)

        episode = np.array(snapshot["episode"])
        reward = np.array(snapshot["reward"])
        data = np.vstack([episode, reward]).T
        df = pd.DataFrame(data).groupby(0, as_index=False).sum().set_index(0)

        # episodes and returns
        df.columns = [label]
        dataframes.append(df)

    df = pd.concat(dataframes, axis=1)

    """Cumulative Summation of Returns(Mean Rewards)"""
    cs_df = df.copy().cumsum(axis=0)
    title = "Cumulative Return"
    if len(args.subtitle) > 0:
        title = f"{args.subtitle[0]}:{title}"
    plt.suptitle(title)
    plt.plot(cs_df.index, cs_df.values, label=cs_df.columns.tolist())
    plt.xlabel("Episodes")
    plt.ylabel("Cumulative Return")
    plt.legend(loc=4)

    file_name = output_folder_path / "cumulative_returns.pdf"
    plt.savefig(file_name, bbox_inches="tight", pad_inches=0)
    file_name = output_folder_path / "cumulative_returns.png"
    plt.savefig(file_name, bbox_inches="tight", pad_inches=0)
    plt.show()

    """Mean Average of Returns(Mean Rewards)"""
    sma_df = df.copy().rolling(50).mean()
    title = "Simple Mean Average Return"
    if len(args.subtitle) > 0:
        title = f"{args.subtitle[0]}:{title}"
    plt.suptitle(title)
    plt.plot(sma_df.index, sma_df.values, label=sma_df.columns.tolist())
    plt.xlabel("Episodes")
    plt.ylabel("Simple Mean Average (M=50)")
    plt.legend(loc=4)

    file_name = output_folder_path / "sma_returns.pdf"
    plt.savefig(file_name, bbox_inches="tight", pad_inches=0)
    file_name = output_folder_path / "sma_returns.png"
    plt.savefig(file_name, bbox_inches="tight", pad_inches=0)
    plt.show()


if __name__ == "__main__":
    main()
