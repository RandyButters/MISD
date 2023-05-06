import argparse

parser = argparse.ArgumentParser()

# Environment
parser.add_argument("--env_name", default='AntMaze-v1', type=str)
parser.add_argument("--reward_shaping", type=str, default="dense", choices=["dense", "sparse"])
parser.add_argument("--stochastic_xy", action="store_true")
parser.add_argument("--stochastic_sigma", default=0., type=float)
parser.add_argument("--gid", type=int, default=0)
parser.add_argument("--algo", default="test", type=str, choices=["higl", "hrac", "hiro", "misd", "test"])
parser.add_argument("--seed", default=2, type=int)
parser.add_argument("--eval_freq", default=5e3, type=float)
parser.add_argument("--max_timesteps", default=5e6, type=float)

# Off-policy correction (from HIRO)
parser.add_argument("--no_correction", action="store_true")
parser.add_argument("--inner_dones", action="store_true")
parser.add_argument("--absolute_goal", action="store_true")
parser.add_argument("--binary_int_reward", action="store_true")

# Manager Parameters
parser.add_argument("--man_tau", default=0.005, type=float)
parser.add_argument("--man_batch_size", default=128, type=int)
parser.add_argument("--man_buffer_size", default=2e5, type=int)
parser.add_argument("--man_rew_scale", default=0.1, type=float)
parser.add_argument("--man_act_lr", default=1e-4, type=float)
parser.add_argument("--man_crit_lr", default=1e-3, type=float)
parser.add_argument("--candidate_goals", default=10, type=int)
parser.add_argument("--manager_propose_freq", "-k", default=10, type=int)
parser.add_argument("--train_manager_freq", default=10, type=int)
parser.add_argument("--discount", default=0.99, type=float)

# Controller Parameters
parser.add_argument("--ctrl_tau", default=0.005, type=float)
parser.add_argument("--ctrl_batch_size", default=128, type=int)
parser.add_argument("--ctrl_buffer_size", default=2e5, type=int)
parser.add_argument("--ctrl_rew_scale", default=1.0, type=float)
parser.add_argument("--ctrl_act_lr", default=1e-4, type=float)
parser.add_argument("--ctrl_crit_lr", default=1e-3, type=float)
parser.add_argument("--ctrl_discount", default=0.95, type=float)

# Noise Parameters
parser.add_argument("--noise_type", default="normal", type=str, choices=["normal", "ou"])
parser.add_argument("--ctrl_noise_sigma", default=1., type=float)
parser.add_argument("--man_noise_sigma", default=1., type=float)
parser.add_argument("--train_ctrl_policy_noise", default=0.2, type=float)
parser.add_argument("--train_ctrl_noise_clip", default=0.5, type=float)
parser.add_argument("--train_man_policy_noise", default=0.2, type=float)
parser.add_argument("--train_man_noise_clip", default=0.5, type=float)

# MISD
parser.add_argument("--lr_i", type=float, default=2e-4)
parser.add_argument("--i_training_epochs", type=int, default=25)
parser.add_argument("--i_batch_size", type=int, default=64)
parser.add_argument("--i_hidden_dim", type=int, default=128)
parser.add_argument("--i_embedding_dim", type=int, default=32)
parser.add_argument("--i_neg_key_num", default=50, type=int)
parser.add_argument("--i_temperature", default=0.1, type=int)
parser.add_argument("--bound_coeff", default=0.1, type=float)
    
# Adjacency Network (from HRAC)
parser.add_argument("--traj_buffer_size", type=int, default=50000)
parser.add_argument("--lr_r", type=float, default=2e-4)
parser.add_argument("--r_margin_pos", type=float, default=1.0)
parser.add_argument("--r_margin_neg", type=float, default=1.2)
parser.add_argument("--r_training_epochs", type=int, default=25)
parser.add_argument("--r_batch_size", type=int, default=64)
parser.add_argument("--r_hidden_dim", type=int, default=128)
parser.add_argument("--r_embedding_dim", type=int, default=32)
parser.add_argument("--goal_loss_coeff", type=float, default=20.)

# HIGL
parser.add_argument("--landmark_loss_coeff", default=20., type=float)
parser.add_argument("--delta", type=float, default=2)
parser.add_argument("--adj_factor", default=0.5, type=float)

# HIGL: Planner, Coverage
parser.add_argument("--landmark_sampling", type=str, choices=["fps", "none"])
parser.add_argument('--clip_v', type=float, default=-38., help="clip bound for the planner")
parser.add_argument("--n_landmark_coverage", type=int, default=20)
parser.add_argument("--initial_sample", type=int, default=1000)
parser.add_argument("--goal_thr", type=float, default=-10.)
parser.add_argument("--planner_start_step", type=int, default=60000)

# HIGL: Novelty
parser.add_argument("--novelty_algo", type=str, default="none", choices=["rnd", "none"])
parser.add_argument("--use_novelty_landmark", action="store_true")
parser.add_argument("--close_thr", type=float, default=0.2)
parser.add_argument("--n_landmark_novelty", type=int, default=20)
parser.add_argument("--rnd_output_dim", type=int, default=128)
parser.add_argument("--rnd_lr", type=float, default=1e-3)
parser.add_argument("--rnd_batch_size", default=128, type=int)
parser.add_argument("--use_ag_as_input", action="store_true")

# Ablation
parser.add_argument("--no_pseudo_landmark", action="store_true")
parser.add_argument("--discard_by_anet", action="store_true")
parser.add_argument("--automatic_delta_pseudo", action="store_true")

# Save
parser.add_argument("--save_models", action="store_true")
parser.add_argument("--save_dir", default="./models", type=str)
parser.add_argument("--save_replay_buffer", type=str)

# Load
parser.add_argument("--load", action="store_true")
parser.add_argument("--load_dir", default="./models", type=str)
parser.add_argument("--load_algo", type=str)
parser.add_argument("--log_dir", default="./logs", type=str)
parser.add_argument("--load_replay_buffer", type=str)
parser.add_argument("--load_adj_net", default=False, action="store_true")
parser.add_argument("--load_info_net", default=False, action="store_true")

parser.add_argument("--version", type=str, default='dense')

args = parser.parse_args()

if args.load_algo is None:
    args.load_algo = args.algo
    
if args.reward_shaping == "sparse":
    args.man_rew_scale = 1.0
    
# Run the algorithm

import torch

import os
import numpy as np
import pandas as pd
import json
import misd.utils as utils
import misd.misd as misd
from misd.models import ANet, INet

import gym
from goal_env import *
from goal_env.mujoco import *

from envs import EnvWithGoal



def run_misd_eval(args):
    if not os.path.exists("./data"):
        os.makedirs("./data")


    if "Ant" in args.env_name:
        step_style = args.reward_shaping == 'sparse'
        env = EnvWithGoal(gym.make(args.env_name,
                                   stochastic_xy=args.stochastic_xy,
                                   stochastic_sigma=args.stochastic_sigma),
                          env_name=args.env_name, step_style=step_style)
    elif "Point" in args.env_name:
        assert not args.stochastic_xy
        step_style = args.reward_shaping == 'sparse'
        env = EnvWithGoal(gym.make(args.env_name), env_name=args.env_name, step_style=step_style)
        
    max_action = float(env.action_space.high[0])
    train_ctrl_policy_noise = args.train_ctrl_policy_noise
    train_ctrl_noise_clip = args.train_ctrl_noise_clip

    train_man_policy_noise = args.train_man_policy_noise
    train_man_noise_clip = args.train_man_noise_clip


    if "AntMaze" in args.env_name or "PointMaze" in args.env_name:
        high = np.array((10., 10.))
        low = - high
    else:
        raise NotImplementedError

    man_scale = (high - low) / 2
    absolute_goal_scale = 0


    no_xy = True

    obs = env.reset()
    print("obs: ", obs)

    goal = obs["desired_goal"]
    state = obs["observation"]

    controller_goal_dim = obs["achieved_goal"].shape[0]

    # Write Hyperparameters to file
    print("---------------------------------------")
    print("Current Arguments:")

    torch.cuda.set_device(args.gid)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    file_name = "{}_{}_{}".format(args.env_name, args.algo, args.seed)
    output_data = {"frames": [], "reward": [], "dist": []}

    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    state_dim = state.shape[0]
    goal_dim = goal.shape[0]
    action_dim = env.action_space.shape[0]


    if "Ant" in args.env_name:
        calculate_controller_reward = utils.get_reward_function(env, args.env_name,
                                                                absolute_goal=args.absolute_goal,
                                                                binary_reward=args.binary_int_reward)
    else:
        raise NotImplementedError

    controller_policy = misd.Controller(
        state_dim=state_dim,
        goal_dim=controller_goal_dim,
        action_dim=action_dim,
        max_action=max_action,
        actor_lr=args.ctrl_act_lr,
        critic_lr=args.ctrl_crit_lr,
        no_xy=no_xy,
        absolute_goal=args.absolute_goal,
        policy_noise=train_ctrl_policy_noise,
        noise_clip=train_ctrl_noise_clip,
    )

    manager_policy = misd.Manager(
        state_dim=state_dim,
        goal_dim=goal_dim,
        action_dim=controller_goal_dim,
        actor_lr=args.man_act_lr,
        critic_lr=args.man_crit_lr,
        candidate_goals=args.candidate_goals,
        correction=not args.no_correction,
        scale=man_scale,
        goal_loss_coeff=args.goal_loss_coeff,
        absolute_goal=args.absolute_goal,
        absolute_goal_scale=absolute_goal_scale,
        landmark_loss_coeff=args.landmark_loss_coeff,
        delta=args.delta,
        policy_noise=train_man_policy_noise,
        noise_clip=train_man_noise_clip,
        no_pseudo_landmark=args.no_pseudo_landmark,
        automatic_delta_pseudo=args.automatic_delta_pseudo,
        planner_start_step=args.planner_start_step,
        planner_cov_sampling=args.landmark_sampling,
        planner_clip_v=args.clip_v,
        n_landmark_cov=args.n_landmark_coverage,
        planner_initial_sample=args.initial_sample,
        planner_goal_thr=args.goal_thr,
        bound_coeff=args.bound_coeff,
    )


    for steps in range(50000, 2000000, 50000):
        i_net = INet(controller_goal_dim, args.i_hidden_dim, args.i_embedding_dim)
        i_net.load_state_dict(torch.load("{}/{}_{}_{}_{}_i_network.pth".format(args.load_dir,
                                                                                    args.env_name,
                                                                                    args.algo,
                                                                                    args.version,
                                                                                    steps)))
        i_net.to(device)
        manager_policy.load(args.load_dir, args.env_name, args.load_algo, args.version, steps)
        controller_policy.load(args.load_dir, args.env_name, args.load_algo, args.version, steps)
        print("Loaded successfully.")
        env.evaluate = True
        with torch.no_grad():
            avg_reward = 0.
            avg_controller_rew = 0.
            global_steps = 0
            goals_achieved = 0
            
            for eval_ep in range(5):
                obs = env.reset()
                output_data = {'t':[], 'ag':[], 'sg':[], 'i_ag':[]}
                goal = obs["desired_goal"]
                achieved_goal = obs["achieved_goal"]
                state = obs["observation"]

                done = False
                step_count = 0
                env_goals_achieved = 0
                while not done:
                    if step_count % 10 == 0:
                        subgoal = manager_policy.sample_goal(state, goal)
                        output_data['sg'].append((subgoal+achieved_goal).tolist())
                    output_data['t'].append(step_count)
                    output_data['ag'].append(achieved_goal.tolist())
                    output_data['i_ag'].append(i_net(torch.FloatTensor(achieved_goal).to(device)).cpu().numpy().tolist())
                    
                    step_count += 1
                    global_steps += 1
                    action = controller_policy.select_action(state, subgoal)
                    new_obs, reward, done, info = env.step(action)
                    is_success = info['is_success']
                    if is_success:
                        env_goals_achieved += 1
                        goals_achieved += 1
                        done = True

                    goal = new_obs["desired_goal"]
                    new_achieved_goal = new_obs['achieved_goal']
                    new_state = new_obs["observation"]

                    subgoal = controller_policy.subgoal_transition(achieved_goal, subgoal, new_achieved_goal)

                    avg_reward += reward
                    avg_controller_rew += calculate_controller_reward(achieved_goal, subgoal, new_achieved_goal,
                                                                    args.ctrl_rew_scale, action)
                    state = new_state
                    achieved_goal = new_achieved_goal
                json_file_path = './data/' + str(steps)
                if not os.path.exists(json_file_path):
                    os.makedirs(json_file_path)
                json_str = json.dumps(output_data, ensure_ascii=False, indent=4)
                with open(json_file_path + '/' + str(eval_ep) + '.json', 'w', encoding='utf-8') as f:
                    f.write(json_str)
            avg_reward /= 5
            avg_controller_rew /= global_steps
            avg_step_count = global_steps / 5
            avg_env_finish = goals_achieved / 5

            print("---------------------------------------")
            print("Evaluation over {} episodes:\nAvg Ctrl Reward: {:.3f}".format(5, avg_controller_rew))
    
            print("Goals achieved: {:.1f}%".format(100*avg_env_finish))
            print("Avg Steps to finish: {:.1f}".format(avg_step_count))
            print("---------------------------------------")

            env.evaluate = False

    print("Training finished.")


if __name__ == '__main__':
    run_misd_eval(args)