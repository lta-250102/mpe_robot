from env.make_env import make_env
import argparse, datetime
import wandb
import numpy as np
import torch
import os
from pettingzoo.mpe import simple_spread_v3, simple_crypto_v3, simple_adversary_v3, simple_push_v3, simple_reference_v3, simple_tag_v3, simple_world_comm_v3
from algo.bicnet.bicnet_agent import BiCNet
from algo.commnet.commnet_agent import CommNet
from algo.maddpg.maddpg_agent import MADDPG

from algo.normalized_env import ActionNormalizedEnv, ObsEnv, reward_from_state

from algo.utils import *
from copy import deepcopy


def main(args):
    print(f'device: {device}')

    if args.scenario == "simple_spread":
        env = simple_spread_v3.parallel_env(render_mode=None, max_cycles = args.max_episodes, N=args.n_agents)
    elif args.scenario == "simple_crypto":
        env = simple_crypto_v3.parallel_env(render_mode=None, max_cycles = args.max_episodes)
    elif args.scenario == "simple_adversary":
        env = simple_adversary_v3.parallel_env(render_mode=None, max_cycles = args.max_episodes)
    elif args.scenario == "simple_push":
        env = simple_push_v3.parallel_env(render_mode=None, max_cycles = args.max_episodes)
    elif args.scenario == "simple_reference":
        env = simple_reference_v3.parallel_env(render_mode=None, max_cycles = args.max_episodes)
    elif args.scenario == "simple_tag":
        env = simple_tag_v3.parallel_env(render_mode=None, max_cycles = args.max_episodes)
    # elif args.scenario == "simple_world_comm": # error
    #     env = simple_world_comm_v3.parallel_env(render_mode=None, max_cycles = args.max_episodes)
    env.reset(seed=42)
    n_agents = len(env.possible_agents)
    n_actions = list(env.action_spaces.values())[0].n
    n_states = [ob.shape[0] for ob in env.observation_spaces.values()]


    torch.manual_seed(args.seed)

    if args.wandb and args.mode == "train":
        wandb.init(
            project="robot_mpe",
            # entity='diogenes-student',
            name=f"{args.algo}_{args.scenario}_{args.n_agents}",
        )

    if args.algo == "bicnet":
        model = BiCNet(n_states, n_actions, n_agents, args)

    if args.algo == "commnet":
        model = CommNet(n_states, n_actions, n_agents, args)

    if args.algo == "maddpg":
        model = MADDPG(n_states, n_actions, n_agents, args)

    print(model)
    model.load_model()

    episode = 0
    total_step = 0

    while episode < args.max_episodes:

        states_dict, _ = env.reset()
        state = [state for state in states_dict.values()]

        episode += 1
        step = 0
        accum_reward = 0
        rewards = [0 for _ in range(n_agents)]
        while True:

            if args.mode == "train":
                action = model.choose_action(state, noisy=True)
                actions_SN = [np.argmax(onehot) for onehot in action]
                count_agent = len(env.agents)
                actions_dict = {env.agents[i]: actions_SN[i] for i in range(env.max_num_agents)}
                next_state, reward, done, info, ___ = env.step(actions_dict)
                next_state = [state for state in next_state.values()]
                reward = [state for state in reward.values()]
                done = [state for state in done.values()]

                step += 1
                total_step += 1
                reward = np.array(reward)

                rew1 = 0
                reward = rew1 + (np.array(reward, dtype=np.float32) / 100.)
                accum_reward += sum(reward)
                rewards = [rewards[i] + reward[i] for i in range(n_agents)]


                if args.algo == "maddpg" or args.algo == "commnet":
                    obs = [torch.tensor(s).float().to(device) for s in state]
                    obs_ = [torch.tensor(s).float().to(device) for s in next_state]
                    if step != args.episode_length - 1:
                        next_obs = obs_
                    else:
                        next_obs = None
                    rw_tensor = torch.FloatTensor(reward).to(device)
                    ac_tensor = torch.FloatTensor(action).to(device)
                    if args.algo == "commnet" and next_obs is not None:
                        model.memory.push(obs, ac_tensor, next_obs, rw_tensor)
                    if args.algo == "maddpg":
                        model.memory.push(obs, ac_tensor, next_obs, rw_tensor)
                    obs = next_obs
                else:
                    model.memory(state, action, reward, next_state, done)

                state = next_state

                if args.episode_length < step or (True in done):
                    c_loss, a_loss = model.update(episode)

                    print("[Episode %05d] reward %6.4f" % (episode, accum_reward))
                    if args.wandb:
                        wandb.log({"reward": accum_reward}, step=episode + args.model_episode)
                        for i in range(n_agents):
                            wandb.log({"reward_" + str(i): rewards[i]}, step=episode + args.model_episode)
                        if c_loss and a_loss:
                            wandb.log({"actor_loss": a_loss, "critic_loss": c_loss}, step=episode + args.model_episode)

                    if c_loss and a_loss:
                        print(" a_loss %3.2f c_loss %3.2f" % (a_loss, c_loss), end='')


                    if episode % args.save_interval == 0 and args.mode == "train":
                        model.save_model(episode)

                    env.reset()
                    break
            elif args.mode == "eval":
                action = model.choose_action(state, noisy=True)
                actions_SN = [np.argmax(onehot) for onehot in action]
                actions_dict = {env.agents[i]: actions_SN[i] for i in range(env.max_num_agents)}
                next_state, reward, done, info, ___ = env.step(actions_dict)
                next_state = [state for state in next_state.values()]
                reward = [state for state in reward.values()]
                done = [state for state in done.values()]
                
                step += 1
                total_step += 1
                state = next_state
                reward = np.array(reward)
                import time
                time.sleep(0.02)
                env.render()

                rew1 = 0
                reward = rew1 + (np.array(reward, dtype=np.float32) / 100.)
                accum_reward += sum(reward)

                if args.wandb:
                    wandb.log({"reward": accum_reward}, step=episode + args.model_episode)
                    for i in range(n_agents):
                        wandb.log({"reward_" + str(i): rewards[i]}, step=episode + args.model_episode)
                    if c_loss and a_loss:
                        wandb.log({"actor_loss": a_loss, "critic_loss": c_loss}, step=episode + args.model_episode)

                if args.episode_length < step or (True in done):
                    print("[Episode %05d] reward %6.4f " % (episode, accum_reward))
                    env.reset()
                    break

    if args.wandb:
        wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', default="simple_crypto", type=str, help="simple_spread/simple_crypto/simple_adversary/simple_push/simple_reference/simple_tag")
    parser.add_argument('--n_agents', default=3, type=int)
    parser.add_argument('--max_episodes', default=1e10, type=int)
    parser.add_argument('--algo', default="maddpg", type=str, help="commnet/bicnet/maddpg")
    parser.add_argument('--mode', default="train", type=str, help="train/eval")
    parser.add_argument('--episode_length', default=50, type=int)
    parser.add_argument('--memory_length', default=int(1e5), type=int)
    parser.add_argument('--tau', default=0.001, type=float)
    parser.add_argument('--gamma', default=0.95, type=float)
    parser.add_argument('--seed', default=777, type=int)
    parser.add_argument('--a_lr', default=0.0001, type=float)
    parser.add_argument('--c_lr', default=0.0001, type=float)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--render_flag', default=False, type=bool)
    parser.add_argument('--ou_theta', default=0.15, type=float)
    parser.add_argument('--ou_mu', default=0.0, type=float)
    parser.add_argument('--ou_sigma', default=0.2, type=float)
    parser.add_argument('--epsilon_decay', default=10000, type=int)
    parser.add_argument('--tensorboard', default=True, action="store_true")
    parser.add_argument('--wandb', default=True, action="store_true")
    parser.add_argument("--save_interval", default=50, type=int)
    parser.add_argument("--model_episode", default=0, type=int)
    parser.add_argument('--episode_before_train', default=200, type=int)
    parser.add_argument('--log_dir', default=datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))

    args = parser.parse_args()
    main(args)
