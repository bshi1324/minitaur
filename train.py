#!/usr/bin/env python3
import argparse
import gym
import os
import pybullet_envs
import time
from tensorboardX import SummaryWriter
import numpy as np

import ptan
from ptan.common.utils import RewardTracker, TBMeanTracker
from ptan.experience import ExperienceSourceFirstLast, PrioritizedReplayBuffer, ExperienceReplayBuffer

import torch
import torch.nn.functional as F
import torch.optim as optim

from lib import model, utils


ENV_ID = 'MinitaurBulletEnv-v0'
GAMMA = 0.99
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
REPLAY_SIZE = 100000
REPLAY_INITIAL = 1000
REWARD_STEPS = 5

# priority replay
PRIO_REPLAY_ALPHA = 0.6
BETA_START = 0.4
BETA_FRAMES = 100000

TEST_ITERS = 1000

VMAX = 10
VMIN = -10
N_ATOMS = 51
DELTA = (VMAX - VMIN) / (N_ATOMS - 1)


def calc_loss(batch, batch_weights, act_net, crt_net, tgt_act_net, tgt_crt_net, device='cpu'):
    states, actions, rewards, dones_mask, last_states = utils.unpack_batch(batch, device)
    batch_weights = torch.tensor(batch_weights).to(device)

    # critic loss
    crt_distr = crt_net(states, actions)
    last_act = tgt_act_net.target_model(last_states)
    last_distr = F.softmax(tgt_crt_net.target_model(last_states, last_act), dim=1)
    proj_distr = distr_projection(last_distr, rewards, dones_mask, gamma=GAMMA ** REWARD_STEPS, device=device)
    prob_distr = - F.log_softmax(crt_distr, dim=1) * proj_distr
    critic_loss = prob_distr.sum(dim=1).mean()
    td_errors = prob_distr.sum(dim=1) * batch_weights

    # actor loss
    cur_actions = act_net(states)
    crt_distr = crt_net(states, cur_actions)
    actor_loss = - crt_net.distr_to_q(crt_distr)
    
    return actor_loss.mean(), critic_loss, td_errors + 1e-5


def test(net, env, count=10, device='cpu'):
    net.eval()
    rewards = 0.0
    steps = 0
    for _ in range(count):
        obs = env.reset()
        done = False
        while True:
            obs = torch.tensor(np.array(obs, dtype=np.float32)).to(device)
            mu = net(obs)
            action = mu.squeeze(dim=0).data.cpu().numpy()
            action = np.clip(action, -1, 1)
            obs, reward, done, _ = env.step(action)
            reward += reward
            steps += 1
            if done:
                break
    return rewards / count, steps / count


def train(actor_loss, critic_loss, act_net, crt_net, tgt_act_net, tgt_crt_net, act_opt, crt_opt, device='cpu'):
    
    act_net.train()
    crt_net.train()

    # train critic
    crt_opt.zero_grad()
    critic_loss.backward()
    crt_opt.step()

    # train actor
    act_opt.zero_grad()
    actor_loss.backward()
    act_opt.step()

    tgt_act_net.alpha_sync(alpha=1 - 1e-3)
    tgt_crt_net.alpha_sync(alpha=1 - 1e-3)


def distr_projection(next_distr, rewards, dones_mask, gamma, device='cpu'):
    next_distr = next_distr.data.cpu().numpy()
    rewards = rewards.data.cpu().numpy()
    dones_mask = dones_mask.cpu().numpy().astype(np.bool)
    batch_size = len(rewards)
    proj_distr = np.zeros((batch_size, N_ATOMS), dtype=np.float32)

    for atom in range(N_ATOMS):
        tz = np.minimum(VMAX, np.maximum(VMIN, rewards + (VMIN + atom * DELTA) * gamma))
        b = (tz - VMIN) / DELTA
        l = np.floor(b).astype(np.int64)
        u = np.ceil(b).astype(np.int64)
        eq_mask = u == l
        proj_distr[eq_mask, l[eq_mask]] += next_distr[eq_mask, atom]
        ne_mask = u != l
        proj_distr[ne_mask, l[ne_mask]] += next_distr[ne_mask, atom] * (u - b)[ne_mask]
        proj_distr[ne_mask, u[ne_mask]] += next_distr[ne_mask, atom] * (b - l)[ne_mask]

    if dones_mask.any():
        proj_distr[dones_mask] = 0.0
        tz = np.minimum(VMAX, np.maximum(VMIN, rewards[dones_mask]))
        b = (tz - VMIN) / DELTA
        l = np.floor(b).astype(np.int64)
        u = np.ceil(b).astype(np.int64)
        eq_mask = u == l
        eq_dones = dones_mask.copy()
        eq_dones[dones_mask] = eq_mask
        if eq_dones.any():
            proj_distr[eq_dones, l] = 1.0
        ne_mask = u != l
        ne_dones = dones_mask.copy()
        ne_dones[dones_mask] = ne_mask
        if ne_dones.any():
            proj_distr[ne_dones, l] = (u - b)[ne_mask]
            proj_distr[ne_dones, u] = (b - l)[ne_mask]
    return torch.FloatTensor(proj_distr).to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action='store_true', help='Enable CUDA')
    parser.add_argument("-n", "--name", required=True, help="Name of the run")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    save_path = os.path.join("saves", "d4pg-" + args.name)
    os.makedirs(save_path, exist_ok=True)

    env = gym.make(ENV_ID)
    test_env = gym.make(ENV_ID) 
    
    act_net = model.D4PGActor(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
    crt_net = model.D4PGCritic(env.observation_space.shape[0], env.action_space.shape[0], N_ATOMS, VMIN, VMAX).to(device)
    tgt_act_net = ptan.agent.TargetNet(act_net)
    tgt_crt_net = ptan.agent.TargetNet(crt_net)

    writer = SummaryWriter(comment="-d4pg_" + args.name)

    agent = model.AgentD4PG(act_net, device=device)
    exp_source = ExperienceSourceFirstLast(env, agent, gamma=GAMMA, steps_count=REWARD_STEPS)
    buffer = PrioritizedReplayBuffer(exp_source, REPLAY_SIZE, PRIO_REPLAY_ALPHA)
    act_opt = optim.Adam(act_net.parameters(), lr=LEARNING_RATE)
    crt_opt = optim.Adam(crt_net.parameters(), lr=LEARNING_RATE)

    frame_idx = 0
    best_reward = 0
    with RewardTracker(writer) as tracker:
        with TBMeanTracker(writer, batch_size=10) as tb_tracker:
            while True:
                frame_idx += 1
                buffer.populate(1)
                beta = min(1.0, BETA_START + frame_idx * (1.0 - BETA_START) / BETA_FRAMES)

                rewards_steps = exp_source.pop_rewards_steps()
                if rewards_steps:
                    rewards, steps = zip(*rewards_steps)
                    tb_tracker.track('episode_steps', steps[0], frame_idx)
                    tracker.reward(rewards[0], frame_idx)

                if len(buffer) < REPLAY_INITIAL:
                    continue

                batch, batch_indices, batch_weights = buffer.sample(BATCH_SIZE, beta)
                actor_loss, critic_loss, sample_prios = calc_loss(batch, batch_weights, act_net, crt_net, tgt_act_net, tgt_crt_net, device)

                train(actor_loss, critic_loss, act_net, crt_net, tgt_act_net, tgt_crt_net, act_opt, crt_opt, device)

                tb_tracker.track('loss_actor', actor_loss, frame_idx)
                tb_tracker.track('loss_critic', critic_loss, frame_idx)

                buffer.update_priorities(batch_indices, sample_prios.data.cpu().numpy())

                if frame_idx % TEST_ITERS == 0:
                    ts = time.time()
                    rewards, steps = test(act_net, test_env, device=device)
                    print('Test done in %.2f sec, reward %.3f, steps %d' % (
                        time.time() - ts, rewards, steps))
                    writer.add_scalar('test_reward', rewards, frame_idx)
                    writer.add_scalar('test_steps', steps, frame_idx)
                    if best_reward is None or best_reward < rewards:
                        if best_reward is not None:
                            print('Best reward updated: %.3f -> %.3f' % (best_reward, rewards))
                            name = 'best_%+.3f_%d.dat' % (rewards, frame_idx)
                            fname = os.path.join(save_path, name)
                            torch.save(act_net.state_dict(), fname)
                        best_reward = rewards

    writer.close()


if __name__ == '__main__':
    main()

