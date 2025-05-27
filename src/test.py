#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------- Start of CleanRL LICENSE ----------------
# MIT License
#
# Copyright (c) 2019 CleanRL developers
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ---------------- End of CleanRL LICENSE ----------------
# @Time    : 2023/9/25 20:50
# @Author  : Guangchen Jiang
# @Email   : guangchen98.jiang@gmail.com
# @File    : src/c51/c51.py
# @Software: PyCharm

import argparse
import os
import random
import sys
import time
from typing import Iterator, Tuple, TypeVar

import gymnasium as gym
import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from distutils.util import strtobool
from lightning.pytorch.loggers import TensorBoardLogger
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), r"")))

ObsType = TypeVar("ObsType")


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help="the name of this experiment.")
    parser.add_argument("--seed", type=int, default=1,
                        help="seed of the experiment.")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, `torch.backends.cudnn.deterministic=False`.")
    # parser.add_argument("--multi-gpus", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
    #                     help="whether to use multiple Gpus for training.")
    parser.add_argument("--gpu-id", type=int, default=0, nargs="?", const=True,
                        help="which gpu to use.")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, this experiment will be tracked with Weights and Biases.")
    parser.add_argument("--wandb-project-name", type=str, default="DRL--c51",
                        help="the wandb's project name.")
    parser.add_argument("--wandb-entity", type=str, default=None,
                        help="the entity (team) of wandb's project.")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="whether to capture videos of the agent performances (check out `videos` folder).")
    parser.add_argument("--save-model", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="whether to save model into the `distributional_DQN/res/c51/models/{run_name}` folder.")
    parser.add_argument("--hf-entity", type=str, default="",
                        help="the user or org name of the model repository from the Hugging Face Hub.")
    parser.add_argument("--save-prefix", type=str, default="../../..",
                        help="storage location.")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="CartPole-v1",
                        help="the id of the environment.")
    parser.add_argument("--total-timesteps", type=int, default=500000,
                        help="total timesteps of the experiments.")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
                        help="the learning rate of the optimizer.")
    parser.add_argument("--num-envs", type=int, default=1,
                        help="the number of parallel game environments.")
    parser.add_argument("--n-atoms", type=int, default=101,
                        help="the number of atoms.")
    parser.add_argument("--v-min", type=float, default=-100,
                        help="the return lower bound.")
    parser.add_argument("--v-max", type=float, default=100,
                        help="the return upper bound.")
    parser.add_argument("--buffer-size", type=int, default=10000,
                        help="the replay memory buffer size.")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="the discount factor gamma.")
    parser.add_argument("--sync-rate", type=int, default=500,
                        help="the timesteps it takes to update the target network.")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="the batch size of sample from the reply memory.")
    parser.add_argument("--start-e", type=float, default=1,
                        help="the starting epsilon for exploration.")
    parser.add_argument("--end-e", type=float, default=0.05,
                        help="the ending epsilon for exploration.")
    parser.add_argument("--exploration-fraction", type=float, default=0.5,
                        help="the fraction of `total-timesteps` it takes from start-e to go end-e.")
    parser.add_argument("--learning-starts", type=int, default=10000,
                        help="timestep to start learning.")
    parser.add_argument("--train-frequency", type=int, default=10,
                        help="the frequency of training.")
    args = parser.parse_args()
    # fmt: on
    assert args.num_envs == 1, "vectorized envs are not supported at the moment."

    return args


def make_env(
    env_id: str,
    seed: int,
    idx: int,
    capture_video: bool,
    run_name: str,
    save_prefix: str = "../../..",
):
    def thunk() -> gym.Env:
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(
                env,
                f"{save_prefix}/res/benchmarks/c51/videos/{run_name}",
                disable_logger=True,
            )
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)

        return env

    return thunk


# Neural Network Structure
class QNetwork(nn.Module):
    def __init__(
        self, input_dim: int, output_dim: int, v_min: int, v_max: int, n_atoms: int
    ):
        super().__init__()
        self.register_buffer("atoms", torch.linspace(v_min, v_max, steps=n_atoms))
        self.l1 = nn.Sequential(
            nn.Linear(input_dim, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, output_dim),
        )

    def forward(self, x):
        return self.l1(x)


class RLDataset(IterableDataset):
    def __init__(self, buffer: ReplayBuffer, sample_size: int = 400) -> None:
        self.buffer = buffer
        self.sample_size = sample_size

    def __iter__(self) -> Iterator[Tuple]:
        observations, actions, next_observations, dones, rewards = self.buffer.sample(
            self.sample_size
        )
        for i in range(len(dones)):
            yield (
                observations[i],
                actions[i],
                next_observations[i],
                dones[i],
                rewards[i],
            )


# ALGO LOGIC: initialize agent here:
class C51ALGO(pl.LightningModule):
    def __init__(self, args: argparse.Namespace, run_name: str):
        super().__init__()
        self.args = args
        # env setup
        self.envs = gym.vector.SyncVectorEnv(
            [
                make_env(
                    self.args.env_id,
                    self.args.seed + _i,
                    _i,
                    self.args.capture_video,
                    run_name,
                )
                for _i in range(self.args.num_envs)
            ]
        )
        assert isinstance(self.envs.single_action_space, gym.spaces.Discrete), (
            "only discrete action space is supported"
        )

        self.n = self.envs.single_action_space.n
        input_dim = np.array(self.envs.single_observation_space.shape).prod()
        self.q_network = QNetwork(
            int(input_dim),
            self.args.n_atoms * self.n,
            self.args.v_min,
            self.args.v_max,
            self.args.n_atoms,
        )
        self.target_network = QNetwork(
            int(input_dim),
            self.args.n_atoms * self.n,
            self.args.v_min,
            self.args.v_max,
            self.args.n_atoms,
        )
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.rb = ReplayBuffer(
            self.args.buffer_size,
            self.envs.single_observation_space,
            self.envs.single_action_space,
            handle_timeout_termination=False,
        )
        self.obs, _ = self.envs.reset(seed=self.args.seed)

        self.start_time = time.time()

        self.populate(self.args.learning_starts)
        self.save_hyperparameters(vars(self.args))

    def get_action(
        self, net: nn.Module, x: torch.Tensor, action=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = net(x)
        # probability mass function for each action
        pmfs = F.softmax(logits.view(-1, self.n, self.args.n_atoms), dim=2)
        q_values = (pmfs * net.atoms).sum(2)
        if action is None:
            action = torch.argmax(q_values, 1)
        return action, pmfs[torch.arange(len(x)), action]

    def qnet_step(self, step: int):
        epsilon = self.action_linear_schedule(step)
        if random.random() < epsilon:
            actions = np.array(
                [
                    self.envs.single_action_space.sample()
                    for _ in range(self.envs.num_envs)
                ]
            )
        else:
            actions, pmf = self.get_action(
                self.q_network, torch.Tensor(self.obs).to(self.device)
            )
            actions = actions.cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminated, truncated, infos = self.envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                # Skip the envs that are not done
                if "episode" not in info:
                    continue
                if step > self.args.learning_starts:
                    self.log_dict(
                        {
                            "charts/episodic_return": float(
                                info["episode"]["r"].item()
                            ),
                            "charts/episodic_length": float(
                                info["episode"]["l"].item()
                            ),
                            "charts/epsilon": epsilon,
                        }
                    )

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, d in enumerate(truncated):
            if d:
                real_next_obs[idx] = infos["final_observation"][idx]
        self.rb.add(self.obs, real_next_obs, actions, rewards, terminated, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        self.obs = next_obs

    def populate(self, steps: int = 10000) -> None:
        # ALGO LOGIC: put action logic here
        for s in range(steps):
            self.qnet_step(s)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        observations, actions, next_observations, dones, rewards = batch

        with torch.no_grad():
            _, next_pmfs = self.get_action(self.target_network, next_observations)
            next_atoms = rewards + self.args.gamma * self.target_network.atoms * (
                1 - dones
            )
            # projection
            delta_z = self.target_network.atoms[1] - self.target_network.atoms[0]
            tz = next_atoms.clamp(self.args.v_min, self.args.v_max)

            b = (tz - self.args.v_min) / delta_z
            l = b.floor().clamp(0, self.args.n_atoms - 1)
            u = b.ceil().clamp(0, self.args.n_atoms - 1)
            # (l == u).float() handles the case where bj is exactly an integer
            # example bj = 1, then the upper ceiling should be uj= 2, and lj= 1
            d_m_l = (u + (l == u).float() - b) * next_pmfs
            d_m_u = (b - l) * next_pmfs
            target_pmfs = torch.zeros_like(next_pmfs)
            for i in range(target_pmfs.size(0)):
                target_pmfs[i].index_add_(0, l[i].long(), d_m_l[i])
                target_pmfs[i].index_add_(0, u[i].long(), d_m_u[i])

        _, old_pmfs = self.get_action(self.q_network, observations, actions.flatten())
        loss = (
            -(target_pmfs * old_pmfs.clamp(min=1e-5, max=1 - 1e-5).log()).sum(-1)
        ).mean()

        self.log_dict(
            {
                "losses/loss": loss.item(),
                "losses/q_values": (old_pmfs * self.q_network.atoms)
                .sum(1)
                .mean()
                .item(),
                "charts/SPS": float(
                    (
                        self.args.learning_starts
                        + self.global_step * self.args.train_frequency
                    )
                    / (time.time() - self.start_time)
                ),
            }
        )

        with torch.no_grad():
            for tfs in range(self.args.train_frequency):
                self.qnet_step(
                    self.args.learning_starts
                    + self.global_step * self.args.train_frequency
                    + tfs
                )
                # update the target network
                if (
                    0
                    == (
                        self.args.learning_starts
                        + self.global_step * self.args.train_frequency
                        + tfs
                    )
                    % self.args.sync_rate
                ):
                    self.target_network.load_state_dict(self.q_network.state_dict())

        return loss

    def train_dataloader(self):
        dataset = RLDataset(self.rb, self.args.batch_size)

        dataloader = DataLoader(dataset=dataset, batch_size=self.args.batch_size)

        return dataloader

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.q_network.parameters(),
            lr=self.args.learning_rate,
            eps=0.01 / self.args.batch_size,
        )
        return optimizer

    def action_linear_schedule(self, t: int) -> int:
        slope = (self.args.end_e - self.args.start_e) / (
            self.args.exploration_fraction * self.args.total_timesteps
        )
        return max(slope * t + self.args.start_e, self.args.end_e)

    def __del__(self):
        self.envs.close()


if __name__ == "__main__":
    import logging

    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            "Ongoing migration: run the following command to install the new dependencies:\n \n"
            'poetry run pip install "stable_baselines3==2.1.0"'
        )

    from src._utils import _logger

    lp_logger = _logger.get_logger(name="lightning.pytorch", level=logging.INFO)

    args = parse_args()

    lp_logger.info(f"PyTorch Version: {torch.__version__}")

    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    if args.track:
        from lightning.pytorch.loggers import WandbLogger

        wandb_logger = WandbLogger(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            save_dir=f"{args.save_prefix}/res/benchmarks/c51",
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    tb_logger = TensorBoardLogger(
        save_dir=f"{args.save_prefix}/res/benchmarks/c51/lightning_logs",
        name=args.env_id,
        default_hp_metric=False,
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    torch.multiprocessing.set_start_method("spawn")

    c51algo = C51ALGO(args, run_name)

    if args.track:
        trainer = pl.Trainer(
            gpus=[args.gpu_id],
            max_epochs=(args.total_timesteps - args.learning_starts)
            // args.train_frequency,
            default_root_dir=f"{args.save_prefix}/res/benchmarks/c51",
            logger=[tb_logger, wandb_logger],
        )

    else:
        trainer = pl.Trainer(
            gpus=[args.gpu_id],
            max_epochs=(args.total_timesteps - args.learning_starts)
            // args.train_frequency,
            default_root_dir=f"{args.save_prefix}/res/benchmarks/c51",
            logger=[tb_logger],
        )

    trainer.fit(c51algo)

    if args.save_model:
        save_dir = f"{args.save_prefix}/res/benchmarks/c51/models/{run_name}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        model_path = f"{args.save_prefix}/res/benchmarks/c51/models/{run_name}/{args.exp_name}.model"
        model_data = {
            "model_weights": c51algo.q_network.state_dict(),
            "args": vars(args),
        }
        torch.save(model_data, model_path)
        lp_logger.info(f"model saved to {model_path}")
        from src._utils._evals.c51_eval import c51_evaluate

        episodic_returns = c51_evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=QNetwork,
            device=c51algo.device,
            epsilon=0.05,
        )
        for idx, episodic_return in enumerate(episodic_returns):
            tb_logger.log_metrics({"eval/episodic_return": episodic_return}, idx)
            if args.track:
                wandb_logger.log_metrics({"eval/episodic_return": episodic_return}, idx)
