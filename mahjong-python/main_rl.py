# -*- coding: utf-8 -*-

import argparse
import os

import numpy as np
from itertools import permutations
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataclasses import dataclass
from torch.nn.utils import clip_grad_norm_
from torch.utils.data.dataloader import default_collate

import framework.session_manager as manager
from framework.logging import log_info
from framework.utils import AverageMeter, Options, iter_batch_spans
from models.baseline_cnn.dataloader import IGNORE_INDEX, Loader
from models.baseline_cnn.model import CNNModel, ModelOptions
from RL.agent import Agent
from RL.arena import Arena


@dataclass
class MainOptions(manager.ManagerOptions, ModelOptions, Options):
    learning_rate: float
    weight_decay: float

    discount: float
    value_coef: float
    entropy_coef: float
    batch_size: int

    clip_grads_norm: float

    only_win_data: bool
    use_extra_reward: bool

    update_n_agent_eps: int
    update_p_agent_freq: int

    use_fixed_agent: bool


class Model(CNNModel):
    def __init__(self, options: MainOptions):
        super().__init__(options)

        self.options = options
        hidden_size = options.hidden_size
        self.value = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, counts, winds, pool, packs, win_tile, return_value=False, **masks):
        features = self.compute_features(counts, winds, pool, packs, win_tile)
        logits = self.compute_logits(features, masks)
        if return_value:
            return logits, self.value(features).squeeze(-1)
        return logits

    def clone(self):
        model = Model(self.options)
        model.load_state_dict(self.state_dict())
        return model.to(next(self.value.parameters()).device)


def _load_model(model, saved_state):
    state_dict = model.state_dict()
    saved_state_dict = saved_state['model']

    flag = set(saved_state_dict) != set(state_dict)
    if flag:
        state_dict.update(saved_state_dict)
    else:
        state_dict = saved_state_dict

    model.load_state_dict(state_dict)
    return flag


class SessionManager(manager.SessionManager):
    def __init__(self, options: MainOptions, training=True):
        super().__init__(options, training)

        self.metrics_name = None

        self.discount = options.discount
        self.value_coef = options.value_coef
        self.entropy_coef = options.entropy_coef
        self.clip_grads_norm = options.clip_grads_norm

        self.only_win_data = options.only_win_data
        self.use_extra_reward = options.use_extra_reward
        self.use_fixed_agent = options.use_fixed_agent

        self.update_n_agent_eps = options.update_n_agent_eps
        self.update_p_agent_freq = options.update_p_agent_freq

        self.batch_size = options.batch_size
        self.ckpt_count = 100

    def build_model(self):
        return Model(self.options)

    def build_optimizer(self):
        options = self.options
        return optim.RMSprop(filter(lambda x: x.requires_grad, self.model.parameters()),
                             lr=options.learning_rate, weight_decay=options.weight_decay)

    def state_dict(self, saved_state):
        saved_state['optimizer'] = self.optimizer.state_dict()
        saved_state['model'] = self.model.state_dict()
        return saved_state

    def load_state_dict(self, saved_state):
        if _load_model(self.model, saved_state):
            self.global_step = 0

            log_info('Load model from pretrained file...')
        else:
            self.optimizer.load_state_dict(saved_state['optimizer'])

    def compute_policy(self, logits, actions):
        batch_size = logits[0].size(0)
        total_logli = torch.zeros(batch_size, device=self.device)
        total_entropy = torch.zeros(batch_size, device=self.device)
        zeros = torch.zeros_like(total_entropy)

        for logit, action in zip(logits, actions):
            probs = F.softmax(logit, dim=-1)
            logit = F.log_softmax(logit, dim=-1)

            total_entropy = torch.where(action != IGNORE_INDEX, -(probs * logit).sum(-1), zeros)
            total_logli += F.nll_loss(logit, action, ignore_index=IGNORE_INDEX, reduction='none')

        return total_logli, total_entropy

    def compute_loss(self, logits, actions, values, advantages, returns):
        logli, entropy = self.compute_policy(logits, actions)

        policy_loss = (logli * advantages).mean()
        value_loss = ((values - returns)**2).mean() * self.value_coef
        entropy_loss = entropy.mean() * self.entropy_coef

        loss = policy_loss + value_loss - entropy_loss

        return loss, (policy_loss, value_loss, entropy_loss)

    def preprocess(self, batch):
        device = self.device
        observation, _, returns = default_collate(batch)
        observation[0] = {key: value.squeeze(dim=1).to(device)
                          for key, value in observation[0].items()}
        for i in range(1, len(observation)):
            observation[i] = observation[i].to(device)

        returns = returns.to(device).float()
        return observation, returns

    def _train_batch(self, batch, meter):
        (obs, *actions), returns = self.preprocess(batch)
        if returns.size(0) <= 1:
            return

        self.model.train()
        self.optimizer.zero_grad()

        logits, values = self.model(**obs, return_value=True)

        advantages = returns - values.detach()
        loss, loss_terms = self.compute_loss(logits, actions, values, advantages, returns)

        loss.backward()
        grads_norm = clip_grad_norm_(self.model.parameters(), self.clip_grads_norm)
        self.optimizer.step()

        meter.add(f'loss', loss.item())
        meter.add(f'grads_norm', grads_norm)
        for name, loss_term in zip(['policy_loss', 'value_loss', 'entropy_loss'], loss_terms):
            meter.add(name, loss_term.item())

    def collect_examples(self, agents, examples):
        n_agent = agents[0]
        examples.extend(e.to_tuple() for es in n_agent.examples for e in es)

    def execute(self, arena, agents, positions, examples, stats):
        scores = [0] * len(agents)
        for agent_indices in set(permutations(positions)):
            fan_value, _ = arena.execute([agents[index] for index in agent_indices])

            for i, agent in enumerate(agents):
                scores[i] += sum(agent.final_rewards)

            stats[0] += fan_value
            stats[1] += 1

            if fan_value == 0:
                stats[2] += 1
                if self.only_win_data:
                    continue

            stats[3] += 1
            self.collect_examples(agents, examples)

        return scores

    def create_agents(self):
        obs_of_player = Loader().obs_of_player
        device = self.device
        discount = self.discount

        n_agent = Agent(self.model, device, discount, obs_of_player, name='NAgent')
        p_agent = Agent(self.model.clone(), device, discount, obs_of_player, name='PAgent')

        if self.use_fixed_agent:
            pretrained_path = os.path.join(self.checkpoint_dir, 'pretrained.pt')
            if os.path.exists(pretrained_path):
                log_info('Init fixed_agent with %s', pretrained_path)
                model = Model(self.model.options)
                _load_model(model, torch.load(pretrained_path))

                fixed_agent = Agent(model.to(device),
                                    device, discount, obs_of_player, name='FixedAgent')
            else:
                log_info('Init fixed_agent with p_agent')
                fixed_agent = Agent(self.model.clone(),
                                    device, discount, obs_of_player, name='FixedAgent')
        else:
            log_info('Do not use fixed_agent')
            fixed_agent = p_agent

        return (n_agent, p_agent, fixed_agent), [0, 0, 1, 2]

    def update_agents(self, agents):
        log_info('update p_agent ...')
        n_agent, p_agent = agents[:2]
        p_agent.model = n_agent.model.clone()

    def learn(self, num_steps=1000000):
        agents, positions = self.create_agents()
        agent_indices = sorted(set(positions))
        agent_names = '/'.join(agents[index].name for index in agent_indices)

        arena = Arena(False, self.use_extra_reward)

        examples = []
        stats = [0, 0, 0, 0]  # fan_value, finished_eps, huang_zhuang, colletced_eps
        scores = np.zeros((len(agents),))

        while self.global_step < num_steps:
            scores += self.execute(arena, agents, positions, examples, stats)

            num_collected_eps = stats[-1]
            if num_collected_eps < self.update_n_agent_eps:
                continue

            meter = AverageMeter()
            for start, end in iter_batch_spans(len(examples), self.batch_size):
                self._train_batch(examples[start:end], meter)

            log_info('stats: fan = %.2f (%d/%d) | LIU: %d | %s : %s',
                     stats[0] / stats[1], *stats[:3],
                     agent_names,
                     '/'.join(str(scores[index]) for index in agent_indices))

            self.on_update(meter, prefix='train')
            self.add_summary('fan', stats[0] / stats[1], prefix='train')
            for i in range(len(scores)):
                self.add_summary(f'score_{i}', scores[i], prefix='train')

            stats = [0, 0, 0, 0]
            examples.clear()

            if self.global_step % self.update_p_agent_freq == 0:
                if scores[0] < 0:
                    log_info('No improvement !!! p_agent keeps unchanged')
                else:
                    self.update_agents(agents)
                    scores[:] = 0


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path_or_restore_dir', type=str)

    cmd_args, _ = parser.parse_known_args(argv)

    options_fn = MainOptions.from_config
    path = cmd_args.config_path_or_restore_dir

    sess_mgr = SessionManager.restore_or_init(path, options_fn, restore=os.path.isdir(path))
    sess_mgr.learn()


if __name__ == '__main__':
    main()
