import random
from types import SimpleNamespace
from typing import List
import yaml
import copy
from collections import namedtuple
from os.path import join as pjoin

import spacy
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from baselines import QAAgent
import command_generation_memory
import qa_memory
from model import DQN
from layers import compute_mask, NegativeLogLoss
from generic import to_np, to_pt, preproc, _words_to_ids, pad_sequences
from generic import max_len, ez_gather_dim_1, ObservationPool
from generic import list_of_token_list_to_char_input, ReplayMemory, Transition


class DQNAgent(nn.Module, QAAgent):
    
    def __init__(self, config: SimpleNamespace):
        super().__init__()
        self.config = config
        self.device = torch.device("cuda" if self.config.general.use_cuda and torch.cuda.is_available() else "cpu")
        
        self.policy_net = DQN(config=self.config,
                              word_vocab=self.word_vocab,
                              char_vocab=self.char_vocab,
                              answer_type=self.answer_type)
        self.target_net = DQN(config=self.config,
                              word_vocab=self.word_vocab,
                              char_vocab=self.char_vocab,
                              answer_type=self.answer_type)

        self.train()
        self.update_target_net()
    
        # optimizer
        self.optimizer = torch.optim.AdamW(self.online_net.parameters(), 
            lr=self.config.training.optimizer.learning_rate)


    def train(self):
        self.policy_net.train()
        self.target_net.train()

    def eval(self):
        self.policy_net.eval()
        self.target_net.eval()

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def load_pretrained(self, model_path):
        """
        Load pretrained checkpoint from file.

        Arguments:
            model_path: File name of the pretrained model checkpoint.
        """
        print("loading model from %s\n" % (model_path))
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            self.online_net.load_state_dict(state_dict)
            self.update_target_net()
        except:
            print("Failed to load checkpoint...")

    def save_model_to_path(self, save_to):
        torch.save(self.online_net.state_dict(), save_to)
        print("Saved checkpoint to %s..." % (save_to))


    def act(self, game_states, cumulative_rewards, done, infos) -> List[str]:
        """ Acts upon the current game state.
        Args:
            game_states: List of length batch_size, each entry is a tuple of the
                tokenized environment feedback and tokenized question.
            rewards: List of length batch_size, accumulated rewards up until now.
            done: List of length batch_size, whether the game is finished.
            infos: Dictionary of game information, each value is a list of 
                length batch_size for each game in the batch.
        Returns:
            List of text commands to be performed in this current state for each
            game in the batch.
        """

        pass

   

    def optimize_model(self, replay_memory: ReplayMemory):
        if len(replay_memory) < self.config.replay.replay_batch_size:
            return
        transitions = replay_memory.sample(self.config.replay.replay_batch_size)
        batch = Transition(*zip(*transitions))

        non_final_next_states = torch.cat(batch.next_state)
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.config.replay.replay_batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[:] = self.target_net(non_final_next_states).max(1)[0]
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.config.replay.discount_gamma) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(),
            self.config.training.optimizer.clip_grad_norm)  
        self.optimizer.step()

   
