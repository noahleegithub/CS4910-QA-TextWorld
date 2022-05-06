import random
from types import SimpleNamespace
from typing import List
import yaml
import copy
import json
from collections import namedtuple
from os.path import join as pjoin
import h5py

import spacy
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from baselines import QAAgent
import command_generation_memory
import qa_memory
from model import DQN, Embedder
from generic import to_np, to_pt, preproc, _words_to_ids, pad_sequences
from generic import max_len, ez_gather_dim_1, ObservationPool
from generic import list_of_token_list_to_char_input, ReplayMemory, Transition


class DQNAgent(nn.Module, QAAgent):
    
    def __init__(self, config: SimpleNamespace):
        super().__init__()
        self.config = config
        self.device = torch.device("cuda" if self.config.general.use_cuda and torch.cuda.is_available() else "cpu")
        
        with open("vocabularies/word_vocab.txt") as f:
            vocab = f.read().split()
        self.word_embeddings = Embedder(vocab, config.model.word_embedding_size, "crawl-300d-2M.vec.h5")
        
        self.policy_net = LSTMDQN(config, self.word_embeddings)
        self.target_net = LSTMDQN(config, self.word_embeddings)

        self.train()
        self.update_target_net()
    
        # optimizer
        self.optimizer = torch.optim.AdamW(self.policy_net.parameters(), 
            lr=self.config.training.optimizer.learning_rate)

    def reset(self, environment):
        pass

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
        torch.save(self.target_net.state_dict(), save_to)
        print("Saved checkpoint to %s..." % (save_to))
    
    def vectorized_get_words(self):
        get_word = lambda idx: self.word_embeddings.vocab[idx]
        return np.vectorize(get_word)

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
        with torch.no_grad():
            outputs = self.policy_net(self.embed_states(game_states))
        return self.vectorized_get_words()(np.argmax(outputs.numpy(), axis=2)).transpose()
        
    def embed_states(self, data):
        data = tuple(zip(*data))
        observations = []
        questions = []
        for obs in data[0]:
            observations.append(torch.stack([self.word_embeddings(word) for word in obs]))
        for ques in data[1]:
            questions.append(torch.stack([self.word_embeddings(word) for word in ques]))

        return observations, questions
   

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
        state_action_values = self.policy_net(state_batch, action_batch)

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

class LSTMDQN(nn.Module):

    def __init__(self, config: SimpleNamespace, embedder) -> None:
        super().__init__()
        self.config = config
        embed_dim = config.model.word_embedding_size
        self.observation_conv = nn.Conv1d(in_channels=embed_dim, out_channels=embed_dim,
            kernel_size=config.model.conv_kernel, stride=config.model.conv_stride)
        self.observation_encoder = nn.LSTM(embed_dim, config.model.hidden_dim)
        self.question_encoder = nn.LSTM(embed_dim, config.model.hidden_dim)
        self.answer_decoder = nn.LSTMCell(embed_dim, config.model.hidden_dim*2)
        self.word_proj = nn.Linear(in_features=config.model.hidden_dim*2, out_features=len(embedder.vocab))
        self.action_length = config.model.action_length
        self.embedder = embedder


    def forward(self, x):
        kernel = self.config.model.conv_kernel
        stride = self.config.model.conv_stride
        observations, questions = x
        obs_lengths, q_lengths = [len(o) for o in observations], [len(q) for q in questions]
        observations = nn.utils.rnn.pad_sequence(observations, batch_first=True)
        questions = nn.utils.rnn.pad_sequence(questions, batch_first=False)

        N, L1, C = observations.shape
        _, L2, _ = questions.shape
        observations = self.observation_conv(observations.permute((0,2,1)))
        observations = observations.permute((2,0,1))
        obs_lengths = [int((l - kernel) / stride) + 1 for l in obs_lengths]

        observations = nn.utils.rnn.pack_padded_sequence(observations, torch.tensor(obs_lengths), enforce_sorted=False)
        questions = nn.utils.rnn.pack_padded_sequence(questions, torch.tensor(q_lengths), enforce_sorted=False)

        _, (obs_hidden, _) = self.observation_encoder(observations)
        _, (q_hidden, _) = self.question_encoder(questions)
        decoder_hidden_state = torch.cat((obs_hidden, q_hidden), dim=2).squeeze()

        decoder_h, decoder_c = decoder_hidden_state, torch.zeros_like(decoder_hidden_state)
        decoded_word = self.embedder([3] * N)
        output = []

        for idx in range(self.action_length):
            decoder_h, decoder_c = self.answer_decoder(decoded_word, (decoder_h, decoder_c))
            output.append(self.word_proj(decoder_h))
            decoded_word = self.embedder(torch.argmax(output[-1], dim=1))
            
        return torch.stack(output, dim=0)

if __name__ == "__main__":
    with open("config.yaml") as config_reader:
        config = yaml.safe_load(config_reader)
        config = json.loads(json.dumps(config), object_hook=lambda d : SimpleNamespace(**d))
    
    agent = DQNAgent(config)
    out = agent.act([(['if', 'you', "'re", 'wondering', 'why', 'everything', 'seems', 'so', 'typical', 'all', 'of', 'a', 'sudden', ',', 'it', "'s", 'because', 'you', "'ve", 'just', 'sauntered', 'into', 'the', 'shed', '.', 'if', 'you', 'have', "n't", 'noticed', 'it', 'already', ',', 'there', 'seems', 'to', 'be', 'something', 'there', 'by', 'the', 'wall', ',', 'it', "'s", 'a', 'stainless', 'oven', '.', 'the', 'oven', 'is', 'empty', '!', 'this', 'is', 'the', 'worst', 'thing', 'that', 'could', 'possibly', 'happen', ',', 'ever', '!', 'you', 'rest', 'your', 'hand', 'against', 'a', 'wall', ',', 'but', 'you', 'miss', 'the', 'wall', 'and', 'fall', 'onto', 'a', 'comfy', 'sofa', '.', 'on', 'the', 'comfy', 'sofa', 'you', 'can', 'make', 'out', 'an', 'interesting', 'cookbook', '.', 'you', 'shudder', ',', 'but', 'continue', 'examining', 'the', 'room', '.', 'there', 'is', 'a', 'closed', 'iron', 'gate', 'leading', 'west', '.', 'there', 'is', 'an', 'open', 'wooden', 'door', 'leading', 'south', '.'], ['is', 'there', 'any', 'gas', 'grill', 'in', 'the', 'world', '?']), (['if', 'you', "'re", 'wondering', 'why', 'everything', 'seems', 'so', 'typical', 'all', 'of', 'a', 'sudden', ',', 'it', "'s", 'because', 'you', "'ve", 'just', 'sauntered', 'into', 'the', 'shed', '.', 'if', 'you', 'have', "n't", 'noticed', 'it', 'already', ',', 'there', 'seems', 'to', 'be', 'something', 'there', 'by', 'the', 'wall', ',', 'it', "'s", 'a', 'stainless', 'oven', '.', 'hmmm', '...', 'what', 'else', ',', 'what', 'else', '?', 'the', 'oven', 'is', 'empty', '!', 'this', 'is', 'the', 'worst', 'thing', 'that', 'could', 'possibly', 'happen', ',', 'ever', '!', 'you', 'rest', 'your', 'hand', 'against', 'a', 'wall', ',', 'but', 'you', 'miss', 'the', 'wall', 'and', 'fall', 'onto', 'a', 'comfy', 'sofa', '.', 'on', 'the', 'comfy', 'sofa', 'you', 'can', 'make', 'out', 'an', 'interesting', 'cookbook', '.', 'you', 'shudder', ',', 'but', 'continue', 'examining', 'the', 'room', '.', 'there', 'is', 'a', 'closed', 'iron', 'gate', 'leading', 'west', '.', 'there', 'is', 'an', 'open', 'wooden', 'door', 'leading', 'south', '.'], ['is', 'there', 'any', 'king', 'bed', 'in', 'the', 'world', '?'])], None, None, None)
    print(out.shape, out)
    #print(obs.shape, q.shape)

