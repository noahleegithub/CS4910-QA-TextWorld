from typing import Tuple, Any, Mapping, List
from types import SimpleNamespace
import gym
from textworld import GameState, Game
from gym.core import Wrapper
import spacy
import re
import numpy as np

from game_generator import generate_qa_pairs

class PreprocessorWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.reGarbageChars = re.compile('[$_|\\\\/>]')
        self.reLocationTag = re.compile('-=.*=-')

    def observation(self, obs):
        if type(obs) is tuple:
            observations, infos = obs
        else:
            observations = obs
        for i in range(len(observations)):
            observations[i] = re.sub(self.reGarbageChars, "", observations[i])
            observations[i] = re.sub(self.reLocationTag, "", observations[i])            
            observations[i] = " ".join(observations[i].split())
        return (observations, infos) if type(obs) is tuple else observations

class QAPairWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, config: SimpleNamespace):
        super().__init__(env)
        self.config = config
        self.questions = None
        self.answers = None
        self.reward_info = None
    
    def reset(self):
        observations, infos = super().reset()
        self.questions, self.answers, self.reward_info = generate_qa_pairs(infos, question_type=self.config.general.question_type)
        return self.append_questions(observations, infos)

    def step(self, command):
        observations, reward, done, infos = super().step(command)
        new_observations, new_infos = self.append_questions(observations, infos)
        return new_observations, reward, done, new_infos
        
    def append_questions(self, observations, infos):
        infos['questions'] = self.questions
        infos['answers'] = self.answers
        infos['reward_info'] = self.reward_info
        new_observations = list(zip(observations, self.questions))
        return new_observations, infos

class TokenizerWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'tagger'])

    def observation(self, obs):
        if type(obs) is tuple:
            observations, infos = obs
        else:
            observations = obs
        for i in range(len(observations)):
            tokenized_feedback = [token.text for token in self.nlp(observations[i][0])]
            tokenized_question = [token.text for token in self.nlp(observations[i][1])]
            observations[i] = (tokenized_feedback, tokenized_question)
        return (observations, infos) if type(obs) is tuple else observations

    
class RewardWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, config: SimpleNamespace):
        super().__init__(env)
        self.config = config
    
    def reset(self):
        states, infos = super().reset()
        self.observation_history = {}
        self.init_facts = {}
        for i, state in enumerate(states):
            self.observation_history[i] = {}
            self.observation_history[i][" ".join(state)] = True
            self.init_facts[i] = set(infos['facts'][i])
        return states, infos

    def step(self, commands):
        states, rewards, done, infos = super().step(commands)
        rewards = np.array(rewards)
        rewards += self.reward_episodic_discovery(states, infos)
        if self.config.general.question_type == "location":
            rewards += self.reward_location(infos)
        elif self.config.general.question_type == "existence":
            rewards
        elif self.config.general.question_type == "attribute":
            rewards
        else:
            raise NotImplementedError
        # episodic discovery reward: 1 for a new state
        # Sufficient info bonus
        # - Location: 1 if final state contains object, else 0
        # - Existence: If answer is yes, use location bonus, else use exploration coverage bonus
        # - Attribute: Heuristic bonus + 0.1 if entity was observed in any state + 0.1 exploration coverage bonus
        for i, state in enumerate(states):
            self.observation_history[i][" ".join(state)] = True

        return states, rewards, done, infos

    def reward_episodic_discovery(self, states, infos):
        rewards = np.zeros(len(states))
        for i, state in enumerate(states):
            if self.observation_history[i][" ".join(state)]:
                rewards[i] = 1.
            else:
                rewards[i] = 0.
        return rewards

    def reward_location(self, infos):
        rewards = np.zeros(len(self.observation_history))
        for sample_idx, history in self.observation_history.items():
            entity = infos['reward_info']["_entities"][sample_idx]
            if entity in list(history.keys())[-1]:
                rewards[sample_idx] = 1.
            else:
                rewards[sample_idx] = 0.
        return rewards

    def reward_exploration_coverage(self, infos):
        for i in range(len(self.init_facts)):
            game = Game.deserialize(infos['game'][i])
            all_facts = set(game.world.facts)
            current_facts = set(infos['facts'][i]) # rework this; should be union of all discovered facts
            initial_facts = self.init_facts[i]
        pass
    
