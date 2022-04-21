from typing import Tuple, Any, Mapping, List
from types import SimpleNamespace
import gym
from networkx.readwrite.graph6 import read_graph6
from textworld import GameState, Game
from gym.core import Wrapper
import spacy
import re
import numpy as np
import copy

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
            observations[i] = observations[i].lower()
        return (observations, infos) if type(obs) is tuple else observations

class QAPairWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, config: SimpleNamespace):
        super().__init__(env)
        self.config = config
    
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
            tokenized_feedback.insert(0, "[START]")
            tokenized_question.insert(0, "[START]")
            tokenized_feedback.append("[END]")
            tokenized_question.append("[END]")
            observations[i] = (tokenized_feedback, tokenized_question)
        return (observations, infos) if type(obs) is tuple else observations

class HandleAnswerWrapper(gym.Wrapper):
    ''' Handle when agents choose "wait" command '''
    def __init__(self, env: gym.Env):
        super().__init__(env)
    
    def reset(self):
        observations, infos = super().reset()
        self.ready_to_answer = np.array([False] * len(observations))
        self.finished = np.array([False] * len(observations))
        return observations, infos

    def step(self, commands):
        observations, rewards, done, infos = super().step(commands)
        for i in range(len(observations)):
            if self.finished[i]:
                observations[i] = ([], [])
                rewards[i] = 0.
                infos['admissible_commands'][i] = ["wait"]
                continue
            elif self.ready_to_answer[i]:
                observations[i] = ([], [])
                rewards[i] = self.config.rewards.correct_answer if commands[i] == infos['answers'][i] else 0
                self.finished[i] = True
                infos['admissible_commands'][i] = ["wait"]
                continue
            elif commands[i] == "wait":
                self.ready_to_answer[i] = True
                observations[i] = ([], observations[i][1])
                infos['admissible_commands'][i] = infos['reward_info']['_all_answers'][i]
            else:
                pass
        
        return observations, rewards, self.finished, infos