from typing import Tuple, Any, Mapping, List
from types import SimpleNamespace
import gym
from textworld import GameState
from gym.core import Wrapper
import spacy
import re

from game_generator import generate_qa_pairs

class PreprocessorWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, config: SimpleNamespace):
        super().__init__(env)
        self.config = config
        self.nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'tagger'])
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
    
    def reset(self):
        observations, infos = super().reset()
        return self.append_questions(observations, infos)

    def step(self, command):
        observations, reward, done, infos = super().step(command)
        new_observations, new_infos = self.append_questions(observations, infos)
        return new_observations, reward, done, new_infos
        
    def append_questions(self, observations, infos):
        if 'questions' in infos:
            questions = infos['questions']
        else:
            questions, answers, reward_info = generate_qa_pairs(infos, question_type=self.config.general.question_type)
            infos['questions'] = questions
            infos['answers'] = answers
            infos['reward_info'] = reward_info
        new_observations = [obs + q for obs, q in zip(observations, questions)]
        return new_observations, infos
    
class RewardWrapper(gym.RewardWrapper):
    # Probably should just make this a Wrapper and override step, reset
    def __init__(self, env):
        super().__init__(env)
    
    def reward(self, rew):
        # modify rew
        return rew