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
            self.observation_history[i][" ".join(state[0])] = True
            self.init_facts[i] = set(infos['facts'][i])
        self.discovered_facts = copy.deepcopy(self.init_facts)
        return states, infos

    def step(self, commands):
        states, rewards, done, infos = super().step(commands)
        rewards = np.array(rewards, dtype=float)
        
        # Episodic Discovery reward: 1 for a new feedback, 0 for already seen feedback
        rewards += self.reward_episodic_discovery(states, infos)

        

        # Sufficient Info Rewards
        # - Location: 1 if feedback contains entity from question, else 0
        # - Existence: If answer is 1, use location bonus, else use exploration coverage bonus
        #   - Exploration coverage: float in [0,1] describing how many facts the player has discovered
        # - Attribute: Heuristic bonus + 0.1 location bonus + 0.1 exploration coverage bonus
        if self.config.general.question_type == "location":
            rewards += self.reward_location(infos, states)
            rewards += 0.1 * self.reward_exploration_coverage(infos)
        elif self.config.general.question_type == "existence":
            answers = np.array(infos['answers'])
            location_rewards = self.reward_location(infos, states)
            coverage_rewards = self.reward_exploration_coverage(infos)
            rewards += np.where(answers == 1, location_rewards, coverage_rewards)
        elif self.config.general.question_type == "attribute":
            rewards += self.reward_attribute(infos, commands) # Attribute heuristic reward
            rewards += 0.1 * self.reward_location(infos, states)
            rewards += 0.1 * self.reward_exploration_coverage(infos)
        else:
            raise NotImplementedError

        # Update observation histories
        # Update discovered facts
        for i in range(len(states)):
            state = states[i]
            facts = infos['facts'][i]
            self.observation_history[i][" ".join(state[0])] = True
            self.discovered_facts[i] = self.discovered_facts[i].union(set(facts))

        return states, rewards, done, infos
        

    def reward_episodic_discovery(self, states, infos):
        rewards = np.zeros(len(states), dtype=float)
        for i, state in enumerate(states):
            if " ".join(state[0]) not in self.observation_history[i]:
                rewards[i] = 1. 
            else:
                rewards[i] = 0.
        return rewards

    def reward_location(self, infos, states):
        ''' Give a reward if the observation contains the entity
        Args:
            infos (dict): extra information from environment
            states (List[Tuple[tokens, tokens]]): feedback and question from env
        Returns: location bonus
        '''
        rewards = np.zeros(len(states), dtype=float)
        for i in range(len(rewards)):
            entity = infos['reward_info']['_entities'][i]
            rewards[i] = 1. if entity in " ".join(states[i][0]) else 0.
        return rewards

    def reward_exploration_coverage(self, infos):
        rewards = np.zeros(len(self.init_facts), dtype=float)
        for i in range(len(self.init_facts)):
            current_facts = set(infos['facts'][i])
            discovered_facts = self.discovered_facts[i]
            initial_facts = self.init_facts[i]
            
            new_facts = current_facts.difference(discovered_facts)
            cumulative_new_facts = discovered_facts.difference(initial_facts)
            
            if len(cumulative_new_facts) == 0:
                return 0.0
            coverage = len(new_facts) / float(len(cumulative_new_facts))
            assert coverage >= 0, "this shouldn't happen, the agent shouldn't be able to lose coverage info."
            rewards[i] = coverage
        return rewards
    
    def reward_attribute(self, infos, commands):
        rewards = np.zeros(len(commands), dtype=float)
        for i in range(len(rewards)):
            attr = infos['reward_info']['_attributes'][i]
            entity = infos['reward_info']['_entities'][i]
            command = commands[i]
            rewards[i] = self.compute_attribute_heuristic(attr, entity, command, 
                infos['inventory'][i], infos['facts'][i])
        return rewards

    def compute_attribute_heuristic(self, attribute, entity, command, inventory, facts):
        reward = 0.

        for prop in facts:
            if prop.name == "at" and prop.arguments[0].name == "P":
                current_room = prop.arguments[1].name

        entities_in_room = set([prop.arguments[0].name 
                                    for prop in facts 
                                        if prop.name == "at" and prop.arguments[1].name == current_room])
        if attribute == "holder":
            if ("put" in command or "insert" in command) \
                and entity in command and entity in entities_in_room:
                reward += 1.
        elif attribute == "portable":
            if ("take" in command or "drop" in command) \
                and entity in command:
                reward += 1.
        elif attribute == "openable":
            if ("open" in command or "close" in command) \
                and entity in command and entity in entities_in_room:
                reward += 1.
        elif attribute == "drinkable":
            if "drink" in command and entity in command:
                reward += 1.
            if "take" in command \
                and entity in command and entity in entities_in_room:
                reward += .5
        elif attribute == "edible":
            if "eat" in command and entity in command:
                reward += 1.
            if "take" in command \
                and entity in command and entity in entities_in_room:
                reward += .5
        elif attribute == "sharp":
            if ("slice" in command or "chop" in command or "dice" in command) \
                and entity in inventory:
                reward += 1.
            if "take" in command \
                and entity in command and entity in entities_in_room:
                reward += .5
        elif attribute == "heat_source":
            if "cook" in command and entity in entities_in_room:
                reward += 1.
        elif attribute == "cookable":
            if "cook" in command and entity in command and entity in inventory:
                reward += .1
            if "take" in command \
                and entity in command and entity in entities_in_room:
                reward += .5
        elif attribute == "cuttable":
            if ("slice" in command or "chop" in command or "dice" in command) \
                and entity in command and entity in inventory:
                reward += 1.
            if "take" in command \
                and entity in command and entity in entities_in_room:
                reward += .5
        else:
            raise NotImplementedError
        return reward