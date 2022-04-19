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
        self.command_history = {}
        for i, state in enumerate(states):
            self.observation_history[i] = {}
            self.observation_history[i][" ".join(state)] = True
            self.init_facts[i] = set(infos['facts'][i])
            self.commmand_history[i] = []
        self.discovered_facts = copy.deepcopy(self.init_facts)
        return states, infos

    def step(self, commands):
        states, rewards, done, infos = super().step(commands)
        rewards = np.array(rewards)
        rewards += self.reward_episodic_discovery(states, infos)
        if done: # should change this to if command is "wait"
            if self.config.general.question_type == "location":
                rewards += self.reward_location(infos)
            elif self.config.general.question_type == "existence":
                answers = np.array(infos['answers'])
                coverage_rewards = self.reward_exploration_coverage(infos)
                location_rewards = self.reward_location(infos)
                rewards += np.where(answers == 1, location_rewards, coverage_rewards)
            elif self.config.general.question_type == "attribute":
                rewards += self.reward_attribute() # Attribute heuristic reward
                rewards += 0.1 * self.reward_discovered_entity(infos)
                rewards += 0.1 * self.reward_exploration_coverage(infos)
            else:
                raise NotImplementedError
        # Episodic Discovery reward: 1 for a new state, 0 for already seen state
        # Sufficient Info bonus
        # - Location: 1 if final state contains entity in question, else 0
        # - Existence: If answer is yes/True/1, use location bonus, else use exploration coverage bonus
        # - Attribute: Heuristic bonus + 0.1 if entity was observed in any state + 0.1 exploration coverage bonus
        
        # Update observation histories
        # Update discovered facts
        # Update command histories
        for i in range(len(states)):
            state = states[i]
            facts = infos['facts'][i]
            command = commands[i]
            self.observation_history[i][" ".join(state)] = True
            self.discovered_facts[i] = self.discovered_facts[i].union(set(facts))
            self.command_history[i].append(command)
        
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
            entity = infos['reward_info']['_entities'][sample_idx]
            if entity in list(history.keys())[-1]:
                rewards[sample_idx] = 1.
            else:
                rewards[sample_idx] = 0.
        return rewards

    def reward_discovered_entity(self, infos):
        rewards = np.zeros(len(self.observation_history))
        for sample_idx, history in self.observation_history.items():
            entity = infos['reward_info']['_entities'][sample_idx]
            if entity in " ".join(list(history.keys())):
                rewards[sample_idx] = 1.
            else:
                rewards[sample_idx] = 0.
        return rewards

    def reward_exploration_coverage(self, infos):
        rewards = np.zeros(len(self.init_facts))
        for i in range(len(self.init_facts)):
            game = Game.deserialize(infos['game'][i])
            all_facts = game.world.facts
            revealed_facts = self.discovered_facts[i]
            initial_facts = self.init_facts[i]

            all_containers = set([var 
                                    for prop in all_facts
                                        for var in prop.arguments
                                            if var.type == "c"])
            all_rooms = set([var 
                                for prop in all_facts 
                                    for var in prop.arguments 
                                        if var.type == "r"])
            opened_containers = set([var 
                                        for prop in revealed_facts
                                            for var in prop.arguments 
                                                if var in all_containers and prop.name == "open"])
            visited_rooms = set([var 
                                    for prop in revealed_facts 
                                        for var in prop.arguments 
                                            if var in all_rooms and prop.name == "at" and prop.arguments[0].name == "P"])
            init_opened_containers = set([var 
                                            for prop in initial_facts 
                                                for var in prop.arguments 
                                                    if var in all_containers and prop.name == "open"])
            init_visited_rooms = set([var 
                                        for prop in initial_facts 
                                            for var in prop.arguments 
                                                if var in all_rooms and prop.name == "at" and prop.arguments[0].name == "P"])
            discoverable = len(all_containers) + len(all_rooms) - len(init_opened_containers) - len(init_visited_rooms)
            discovered = len(opened_containers) + len(visited_rooms) - len(init_opened_containers) - len(init_visited_rooms)
            if discoverable == 0:
                return 0.0
            coverage = float(discovered) / float(discoverable)
            assert coverage >= 0, "this shouldn't happen, the agent shouldn't be able to lose coverage info."
            rewards[i] = coverage
        return rewards
    
    def reward_attribute(self, infos):
        # The paper implementation seems flawed. Why not just keep command history and check if the agent has performed
        # the correct commands in the past?
        rewards = np.zeros(len(self.discovered_facts))
        for i in range(len(rewards)):
            attr = infos['reward_info']['_attributes'][i]
            commands = self.command_history[i]
            rewards[i] = self.compute_attribute_heuristic(attr, commands)
        return rewards

    def compute_attribute_heuristic(self, attribute, commands):
        reward = 0.
        if attribute == "holder":
            pass
        elif attribute == "portable":
            pass
        elif attribute == "openable":
            pass
        elif attribute == "drinkable":
            pass
        elif attribute == "edible":
            pass
        elif attribute == "sharp":
            pass
        elif attribute == "heat_source":
            pass
        elif attribute == "cookable":
            pass
        elif attribute == "cuttable":
            pass
        else:
            raise NotImplementedError
        return reward