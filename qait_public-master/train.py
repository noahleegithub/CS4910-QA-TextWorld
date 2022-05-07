import argparse
from gym.core import RewardWrapper
import yaml
import datetime
import os
import copy
import time
import json
from types import SimpleNamespace
import visdom
import torch
import numpy as np
import tempfile
from os.path import join as pjoin
from distutils.dir_util import copy_tree

import gym
import textworld
from textworld.gym import register_game, make_batch2
from wrappers import PreprocessorWrapper, QAPairWrapper, TokenizerWrapper, HandleAnswerWrapper
from reward_wrapper import RewardWrapper
from agent import DQNAgent
from generic import GameBuffer, Transition, ReplayMemory, append_dict_to_csv
from game_generator import game_generator, game_generator_queue
import evaluate
from query import process_facts
from baselines import RandomAgent, HumanAgent, NaiveCNAgent

from tqdm import tqdm
import time

request_infos = textworld.EnvInfos(description=True,
                                   inventory=True,
                                   verbs=True,
                                   location_names=True,
                                   location_nouns=True,
                                   location_adjs=True,
                                   object_names=True,
                                   object_nouns=True,
                                   object_adjs=True,
                                   facts=True,
                                   last_action=True,
                                   game=True,
                                   admissible_commands=True,
                                   extras=["object_locations", "object_attributes", "uuid"])


def create_games(config: SimpleNamespace, data_path: str):
    # Create temporary folder for the generated games.
    global GAMES_DIR
    GAMES_DIR = tempfile.TemporaryDirectory(prefix="tw_games_") # This is not deleted upon error. It would be better to use a with statement.
    games_dir = os.path.join(GAMES_DIR.name, "") # So path ends with '/'.

    textworld_data_path = os.path.join(data_path, "textworld_data")
    tmp_textworld_data_path = os.path.join(games_dir, "textworld_data")

    assert os.path.exists(textworld_data_path), "Oh no! textworld_data folder is not there..."
    os.mkdir(tmp_textworld_data_path)
    copy_tree(textworld_data_path, tmp_textworld_data_path)
    if config.evaluate.run_eval:
        testset_path = os.path.join(data_path, config.general.testset_path)
        tmp_testset_path = os.path.join(games_dir, config.general.testset_path)
        assert os.path.exists(testset_path), "Oh no! test_set folder is not there..."
        os.mkdir(tmp_testset_path)
        copy_tree(testset_path, tmp_testset_path)
    
    training_game_queue = game_generator_queue(path=games_dir, 
        random_map=config.general.random_map, question_type=config.general.question_type,
        max_q_size=config.training.batch_size * 2, nb_worker=8)
    
    fixed_buffer = True if config.general.train_data_size != -1 else False
    buffer_capacity =  config.general.train_data_size if fixed_buffer else config.training.batch_size * 2
    training_game_buffer = GameBuffer(buffer_capacity, fixed_buffer, training_game_queue)

    return training_game_buffer

def train_2(config: SimpleNamespace, data_path: str, games: GameBuffer):
    episode_no = 0
    agent = DQNAgent(config)
    memory = ReplayMemory(capacity=config.replay.replay_memory_capacity)
    log_csv = 'logs/{}.csv'.format(config.checkpoint.experiment_tag)
    with open(log_csv, 'w+') as f:
        pass

    print("Started training...")
    while episode_no < config.training.max_episode:
        rand = np.random.default_rng(episode_no)
        games.poll()
        if len(games) == 0:
            time.sleep(0.1)
            continue
        sampled_games = np.random.choice(games, config.training.batch_size).tolist()
        env_ids = [register_game(gamefile, request_infos=request_infos) for gamefile in sampled_games]
        env_id = make_batch2(env_ids, parallel=True)
        env = gym.make(env_id)
        env.seed(episode_no)

        env = PreprocessorWrapper(env)
        env = QAPairWrapper(env, config)
        env = TokenizerWrapper(env)
        env = RewardWrapper(env, config)
        env = HandleAnswerWrapper(env)

        
        agent.reset(env) # reset for the next game
        states, infos = env.reset() # state is List[(tokenized observation, tokenized question)] of length batch_size

        cumulative_rewards = np.zeros(len(states), dtype=float)
        done = np.array([False] * len(states))
        losses = []
        for step_no in range(config.training.max_nb_steps_per_episode):
            # actions = agent.act(states, cumulative_rewards, done, infos) # list of strings (batch_size)
            actions = agent.act(states, cumulative_rewards, done, infos) # list of strings (batch_size)
            next_states, rewards, done, infos = env.step(actions) # modify to output rewards
            cumulative_rewards += rewards

            # Store the transition in memory
            for s_0, a, r, s_1 in zip(states, actions, rewards, next_states):
                # dont push states from finished games
                if len(s_0[0]) + len(s_0[1]) != 0:
                    memory.push(Transition(s_0, a, r, s_1))

            states = next_states

            # Perform one step of the optimization (on the policy network)
            if step_no % config.replay.update_per_k_game_steps == 0:
                if callable(getattr(agent, "optimize_model", None)):
                    loss = agent.optimize_model(memory)
                    if loss is not None:
                        losses.append(loss)

            if np.all(done):
                # record some evaluation metrics?
                batch_results = {
                    'loss': np.mean(losses),
                    'reward': np.mean(cumulative_rewards),
                    'qa_acc': np.mean(infos['results']['qa_correct']),
                    'suff_info_acc': np.mean(infos['results']['suff_info_correct'])
                }
                print(batch_results)
                append_dict_to_csv(batch_results, log_csv)
                break

        # Update the target network, copying all weights and biases in DQN
        if episode_no % config.training.target_net_update_frequency == 0:
            if callable(getattr(agent, "update_target_net", None)):
                agent.update_target_net()

        episode_no += config.training.batch_size
        env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train an agent.")
    parser.add_argument("config_path", default="./config.yaml")
    parser.add_argument("data_path",
                        default="./",
                        help="where the data (games) are.")
    args = parser.parse_args()
    with open(args.config_path) as config_reader:
        config = yaml.safe_load(config_reader)
        config = json.loads(json.dumps(config), object_hook=lambda d : SimpleNamespace(**d))
    
    try:
        game_gen = create_games(config, args.data_path)
        train_2(config, args.data_path, game_gen)
    finally:
        if GAMES_DIR:
            GAMES_DIR.cleanup()
    
