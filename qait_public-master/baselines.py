import textworld
from typing import List, Tuple
import numpy as np

QAGameState = List[Tuple[List[str], List[str]]]

class QAAgent(textworld.Agent):
    """ Interface for any agent that want to play a text-based game. """

    def reset(self, env: textworld.Environment) -> None:
        """ Let the agent set some environment's flags.
        Args:
            env: TextWorld environment.
        """
        pass

    def act(self, game_states: QAGameState, reward: List[float], done: List[bool], infos: dict) -> str:
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
        raise NotImplementedError

    def finish(self, game_states: QAGameState, reward: List[float], done: List[bool], infos: dict) -> None:
        """ Let the agent know the game has finished.
        Args:
            game_states: List of length batch_size, each entry is a tuple of the
                        tokenized environment feedback and tokenized question.
            rewards: List of length batch_size, accumulated rewards up until now.
            done: List of length batch_size, whether the game is finished.
            infos: Dictionary of game information, each value is a list of 
                    length batch_size for each game in the batch.
        """
        pass

class NaiveAgent(QAAgent):
    """ Some Agent that has a hardcoded algorithm to act. Maybe we could test using the knowledge graph here. """
    def __init__(self, seed=1234):
        self.seed = seed
        self.rng = np.random.RandomState(self.seed)
        self.actions = ["north", "south", "east", "west", "up", "down",
                        "look", "inventory", "take all", "YES", "wait",
                        "take", "drop", "eat", "attack"]

    def reset(self, env):
        env.display_command_during_render = True
        env.activate_state_tracking()

    def act(self, game_state, reward, done):

        action = self.rng.choice(self.actions)
        if action in ["take", "drop", "eat", "attack"]:
            words = game_state.feedback.split()  # Observed words.
            words = [w for w in words if len(w) > 3]  # Ignore most stop words.
            if len(words) > 0:
                action += " " + self.rng.choice(words)

        return action


class RandomAgent(QAAgent):
    """ Agent that randomly selects commands from the admissible ones. """

    def __init__(self, seed=1234):
        self.rng = np.random.default_rng(seed)

    def act(self, game_states: QAGameState, reward: List[float], done: List[bool], infos: dict) -> str:
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
        commands = []
        for i in range(len(game_states)):
            commands.append(self.rng.choice(infos['admissible_commands'][i]))
        return commands
