import spacy
import textworld

class QAAgent(textworld.Agent):
    """ Interface for any agent that want to play a text-based game. """

    def reset(self, env: textworld.Environment) -> None:
        """ Let the agent set some environment's flags.
        Args:
            env: TextWorld environment.
        """
        pass

    def act(self, game_state: textworld.GameState, reward: float, done: bool) -> str:
        """ Acts upon the current game state.
        Args:
            game_state: Current game state.
            reward: Accumulated reward up until now.
            done: Whether the game is finished.
        Returns:
            Text command to be performed in this current state.
        """
        raise NotImplementedError()

    def answer_question(self, input_quest, game_state: textworld.GameState):
        """ Answer the question from the start of the game.
        Args:
            input_quest: the question to answer
            game_state: the game state from the step right before asking the Agent the question.
        Returns:
            Text answer to the question.
        """
        raise NotImplementedError()

    def finish(self, game_state: textworld.GameState, reward: float, done: bool) -> None:
        """ Let the agent know the game has finished.
        Args:
            game_state: Game state at the moment the game finished.
            reward: Accumulated reward up until now.
            done: Whether the game has finished normally or not.
                If False, it means the agent's used up all of its actions.
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

class CNAgent(QAAgent):
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
        nlp = spacy.load("en_core_web_lg")
        words = game_state.feedback
        action = np.argmax(map(lambda x: nlp(game_state).similarity(nlp(x)), game_state.admissible_commands))
        return action

class RandomAgent(QAAgent):
    """ Agent that randomly selects commands from the admissible ones. """

    def __init__(self, seed=1234):
        self.seed = seed
        self.rng = np.random.RandomState(self.seed)

    def reset(self, env):
        # Activate state tracking in order to get the admissible commands.
        env.activate_state_tracking()
        env.compute_intermediate_reward()  # Needed to detect if a game is lost.

    def act(self, game_state, reward, done):
        # print("Admissible actions: " + str(game_state.admissible_commands))
        return self.rng.choice(game_state.admissible_commands)


class WalkthroughDone(NameError):
    pass


class WalkthroughAgent(QAAgent):
    """ Agent that simply follows a list of commands. """

    def __init__(self, commands=None):
        self.commands = commands

    def reset(self, env):
        env.activate_state_tracking()
        env.display_command_during_render = True
        if self.commands is not None:
            self._commands = iter(self.commands)
            return  # Commands already specified.

        if not hasattr(env, "game"):
            msg = "WalkthroughAgent is only supported for generated games."
            raise NameError(msg)

        # Load command from the generated game.
        self._commands = iter(env.game.quests[0].commands)

    def act(self, game_state, reward, done):
        try:
            action = next(self._commands)
        except StopIteration:
            raise WalkthroughDone()

        action = action.strip()  # Remove trailing \n, if any.
        return action
