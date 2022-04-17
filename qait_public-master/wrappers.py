from types import SimpleNamespace
import gym
import spacy

from game_generator import generate_qa_pairs

class PreprocessorWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, config: SimpleNamespace):
        super().__init__(env)
        self.config = config
        self.nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'tagger'])

    def observation(self, obs_and_infos):
        observations, infos = obs_and_infos
        observations = [self.nlp(obs) for obs in observations]
        return (observations, infos)

class QAPairWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, config: SimpleNamespace):
        super().__init__(env)
        self.config = config
    
    def observation(self, obs_and_infos):
        observations, infos = obs_and_infos
        if 'questions' in infos:
            questions = infos['questions']
        else:
            questions, answers, reward_info = generate_qa_pairs(infos, question_type=self.config.general.question_type)
            infos['questions'] = questions
            infos['answers'] = answers
            infos['reward_info'] = reward_info
        observations = [obs + q for obs, q in zip(observations, questions)]
        return observations, infos
    
class RewardWrapper(gym.RewardWrapper):
    # Probably should just make this a Wrapper and override step, reset
    def __init__(self, env):
        super().__init__(env)
    
    def reward(self, rew):
        # modify rew
        return rew