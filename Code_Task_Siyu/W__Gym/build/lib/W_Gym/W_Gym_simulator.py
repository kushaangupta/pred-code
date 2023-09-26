import numpy as np
import pygame
from gym.wrappers import RecordVideo

class W_env_simulator():
    keys_actions = None
    def __init__(self, env, is_record = False, log_dir = './video'):
        if is_record:
            env = self.wrap_env(env, log_dir)
        self.env = env

    def wrap_env(self, env, log_dir):
        env = RecordVideo(env, log_dir)
        return env
    
    def set_keys(self, keys, actions):
        self.keys_actions = {'keys':keys, 'actions': actions}

    def play(self, mode = "human", model = None):
        env = self.env
        obs = env.reset()
        # if model is not None:
        #     model.playmode()
        done = False
        while not done:
            if mode == "human":
                assert self.keys_actions is not None
                action = None
                while action is None:
                    for event in pygame.event.get():
                        if event.type == pygame.KEYDOWN:
                            key = pygame.key.name(event.key)
                            if key == "escape":
                                env.close()
                                return
                            elif key in self.keys_actions['keys']:
                                kid = np.where([key == i for i in self.keys_actions['keys']])[0][0]
                                action = self.keys_actions['actions'][kid]
            elif mode == "random":
                action = env.action_space.sample()
            elif mode == "model":
                action = model.predict(obs)
            obs, R, done, _, info = env.step(action)            
                       
