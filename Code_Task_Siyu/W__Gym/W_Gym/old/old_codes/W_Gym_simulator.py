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
        if mode in ["oracle","oracle_human"]:
            self.env.is_oracle = True
        if mode == "oracle_human":
            self.set_keys(['space'], [0])
        env = self.env
        obs = env.reset()
        # if model is not None:
        #     model.playmode()
        done = False
        while not done:
            if mode in ["human", "oracle_human"]:
                assert self.keys_actions is not None
                action = None
                while action is None:
                    for event in pygame.event.get():
                        if event.type == pygame.KEYDOWN:
                            key = pygame.key.name(event.key)
                            if key == "escape":
                                env.close()
                                return
                            elif key == "space" and mode == "oracle_human":
                                action = env.get_oracle_action() 
                            elif key in self.keys_actions['keys']:
                                kid = np.where([key == i for i in self.keys_actions['keys']])[0][0]
                                action = self.keys_actions['actions'][kid]
            elif mode == "random":
                action = env.action_space.sample()
            elif mode == "model":
                action = model.predict(obs)
            elif mode == "oracle":
                action = None
            obs, R, done, _, info = env.step(action)            
                       
