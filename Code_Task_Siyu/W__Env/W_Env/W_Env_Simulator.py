from W_Python.W import W
import numpy as np
import pygame

class W_Env_Simulator():
    env = None
    envmode = None

    def __init__(self, env, envmode = "human", *arg, **kwarg):
        self.env = env
        self.envmode = envmode
        if self.envmode == "oracle_human":
            self.env.set_human_keys(['space'], [0])

    def _get_human_keypress(self, mode = "human"):
            assert self.env.human_key_action is not None
            action = None
            while action is None:
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN:
                        key = pygame.key.name(event.key)
                        if key == "escape":
                            self.env.render_close()
                            return None
                        elif key == "space" and mode == "oracle_human":
                            action = self.env.get_oracle_action() 
                        elif key in self.env.human_key_action['keys']:
                            kid = W.W_list_findidx(key, self.env.human_key_action['keys'])
                            action = self.env.human_key_action['actions'][kid]
            return action

    def _get_action(self, mode, model, obs):
        if mode == "model":
            action = model.predict(obs)
        elif mode == "random":
            action = self.env.action_space.sample()
        elif mode in ["human", "oracle_human"]:
            action = self._get_human_keypress(mode)
        return action
    
    def play(self, mode = None, model = None):
        if mode is None:
            mode = self.envmode
        env = self.env
        obs = env.reset()
        done = False
        while not done:
            action = self._get_action(mode, model, obs)
            if action is None:
                return
            obs, reward, done, timemark = env.step(action)    

    