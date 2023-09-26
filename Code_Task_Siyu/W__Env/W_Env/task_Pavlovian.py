from W_Gym.W_Gym import W_Gym
from W_Python.W import W
# from gym import spaces
import random
import numpy as np

class task_Pavlovian(W_Gym):
    _param_task = {'baseline': 5, 'CS_duration': 1, \
                  'CS_US_delay': 5, "max_US": 1, 'action_delay': 3}
    def __init__(self, is_ITI = False, *arg, **kwarg):
        super().__init__(is_ITI = is_ITI, *arg, **kwarg)
        self.env_name = "Pavlovian"
        # observation space
        self.setup_obs_dim(2) # 0: CS, 1: US
        # set action space
        self.setup_action_dim(2) # 0: hold, 1: lick
        # set rendering dimension names
        self.setup_obs_channel_namedict({'CS':0, 'US':1})
        # set stages
        state_names = ["wait", "CS", "CS_US_delay", "US"]
        state_timelimits = [self._param_task['baseline'], self._param_task['CS_duration'], self._param_task['CS_US_delay'], \
                            self._param_task['max_US']]
        self.setup_state_parameters(state_names=state_names, \
                                    state_timelimits = state_timelimits)
        self.setup_human_keys_auto('binary')

    def transform_actions(self, action_motor):
        self._env_vars['action_history'] += [action_motor]
        action = self._env_vars['action_history'].pop(0)
        return action

    def custom_reset_trial(self):
        self._env_vars['action_history'] = W.W_enlist(np.zeros(self._param_task['action_delay'], dtype = 'int'))

    def custom_step_reward(self, action):
        reward = 0
        if self._metadata_state['statenames'][self._state] == "US" and action == 1:
            reward += 1
        if action == 1:
            reward -= 0.1
        return reward

    def draw_observation(self):
        if self._metadata_state['statenames'][self._state] == "CS":
            self.draw_onehot('CS', 1)
        elif self._metadata_state['statenames'][self._state] == "US": 
            self.draw_onehot('US', 1)
        else:
            self.blankscreen()
        self.flip()

    def setup_render_parameters(self):        
        plottypes = ["circle", "circle"]
        colors = [(0,255,0), (0,0,255)]
        radius = [0.25, 0.25]
        position = [(0.2, 0.8), (0.8, 0.2)]
        self._render_set_auto_parameters('obs', plottypes, colors, radius, position)
        plottypes = ["action_binary"]
        self._render_set_auto_parameters('action', plottypes)