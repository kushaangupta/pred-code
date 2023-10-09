from W_Gym.W_Gym import W_Gym
from W_Python.W import W
from gym import spaces
import numpy as np

class task_ITC(W_Gym):
    _param_task = {'delay': [1,1,1,5,5,5,10,10,10], 'drop': [2,4,6,2,4,6,2,4,6]}
    def __init__(self, is_ITI = False, *arg, **kwarg):
        super().__init__(is_ITI = is_ITI, *arg, **kwarg)
        self.env_name = "ITC"
        self.setup_obs_dim(12)  # 9 cues + 1 red + 1 purple + 1 green
        # set action space
        self.setup_action_dim(2) # release, hold
        # set rendering dimension names
        self.setup_obs_channel_namedict({'image':np.arange(9).tolist(), 'red':9, 'purple':10, \
                                        'green':11})
        # set stages
        state_names = ["image", "red", \
                       "purple", "green", "reward"]
        state_immediateadvance = ["red", "purple"]
        self.setup_state_parameters(state_names = state_names, state_immediateadvance = state_immediateadvance, \
                        state_timelimits= [1,1,1,99,1])
        self.setup_human_keys_auto('binary')

    def custom_reset_trial(self):
        image = int(np.random.choice(9,1).astype('int32'))
        delay_vals = self._param_task['delay']
        drop_vals = self._param_task['drop']
        self._param_trial = {'image':image, 'delay':delay_vals[image], "drop": drop_vals[image]} # 1,5,10 (change this)
        self._env_vars['choice'] = None

    def custom_step_set_validactions(self):
        if self._metadata_state['statenames'][self._state] in ["image","green","reward"]:
            self._state_valid_actions = [0]
        elif self._metadata_state['statenames'][self._state] in ["red", "purple"]:
            self._state_valid_actions = [0,1]
            self._state_effective_actions = 1

    def custom_step_reward(self, action):
        R = 0
        # register pos choice 1 and choice 2
        if self._env_vars['choice'] is None:
            if self._metadata_state['statenames'][self._state] in ["red"] and action == 1:
                self._env_vars['choice'] = "reject"
            elif self._metadata_state['statenames'][self._state] in ["purple"] and action == 1: 
                self._env_vars['choice'] = "accept"
        if self._metadata_state['statenames'][self._state] == "image":    
            sid = self._find_state('green')
            self._metadata_state['timelimits'][sid] = self._param_trial['delay']
        if self._metadata_state['statenames'][self._state] == "reward":
            R += self._param_trial['drop'] * self._param_rewards['R_reward']
        return R
    
    def custom_state_transition(self, action, is_effective, is_transition):
        is_error = False
        R = 0
        is_done = False
        if self._metadata_state['statenames'][self._state] == "red" and is_effective:
            is_done = self._advance_trial()
        elif self._metadata_state['statenames'][self._state] == "purple":
            if is_effective:
                self._go_to_state('green')
            elif is_transition and self._env_vars['choice'] is None:
                is_error = True
        elif is_transition:
            R, is_done = self._auto_state_transition()
        return is_error, R, is_done

    def draw_observation(self):
        if self._metadata_state['statenames'][self._state] == "image":
            timg = W.W_onehot(self._param_trial['image'], 9)
            self.draw_onehot("image", timg)
        elif self._metadata_state['statenames'][self._state] != "reward":
            self.draw_onehot(self._metadata_state['statenames'][self._state], 1)
        self.flip()
    
    def setup_render_parameters(self):
        plottypes = ["image", "square", "square", "square"]
        colors = [(0,0,0), (255, 0, 0), (255, 0, 255), (0,255,0)]
        radius = [0.1, 0.04, 0.04, 0.04]
        self._render_set_auto_parameters('obs', plottypes, colors, radius)
        plottypes = ["action_binary"]
        # plotparams = [1,0,2,-1]
        self._render_set_auto_parameters('action', plottypes)
    
    def custom_render_frame_obs_format(self, obs, lst):
        out = []
        for i, j in lst.items():
            if i == "image":
                tobs = obs[j].reshape((3,3))
                out += [tobs * 128 + np.any(tobs > 0) * 127]
            else:
                out += [obs[j]]
        return out