from W_Gym.W_Gym import W_Gym
from W_Python.W import W
# from gym import spaces
import random
import numpy as np

class task_TwoStep(W_Gym):
    _param_task = {'p_switch_reward': 0, 'p_switch_transition': 0, \
                  'ps_high_state':[1], 'ps_low_state':None, \
                  'ps_common_trans':[1], 'ps_ambiguity': [0], \
                  'is_random_common0': False, \
                  }
    
    def __init__(self, is_ITI = False, *arg, **kwarg):
        super().__init__(is_ITI = is_ITI, *arg, **kwarg)
        self.env_name = "TwoStep"
        # observation space
        self.setup_obs_dim(3) # state 0, state 1, state 2
        # set action space
        self.setup_action_dim(2) # shuttle 1, shuttle 2
        # set rendering dimension names
        self.setup_obs_channel_namedict({'planet0':0, 'planet1':1, 'planet2':2})
        # set stages
        state_names = ["stage1"]
        state_immediateadvance = ["stage1"]
        self.setup_state_parameters(state_names=state_names, state_immediateadvance=state_immediateadvance)
        self.setup_human_keys_auto('binary')

    def custom_savename(self):
        pST0 = 'T' if self._param_task['is_random_common0'] else 'F'
        R = W.W_str_numbers_separated_by_underscore(self._param_task['ps_high_state']*100)
        SR = self._param_task['p_switch_reward']*1000
        pT = W.W_str_numbers_separated_by_underscore(self._param_task['ps_common_trans']*100)
        pST = self._param_task['p_switch_transition']*1000
        A = W.W_str_numbers_separated_by_underscore(self._param_task['ps_ambiguity']*100)
        return f"pR{R:.0f}_pSR{SR:.0f}_pT{pT:.0f}_pST{pST:.0f}{pST0}_pA{A:.0f}"

    def custom_reset_block(self):
        p = random.sample(W.W_enlist(self._param_task['ps_common_trans']),1)[0]
        if self._param_task['is_random_common0'] and np.random.rand() < 0.5:
            p = 1 - p
        self._env_vars['p_trans'] = [p,p]
        self._env_vars['dominant_trans'] = 1 if p > 0.5 else 2
        self._env_vars['high_state'] = np.random.choice(2,1)[0]

        tid = np.random.choice(len(W.W_enlist(self._param_task['ps_high_state'])),1)[0]
        p = W.W_enlist(self._param_task['ps_high_state'])[tid]
        
        self._param_block['p_reward_high'] = p

        if self._param_task['ps_low_state'] is None:
            p = 1-p
        else:
            p = self._param_task['ps_low_state'][tid]
        self._param_block['p_reward_low'] = p

        p = random.sample(W.W_enlist(self._param_task['ps_ambiguity']),1)[0]
        self._param_block['p_ambiguity'] = p
        self._env_vars['planet'] = None
    
    def custom_reset_trial(self):
        if np.random.rand() < self._param_task['p_switch_reward']: # flip reward
            self._env_vars['high_state']  = 1- self._env_vars['high_state'] 
        if np.random.rand() < self._param_task['p_switch_transition']: # flip transition
            self._env_vars['p_trans'] = [1 - x for x in self._env_vars['p_trans']]
            self._env_vars['dominant_trans'] = 3 - self._env_vars['dominant_trans']
        r_high = np.array(np.random.rand() < self._param_block['p_reward_high']).astype(int)
        r_low = np.array(np.random.rand() < self._param_block['p_reward_low']).astype(int)
        r = np.zeros(2)
        r[self._env_vars['high_state']] = r_high
        r[1 - self._env_vars['high_state']] = r_low

        trans = np.zeros(2)
        for i in range(2):
            if np.random.rand() < self._env_vars['p_trans'][i]:
                trans[i] = i
            else:
                trans[i] = 1-i

        if np.random.rand() < self._param_block['p_ambiguity']:
            planet = np.random.choice(2,1)[0]
        else:
            planet = None
        self._param_trial.update({'transition':trans.astype(int), 'rewardplanet':r, 'randomplanet': planet})
                                      
    def custom_step_reward(self, action):
        reward = 0
        if self._metadata_state['statenames'][self._state] == "stage1":
            self._env_vars['spaceship'] = action
            self._env_vars['planet'] = self._param_trial['transition'][self._env_vars['spaceship']]
            reward += self._param_trial['rewardplanet'][self._env_vars['planet']]
        return reward

    def draw_observation(self):
        if self._env_vars['planet'] is None:
            self.draw_onehot('planet0', 1)
        else:
            if self._param_trial['randomplanet'] is not None:
                planet = self._param_trial['randomplanet']
            else:
                planet = self._env_vars['planet']
            if planet == 0:
                self.draw_onehot('planet1',1)
            elif planet == 1:
                self.draw_onehot('planet2',1)
        self.flip()

    def setup_render_parameters(self):        
        plottypes = ["circle", "circle", "circle"]
        colors = [(100,100,100), (0,255,0), (0,0,255)]
        radius = [0.25, 0.25, 0.25]
        position = [None, None, None]
        self._render_set_auto_parameters('obs', plottypes, colors, radius, position)
        plottypes = ["action_binary"]
        self._render_set_auto_parameters('action', plottypes)

    def format_obs_for_save(self, obs):
        return np.sum(obs * np.array([0,1,2]))