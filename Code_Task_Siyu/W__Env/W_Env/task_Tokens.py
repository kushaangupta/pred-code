from W_Gym.W_Gym_Grid2D import W_Gym_Grid2D
from W_Python.W import W
import numpy as np
import random

class task_Tokens(W_Gym_Grid2D):
    _param_task = {'p_reward': 1, 'CueValues': np.array([1,2,-1,-2]), 'endowment': 0, \
                   'rewardtype': 'juice'}
    def __init__(self, *arg, **kwarg):
        super().__init__(1,3,4, key_preset = "binary", is_ITI = False, n_maxTrialsPerBlock = 180, \
                         option_obs_augment = ["tokens", "reward", "motor"], *arg, **kwarg)
        self._param_task['CueValues'] = np.array(self._param_task['CueValues'])
        # set action space
        # stay, left, right
        # set rendering dimension names
        self.setup_obs_channel_namedict({'image1':0, 'image2':1, 'image3':2, 'image4':3})
        # set stages
        state_names = ["choice"]
        state_immediateadvance = ["choice"]
        self.setup_state_parameters(state_names = state_names, state_immediateadvance = state_immediateadvance,\
                        state_timelimits = "ones")        
        self.setup_human_keys_auto('binary')

    def format_supervised(self, data):
        IMid = np.array([data['cueLR_1'], data['cueLR_2']])
        for i in range(2):
            tstr = "image" + str(IMid[i])
            self.draw_onehot_2D((0, i*2), tstr)
        self.flip()
        self._last_action_motor = data['last_action']
        self._last_reward = data['last_reward']
        return self._get_obs(), None

    def custom_reset(self):
        self._env_vars['tokens'] = 0

    def custom_reset_block(self):
        self._param_block['ImageValues'] = self._param_task['CueValues'].copy()
        random.shuffle(self._param_block['ImageValues'])
        te = np.array([], dtype = int)
        for repi in range(9):
            te = np.hstack((te, np.random.choice(12, 12, replace = False)))
        cue_L_cond = np.array([1,2,1,3,1,4,2,3,2,4,3,4])
        cue_R_cond = np.array([2,1,3,1,4,1,3,2,4,2,4,3])
        self._temp_vars['cue_L'] = cue_L_cond[te]
        self._temp_vars['cue_R'] = cue_R_cond[te]
        self._temp_vars['cue_LR'] = np.moveaxis(np.vstack((self._temp_vars['cue_L'], self._temp_vars['cue_R'])), 1, 0) - 1
        self._temp_vars['condition'] = (np.ceil((te+1)/2)).astype(int)
        self.cashout()

    def custom_reset_trial(self):
        self.gaze.set_pos(0,1)
        triali = self._count_block_trial
        IM_id = self._temp_vars['cue_LR'][triali] # np.random.choice(4,2, replace = False)
        R_side = self._param_block['ImageValues'][IM_id]
        randv = 0+(np.random.rand(2) < self._param_task['p_reward'])
        R_side = R_side * randv
        self._temp_vars['trial_until_cashout'] -= 1
        iscashout = self._temp_vars['trial_until_cashout'] == 0
        self._param_trial = {'IM_id': IM_id, 'R_side': R_side, 'condition': self._temp_vars['condition'][triali], \
                             'cue_L': self._temp_vars['cue_L'][triali], 'cue_R': self._temp_vars['cue_R'][triali], \
                             'is_cashout': iscashout}

    # def _step_set_validactions(self):
    #     # if self._metadata_state['statenames'][self._state] in ["fixation"]:
    #     #     self.valid_actions = 1
    #     if self._metadata_state['statenames'][self._state] == "choice":
    #         self.valid_actions = [0,1]
    
    def draw_observation(self):
        # calculate new observation
        # if self.metadata_stage['statenames'][self._state] in ["fixation"]:
        # self.draw((0,1),"fixation")
        if self._metadata_state['statenames'][self._state] == "choice":
            for i in range(2):
                IMid = self._param_trial['IM_id'][i]
                tstr = "image" + str(IMid+1)
                self.draw_onehot_2D((0, i*2), tstr)
        self.flip()

    def custom_step_reward(self, action):
        R = 0
        # register pos choice 1 and choice 2
        if self._metadata_state['statenames'][self._state] == "choice":
            self._env_vars['choice'] = int(action/2)      
            Rtoken = self.update_tokens(self._param_trial['R_side'][self._env_vars['choice']])
            if self._param_trial['is_cashout']:
                Rjuice = self.cashout()
            else:
                Rjuice = 0
            R = Rtoken if self._param_task['rewardtype'] == "tokens" else Rjuice
        return R

    def update_tokens(self, tokenchange): 
        token_before = self._env_vars['tokens']
        self._env_vars['tokens'] += tokenchange
        self._env_vars['tokens'] = np.max([self._env_vars['tokens'],0])
        self._env_vars_after['tokens_after'] = self._env_vars['tokens']
        return self._env_vars['tokens'] - token_before
    
    def cashout(self):
        R = self._env_vars['tokens']
        self._env_vars['tokens'] = self._param_task['endowment']
        self._temp_vars['trial_until_cashout'] = np.random.choice([4,5,6])
        return R
    
    def setup_render_parameters(self):
        plottypes = ["square", "square", "square", "square"]
        colors = [(255,0,255), (0, 255, 0), (0, 0, 255), (255,255,0)] # purple, green, blue, yellow
        radius = [0.08, 0.08, 0.08, 0.08]
        self._render_set_auto_parameters('obs', plottypes, colors, radius)
        plottypes = ["circle"]
        colors = [(255,0,0)]
        radius = [0.01]
        self._render_set_auto_parameters('action', plottypes, colors, radius)

    def custom_render_frame_reward(self):
        self._render_text('_renderer_text_Reward', f"R = {self._last_reward}, tokens = {self._env_vars['tokens']}")
        
        

    