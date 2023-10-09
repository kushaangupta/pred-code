from W_Gym.W_Gym_Grid2D import W_Gym_Grid2D
from W_Python.W import W
# from gym import spaces
import numpy as np

class task_GoalsActions(W_Gym_Grid2D):
    _param_task = {'p_reward': 1}
    def __init__(self, n_maxTrialsPerBlock = 30, **kwarg):
        self.env_name = "GoalsActions"
        super().__init__(3,3,6, key_preset = "arrow+", n_maxTrialsPerBlock = n_maxTrialsPerBlock, **kwarg)
        # set action space
        # stay, left, up, right, down
        # set rendering dimension names
        self.setup_obs_channel_namedict({'fixation':0, \
                                       'square':1, 'imageA':2, 'imageB':3, \
                                        "dot1":4, "dot2":5})
        # set stages
        state_names = ["fix0", "flash1", "post1", \
                       "flash2", "post2", "choice1", "hold1", \
                        "choice2", "hold2"]
        state_immediateadvance = ["choice1", "choice2"]
        self.setup_state_parameters(state_names = state_names, state_immediateadvance = state_immediateadvance, \
                                    state_timelimits = "ones")
        # task constants 
        self.cccs = np.array([0,2,8,6])
        self.ccs = np.array([3,1,5,7])
        self.setup_human_keys_auto('arrows_plus')
    
    def setup_render_parameters(self):
        plottypes = ["circle", "square", "square", "square", "square", "square"]
        colors = [(255,255,255), (255,255,255), (0, 255, 0), (0, 0, 255), (255,255,255), (255,255,255)]
        radius = [0.02, 0.04, 0.08, 0.08, 0.04, 0.08]
        self._render_set_auto_parameters('obs', plottypes, colors, radius)
        plottypes = ["circle"]
        colors = [(255,0,0)]
        radius = [0.01]
        self._render_set_auto_parameters('action', plottypes, colors, radius)

    def custom_reset_block(self):
        self._param_block['oracle'] = np.random.randint(2)
    
    def custom_reset_trial(self):
        self.gaze.set_pos(1,1)
        cccs = self.cccs 
        ccs = self.ccs 
        ccc_id = np.random.choice(4,2, replace = False)
        cc_id = np.array([0,2]) + np.random.choice(2,1)
        self._param_trial = {'ccc_id': ccc_id, 'ccc_pos': cccs[ccc_id], 'cc_id':cc_id, 'cc_pos':ccs[cc_id]}
        # print(self._param_trial)
        self._env_vars['pos_choice1'] = None
        self._env_vars['pos_choice2'] = None
    
    def get_valid_option3(self, action):
        action = np.where(action == self.ccs)[0][0]
        te = np.array([action, action-1])
        te[te == -1] = 3
        cccs = self.cccs
        return cccs[te].tolist()

    def custom_step_reward(self, action):
        R = 0
        # register pos choice 1 and choice 2
        if self._metadata_state['statenames'][self._state] == "choice1" and \
            self._env_vars['pos_choice1'] is None:
            self._env_vars['pos_choice1'] = action
        if self._metadata_state['statenames'][self._state] == "choice2" and \
            self._env_vars['pos_choice2'] is None:
            self._env_vars['pos_choice2'] = action      
        return R
    
    def custom_step_reward_newstate(self, action):
        R = 0
        # get reward
        if not self._trial_is_error and self._metadata_state['statenames'][self._state] == "ITI":
            if self._env_vars['pos_choice2'] == self._param_trial['ccc_pos'][self._param_block['oracle']]: # correct choice
                treward = self.get_probabilistic_reward(self._param_task['p_reward'])
            else:
                treward = self.get_probabilistic_reward(1 - self._param_task['p_reward'])
            R += treward     
        return R
    
    def custom_step_set_validactions(self):
        if self._metadata_state['statenames'][self._state] in ["fix0","post1", "post2","flash1","flash2"]:
            self._state_valid_actions = 4
        elif self._metadata_state['statenames'][self._state] == "choice1":
            self._state_valid_actions = [1,3,5,7]
        elif self._metadata_state['statenames'][self._state] == "choice2":
            self._state_valid_actions = self.get_valid_option3(self._env_vars['pos_choice1'])
        elif self._metadata_state['statenames'][self._state] in ["hold1"]:
            self._state_valid_actions = np.intersect1d(self._env_vars['pos_choice1'], self._param_trial['cc_pos'])
        elif self._metadata_state['statenames'][self._state] in ["hold2"]:
            self._state_valid_actions = np.intersect1d(self._env_vars['pos_choice2'], self._param_trial['ccc_pos'])
    
    def draw_observation(self):
        # calculate new observation
        if self._metadata_state['statenames'][self._state] in ["fix0","post1", "post2"]:
            self.draw_onehot_2D((1,1),"fixation")
        elif self._metadata_state['statenames'][self._state] in ["flash1"]:
            self.draw_onehot_2D((1,1),"fixation")
            tpos = self._param_trial['ccc_pos']
            self.draw_onehot_2D(self.vec2mat(tpos[0]), "imageA")
            self.draw_onehot_2D(self.vec2mat(tpos[1]), "imageB")
        elif self._metadata_state['statenames'][self._state] in ["flash2"]:
            self.draw_onehot_2D((1,1),"fixation")
            tpos = self._param_trial['cc_pos']
            self.draw_onehot_2D(self.vec2mat(tpos[0]), "square")
            self.draw_onehot_2D(self.vec2mat(tpos[1]), "square")
        elif self._metadata_state['statenames'][self._state] == "choice1":
            self.draw_onehot_2D((1,0),"dot1")
            self.draw_onehot_2D((2,1),"dot1")
            self.draw_onehot_2D((0,1),"dot1")
            self.draw_onehot_2D((1,2),"dot1")
        elif self._metadata_state['statenames'][self._state] in ["hold1"]: 
            (gx, gy) = self.vec2mat(self._env_vars['pos_choice1'])
            self.draw_onehot_2D((gx, gy), "square")
        elif self._metadata_state['statenames'][self._state] == "choice2": 
            for tp in self._state_valid_actions:
                (x,y) = self.vec2mat(tp)
                self.draw_onehot_2D((x,y), "dot2")
        elif self._metadata_state['statenames'][self._state] in ["hold2"]: 
            (gx, gy) = self.vec2mat(self._env_vars['pos_choice2'])
            if self._env_vars['pos_choice2'] == self._param_trial['ccc_pos'][0]:
                self.draw_onehot_2D((gx, gy), "imageA")
            elif self._env_vars['pos_choice2'] == self._param_trial['ccc_pos'][1]:
                self.draw_onehot_2D((gx, gy), "imageB")
        self.flip()
