from W_Gym.W_Gym import W_Gym
from W_Python.W import W
from gym import spaces
import numpy as np

class task_WV(W_Gym):
    def __init__(self, is_ITI = False, *arg, **kwarg):
        super().__init__(is_ITI = is_ITI, *arg, **kwarg)
        self.env_name = "WV"
        self._param_task = W.W_dict_updateonly(self._param_task, kwarg)
        self.observation_space = spaces.Discrete(12) # 9 cues + 1 red + 1 purple + 1 green
        # set action space
        self.action_space = spaces.Discrete(2) # release, hold
        # set rendering dimension names
        self.setup_obs_channel_namedict({'image':np.arange(9).tolist(), 'red':9, 'purple':10, \
                                        'green':11})
        # set stages
        stage_names = ["image", "red", \
                       "purple", "purple_overtime", "green"]
        stage_advanceuponaction = ["red", "purple"]
        self.setW_stage(stage_names = stage_names, stage_advanceuponaction = stage_advanceuponaction, \
                        stage_timings_user= [1,1,1,1,1,999])
        self.effective_actions = [1]

    def _step_set_validactions(self):
        if self.metadata_stage['stage_names'][self.stage] in ["fixation", "image","green"]:
            self.valid_actions = [0]
        elif self.metadata_stage['stage_names'][self.stage] in ["red", "purple"]:
            self.valid_actions = [1,2]

    def _setup_render(self):
        plottypes = ["circle", "image", "square", "square", "square"]
        colors = [(255,255,255), (0,0,0), (255, 0, 0), (255, 0, 255), (0,255,0)]
        radius = [0.02, 0.1, 0.04, 0.04, 0.04]
        self._render_setplotparams('obs', plottypes, colors, radius)
        plottypes = ["arrows"]
        plotparams = [1,0,2,-1]
        self._render_setplotparams('action', plottypes, plotparams = plotparams)

    def _reset_trial(self):
        image = np.random.choice(9,1)
        delay_vals = np.array([1, 5, 10])
        drop_vals = np.array([2, 4, 6])
        delay = np.floor(image/3)
        drop = image % 3
        param = {'image':image, 'delay':delay_vals[delay.astype('int32')], "drop": drop_vals[drop]} # 1,5,10 (change this)
        self.param_trial = param
        self.trial_choice = None
    
    def _step(self, action):
        R_ext = 0
        R_int = 0
        # register pos choice 1 and choice 2
        if self.trial_choice is None:
            if self.metadata_stage['stage_names'][self.stage] in ["red"] and action == 1:
                self.trial_choice = "reject"
            elif self.metadata_stage['stage_names'][self.stage] in ["purple"] and action == 1: 
                self.trial_choice = "accept"
        if self.metadata_stage['stage_names'][self.stage] == "image":    
            sid = self.find_stage('green')
            self.metadata_stage['stage_timings'][sid] = self.param_trial['delay']
        return R_ext, R_int
    
    def _advance_stage(self):
        is_error = False
        if self.metadata_stage['stage_names'][self.stage] == "red" and self.is_effective_action:
            stage = self.find_stage('ITI')
        elif self.metadata_stage['stage_names'][self.stage] == "purple" and self.is_effective_action:
            stage = self.find_stage('green')
        else:
            stage = self.stage + 1

        if stage < len(self.metadata_stage['stage_names']) and self.metadata_stage['stage_names'][stage] == "purple_overtime":
            is_error = True 
        return stage, is_error

    def _step_after(self, action):
        R_ext = 0
        R_int = 0

        # get reward
        if not self.trial_is_error and self.trial_choice == "accept" and \
            self.metadata_stage['stage_names'][self.stage] == "ITI":
            R_ext += self.param_trial['drop'] * self.Rewards['R_reward']
        return R_ext, R_int

    def _draw_obs(self):
        if self.metadata_stage['stage_names'][self.stage] == "image":
            timg = W.W_onehot(self.param_trial['image'], 9)
            self.draw("image", timg)
        else:
            self.draw(self.metadata_stage['stage_names'][self.stage], 1)
        self.flip()

    def _render_frame_obs_format(self, obs, lst):
        c = 0
        for i, j in lst.items():
            if i == "image":
                obs[c] = obs[c].reshape((3,3))
                obs[c] = obs[c] * 128 + np.any(obs[c] > 0) * 127
            c += 1
        return obs
