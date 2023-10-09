from gym import spaces
from W_Gym.W_Gym import W_Gym
from W_Python import W_tools as W
import numpy as np

class task_TwoStep_Confidence(W_Gym):
    task_param = {'p_switch': 0.05, 'p_reward_high': 0.9, 'p_reward_low': 0.1, 'reward_safe': 0.1}
    high_state = None
    def __init__(self, *arg, **kwarg):
        super().__init__(*arg, **kwarg)
        self.observation_space = spaces.Discrete(6)
        # set action space
        self.action_space = spaces.Discrete(4) # fix, L, R, safebet

        # set rendering dimension names
        self.setup_obs_Name2DimNumber({'fixation':0, \
                                       'planet0':1, 'planet1':2, 'planet2':3,\
                                       'guess':4, 'safebet':5})
        # set stages
        stage_names = ["fixation", "stage1", "guessplanet", "stage2"]
        stage_advanceuponaction = ["stage1", "guessplanet", "stage2"]
        self.setW_stage(stage_names = stage_names, stage_advanceuponaction = stage_advanceuponaction)
        self.effective_actions = [1, 2]

    def _setup_render(self):
        plottypes = ["circle", "circle", "circle", "circle", "circle", "square"]
        colors = [(255,255,255), (100,100,100), (0,255,0), (0,0,255), (255,255,255), (255,255,255)]
        radius = [0.02, 0.25, 0.25, 0.25, 0.05, 0.05]
        position = [None, None, None, None, [0.5, 0.9], [0.9, 0.1]]
        self._render_setplotparams('obs', plottypes, colors, radius, position)
        plottypes = ["arrows"]
        self._render_setplotparams('action', plottypes, plotparams = [1,3,2,-1])

    def _reset_block(self):
        p = np.random.rand()
        p = np.max((p, 1-p))
        self.task_param['p_trans'] = [p,p]
        self.high_state = np.random.choice(2,1)[0]

    def _reset_trial(self):
        if np.random.rand() < self.task_param['p_switch']:
            self.high_state = 1- self.high_state
        r_high = np.array(np.random.rand() < self.task_param['p_reward_high']).astype(int)
        r_low = np.array(np.random.rand() < self.task_param['p_reward_low']).astype(int)
        r = np.zeros(2)
        r[self.high_state] = r_high
        r[1 - self.high_state] = r_low
        trans = np.zeros(2)
        for i in range(2):
            if np.random.rand() < self.task_param['p_trans'][i]:
                trans[i] = i
            else:
                trans[i] = 1-i
        is_safebet_avail = np.array(np.random.rand(3) < 0.5)
        param = {'transition':trans.astype(int), 'reward':r, 'is_safebet_avail':is_safebet_avail} # 1,5,10 (change this)
        self.param_trial = param
        self.planet = None

    def _step(self, action):
        R_ext = 0
        R_int = 0
        if self.metadata_stage['stage_names'][self.stage] == "stage1":
            if self.is_effective_action:
                self.planet = self.param_trial['transition'][action -1]
            else:
                R_ext += self.task_param['reward_safe']
        if self.metadata_stage['stage_names'][self.stage] == "guessplanet":
            if action == 3:
                R_ext += self.task_param['reward_safe']
            else:
                R_ext += np.array(self.planet == action - 1).astype(int)
        if self.metadata_stage['stage_names'][self.stage] == "stage2":
            if action == 3:
                R_ext += self.task_param['reward_safe']
            else:
                R_ext += self.param_trial['reward'][self.planet]
        return R_ext, R_int
    
    
    def _advance_stage(self):
        is_error = False
        if self.metadata_stage['stage_names'][self.stage] == "stage1" and self.planet is None:
            stage = self.find_stage('ITI')
        else:
            stage = self.stage + 1
        return stage, is_error
    
    def _step_set_validactions(self):
        if self.metadata_stage['stage_names'][self.stage] in ["fixation"]:
            self.valid_actions = [0]
        elif self.metadata_stage['stage_names'][self.stage] in ["stage1"]:
            if self.param_trial['is_safebet_avail'][0]:
                self.valid_actions = [1,2,3]
            else:
                self.valid_actions = [1,2]
        elif self.metadata_stage['stage_names'][self.stage] in ["guessplanet"]:
            if self.param_trial['is_safebet_avail'][1]:
                self.valid_actions = [1,2,3]
            else:
                self.valid_actions = [1,2]
        elif self.metadata_stage['stage_names'][self.stage] in ["stage2"]:
            if self.param_trial['is_safebet_avail'][2]:
                self.valid_actions = None
            else:
                self.valid_actions = [0,1,2]

    def _draw_obs(self):
        if self.metadata_stage['stage_names'][self.stage] == "fixation":
            self.draw("fixation",1)
        elif self.metadata_stage['stage_names'][self.stage] == "stage1":
            self.draw('planet0', 1)
            if self.param_trial['is_safebet_avail'][0]:
                self.draw('safebet', 1)
        elif self.metadata_stage['stage_names'][self.stage] == "guessplanet":
            self.draw('guess', 1)
            if self.param_trial['is_safebet_avail'][1]:
                self.draw('safebet', 1)
        elif self.metadata_stage['stage_names'][self.stage] == "stage2":
            if self.planet == 0:
                self.draw('planet1',1)
            elif self.planet == 1:
                self.draw('planet2',1)
            if self.param_trial['is_safebet_avail'][2]:
                self.draw('safebet', 1)
        self.flip()
