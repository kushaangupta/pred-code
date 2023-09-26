from gym import spaces
from W_Gym.W_Gym import W_Gym
from W_Python import W_tools as W
import numpy as np

class task_TwoStep_simple(W_Gym):
    task_param = {'p_switch': 0.05, 'p_trans':[1,1], 'p_reward_high': 1, 'p_reward_low': 0}
    high_state = None
    def __init__(self, *arg, **kwarg):
        super().__init__(is_ITI=False, *arg, **kwarg)
        self.observation_space = spaces.Discrete(3)
        # set action space
        self.action_space = spaces.Discrete(3) # fix, L, R
        # set rendering dimension names
        self.setup_obs_Name2DimNumber({'planet0':0, 'planet1':1, 'planet2':2})
        # set stages
        stage_names = ["stage1", "stage2"]
        stage_advanceuponaction = ["stage1", "stage2"]
        self.setW_stage(stage_names = stage_names, stage_advanceuponaction = stage_advanceuponaction)
        self.effective_actions = [1,2]

    def _setup_render(self):
        plottypes = ["circle", "circle", "circle"]
        colors = [(100,100,100), (0,255,0), (0,0,255)]
        radius = [0.25, 0.25, 0.25]
        self._render_setplotparams('obs', plottypes, colors, radius)
        plottypes = ["binary"]
        self._render_setplotparams('action', plottypes, plotparams = [1,2])

    def _reset_block(self):
        self.high_state = np.random.choice(2,1)[0]
        self.info = []

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
        param = {'transition':trans.astype(int), 'reward':r} # 1,5,10 (change this)
        self.param_trial = param
        self.planet = None

    def _step(self, action):
        R_ext = 0
        R_int = 0
        if self.metadata_stage['stage_names'][self.stage] == "stage1" and self.is_effective_action:
            self.planet = self.param_trial['transition'][action-1]
        if self.metadata_stage['stage_names'][self.stage] == "stage2":
            R_ext = self.param_trial['reward'][self.planet]
        return R_ext, R_int
    
    def _step_set_validactions(self):
        if self.metadata_stage['stage_names'][self.stage] in ["stage1"]:
            self.valid_actions = [1,2]
        elif self.metadata_stage['stage_names'][self.stage] in ["stage2"]:
            self.valid_actions = [0]

    def _draw_obs(self):
        self.info = [self.trial_counter, self.t]
        if self.metadata_stage['stage_names'][self.stage] == "stage1":
            self.draw('planet0', 1)
        elif self.metadata_stage['stage_names'][self.stage] == "stage2":
            if self.planet == 0:
                self.draw('planet1',1)
            elif self.planet == 1:
                self.draw('planet2',1)
        self.flip()
