from gym import spaces
from W_Gym.W_Gym import W_Gym
from W_Python import W_tools as W
import numpy as np
import random

class task_TwoStep_Confidence_2frame(W_Gym):
    task_hyper_param = {'p_switch_reward': 0, 'p_switch_transition': 0, \
                  'ps_high_state':[1], 'ps_common_trans':[1], \
                  'is_random_common0': False, \
                  }
    task_param = {}
    high_state = None
    is_fixed = False
    def __init__(self, *arg, **kwarg):
        super().__init__(is_ITI = False, *arg, **kwarg)
        self.task_hyper_param = W.W_dict_updateonly(self.task_hyper_param, kwarg)
        self.observation_space = spaces.Discrete(5)
        # set action space
        self.action_space = spaces.Discrete(3) # take reward, shuttle1,2
        # set rendering dimension names
        self.setup_obs_Name2DimNumber({'planet0':0, 'planet1':1, 'planet2':2, 'stage1': 3, 'stage2': 4})
        # set stages
        stage_names = ["stage1", "stage2"]
        stage_advanceuponaction = ["stage1", "stage2"]
        self.setW_stage(stage_names = stage_names, stage_advanceuponaction = stage_advanceuponaction)

    def _reset_block(self):
        p = random.sample(self.task_hyper_param['ps_common_trans'],1)[0]
        # need to implement continuous p
        if self.task_hyper_param['is_random_common0'] and np.random.rand() < 0.5:
            p = 1 - p
        self.task_param['p_trans'] = [p,p]
        self.high_state = np.random.choice(2,1)[0]
        p = random.sample(self.task_hyper_param['ps_high_state'], 1)[0]
        # need to implement continuous p
            # p = np.random.rand()
            # p = np.max((p, 1-p))
        self.task_param['p_reward_high'] = p
        self.task_param['p_reward_low'] = 1-p

    def _reset_trial(self):
        if np.random.rand() < self.task_hyper_param['p_switch_reward']: # flip reward
            self.high_state = 1- self.high_state
        if np.random.rand() < self.task_hyper_param['p_switch_transition']: # flip transition
            self.task_param['p_trans'] = [1 - x for x in self.task_param['p_trans']]
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
        self.param_trial = {'transition':trans.astype(int), 'reward':r}
        self.planet = None

    def _step_set_validactions(self):
        if self.metadata_stage['stage_names'][self.stage] in ["stage1"]:
            self.valid_actions = [1,2]
        elif self.metadata_stage['stage_names'][self.stage] in ["stage2"]:
            self.valid_actions = [0]
    
    def _step(self, action):
            R_ext = 0
            R_int = 0
            if self.metadata_stage['stage_names'][self.stage] == "stage1":
                self.shuttle = action - 1
                self.planet = self.param_trial['transition'][self.shuttle]
            if self.metadata_stage['stage_names'][self.stage] == "stage2":
                R_ext += self.param_trial['reward'][self.planet]
            return R_ext, R_int

    def _draw_obs(self):
        if self.metadata_stage['stage_names'][self.stage] == "stage1":
            self.draw('planet0', 1)
            self.draw('stage1', 1)
        elif self.metadata_stage['stage_names'][self.stage] == "stage2":
            self.draw('stage2', 1)
            if self.planet == 0:
                self.draw('planet1',1)
            elif self.planet == 1:
                self.draw('planet2',1)
        self.flip()

    def _setup_render(self):
        plottypes = ["circle", "circle", "circle", "square", "square"]
        colors = [(100,100,100), (0,255,0), (0,0,255), (255,255,255), (255,255,255)]
        radius = [0.25, 0.25, 0.25, 0.05, 0.05]
        position = [None, None, None,[0.2,0.8],[0.8,0.8]]
        self._render_setplotparams('obs', plottypes, colors, radius, position)
        plottypes = ["binary"]
        self._render_setplotparams('action', plottypes, plotparams = [1,2])

   
    
