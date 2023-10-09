from gym import spaces
from W_Gym.W_Gym import W_Gym
from W_Python import W_tools as W
import numpy as np
import random

class task_TwoStep_Ambiguity_1frame(W_Gym):
    task_hyper_param = {'p_switch_reward': 0, 'p_switch_transition': 0, \
                  'ps_high_state':[1], 'ps_low_state':None, 'ps_common_trans':[1], \
                  'ps_ambiguity': [0], \
                  'is_random_common0': False, \
                  }
    task_param = {}
    high_state = None
    is_fixed = False
    def __init__(self, *arg, **kwarg):
        super().__init__(is_ITI = False, *arg, **kwarg)
        self.task_hyper_param = W.W_dict_updateonly(self.task_hyper_param, kwarg)

        self.observation_space = spaces.Discrete(3)
        # set action space
        self.action_space = spaces.Discrete(2) # shuttle1,2
        # set rendering dimension names
        self.setup_obs_Name2DimNumber({'planet0':0, 'planet1':1, 'planet2':2})
        # set stages
        stage_names = ["stage1"]
        stage_advanceuponaction = ["stage1"]
        self.setW_stage(stage_names = stage_names, stage_advanceuponaction = stage_advanceuponaction)
        self.planet = None

    def get_versionname(self):
        tstr = 'T' if self.task_hyper_param['is_random_common0'] else 'F'
        return f"pR{self.task_hyper_param['ps_high_state']*100:.0f}_pSR{self.task_hyper_param['p_switch_reward']*1000:.0f}_pT{self.task_hyper_param['ps_common_trans']*100:.0f}_pST{self.task_hyper_param['p_switch_transition']*1000:.0f}_PST0{tstr}_pA{self.task_hyper_param['ps_ambiguity']*100:.0f}"
        
    def _reset_block(self):
        self.info = {'info_task':[], 'info_block':[], 'info_trial':[], 'info_step': []}
        p = random.sample(W.enlist(self.task_hyper_param['ps_common_trans']),1)[0]
        # need to implement continuous p
        if self.task_hyper_param['is_random_common0'] and np.random.rand() < 0.5:
            p = 1 - p
        self.task_param['p_trans'] = [p,p]
        self.high_state = np.random.choice(2,1)[0]

        tid = np.random.choice(len(W.enlist(self.task_hyper_param['ps_high_state'])),1)[0]
        p = W.enlist(self.task_hyper_param['ps_high_state'])[tid]
        # need to implement continuous p
            # p = np.random.rand()
            # p = np.max((p, 1-p))
        self.task_param['p_reward_high'] = p

        if self.task_hyper_param['ps_low_state'] is None:
            p = 1-p
        else:
            p = self.task_hyper_param['ps_low_state'][tid]

        self.task_param['p_reward_low'] = p

        p = random.sample(W.enlist(self.task_hyper_param['ps_ambiguity']),1)[0]
        self.task_param['p_ambiguity'] = p

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

        if np.random.rand() < self.task_param['p_ambiguity']:
            planet = np.random.choice(2,1)[0]
        else:
            planet = None
        self.param_trial = {'transition':trans.astype(int), 'rewardplanet':r, 'randomplanet': planet, 'highstate': self.high_state}

    def _block_info(self):
        self.info_block['params'] = self.task_param

    def _trial_info(self):
        self.info_trial['params'] = self.param_trial 
    
    def _step(self, action):
            R_ext = 0
            R_int = 0
            if self.metadata_stage['stage_names'][self.stage] == "stage1":
                self.shuttle = action
                self.planet = self.param_trial['transition'][self.shuttle]
                R_ext += self.param_trial['rewardplanet'][self.planet]
            return R_ext, R_int

    def _draw_obs(self):
        if self.planet is None:
            self.draw('planet0', 1)
        else:
            if self.param_trial['randomplanet'] is not None:
                planet = self.param_trial['randomplanet']
            else:
                planet = self.planet
            if planet == 0:
                self.draw('planet1',1)
            elif planet == 1:
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

    def format4save(self):
        d = super().format4save()
        d.obs = np.matmul(np.stack(d.obs), np.array([[0],[1],[2]])).squeeze()
        d.obs_next = np.matmul(np.stack(d.obs_next), np.array([[0],[1],[2]])).squeeze()
        # d.trialID = np.ceil(d.tot_t/2)
        d.stage = 1 - d.stage
        d.transition += 1
        return d