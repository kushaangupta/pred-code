from gym import spaces
from W_Gym.W_Gym import W_Gym
from W_Python import W_tools as W
import numpy as np

class task_Horizon(W_Gym):
    task_param = {'mu':[40, 60], 'sd':8, 'diff': [-20,-12,-8,-4,4,8,12,20], \
        'n_instructed':4, 'horizon':[5,10]}
    def __init__(self, *arg, **kwarg):
        super().__init__(*arg, **kwarg)
        self.observation_space = spaces.Discrete(13) # 1 + 2 + 10
        # set action space
        self.action_space = spaces.Discrete(3) # fix, L, R
        # set rendering dimension names
        self.setup_obs_Name2DimNumber({'fixation':0, \
                                       'cue':[1,2], \
                                       'horizon': np.arange(3, 13).tolist()})
        # set stages
        stage_names = ["fixation", "horizon", "guided", "choice"]
        stage_advanceuponaction = ["choice", "guided"]
        self.setW_stage(stage_names = stage_names, stage_advanceuponaction = stage_advanceuponaction)
        self.effective_actions = [1, 2]

    def _setup_render(self):
        plottypes = ["circle", "image", "image"]
        colors = [(255,255,255), (0,0,0), (0,0,0)]
        radius = [0.02, 0.1, 0.5]
        position = [[0.5,0.5], [0.5,0.3],[0.5,0.7]]
        self._render_setplotparams('obs', plottypes, colors, radius, position)
        plottypes = ["binary"]
        self._render_setplotparams('action', plottypes, plotparams = [1,2])


    def _reset_trial(self):
        mainbandit = np.random.choice(2,1)[0]
        mu = np.zeros(2)
        mu[mainbandit] = np.random.choice(self.task_param['mu'], 1)
        mu[1-mainbandit] = mu[mainbandit] + np.random.choice(self.task_param['diff'])
        sd = self.task_param['sd']
        horizon = np.random.choice(self.task_param['horizon'],1)[0]
        r = np.zeros((horizon,2))
        r[:,0] = np.clip(np.random.normal(mu[0], sd, (horizon)).round(),a_min = 1, a_max = 99)
        r[:,1] = np.clip(np.random.normal(mu[1], sd, (horizon)).round(),a_min = 1, a_max = 99)
        opts_guided = np.array([[0,0,0,1],[0,0,1,1],[1,1,1,0],[1,1,0,0]])
        guided = opts_guided[np.random.choice(4,1)[0]]
        np.random.shuffle(guided)
        param = {'horizon':horizon, 'r':r, 'guided':guided} # 1,5,10 (change this)
        self.param_trial = param
        self.number_of_trials_left = horizon
    
    def _step(self, action):
        R_ext = 0
        R_int = 0
        if self.metadata_stage['stage_names'][self.stage] in ["choice", "guided"]:
            R_ext = self.param_trial['r'][self.param_trial['horizon'] - self.number_of_trials_left][action-1]
        return R_ext, R_int
    
    def _step_set_validactions(self):
        if self.metadata_stage['stage_names'][self.stage] in ["fixation", "horizon"]:
            self.valid_actions = [0]
            self.trialnumber = 0
        elif self.metadata_stage['stage_names'][self.stage] in ["guided"]:
            self.valid_actions = [self.param_trial['guided'][self.trialnumber] + 1]
        elif self.metadata_stage['stage_names'][self.stage] in ["choice"]:
            self.valid_actions = [1,2]

    def _advance_stage(self):
        is_error = False
        if self.metadata_stage['stage_names'][self.stage] == "choice" and self.number_of_trials_left == 1:
            stage = self.find_stage('ITI')
        elif self.metadata_stage['stage_names'][self.stage] == "guided" and self.trialnumber == 3:
            stage = self.find_stage('choice')
        elif self.metadata_stage['stage_names'][self.stage] in ["choice", "guided"]:
            stage = self.stage
        else:
            stage = self.stage + 1
        return stage, is_error

    def _step_after(self, action):
        R_ext = 0
        R_int = 0
        if self.metadata_stage['stage_names'][self.last_stage] in ["choice", "guided"]:
            self.number_of_trials_left -= 1
            self.trialnumber += 1
        return R_ext, R_int

    def _draw_obs(self):
        if self.metadata_stage['stage_names'][self.stage] == "fixation":
            self.draw("fixation",1)
        else:
            if self.metadata_stage['stage_names'][self.stage] == "guided":
                tcue = W.W_onehot(self.param_trial['guided'][self.trialnumber], 2)
                self.draw("cue", tcue)
            elif self.metadata_stage['stage_names'][self.stage] == "choice":
                self.draw("cue", [1,1])
            horizon = np.zeros(10)
            horizon[0:(self.number_of_trials_left)] = 1
            self.draw("horizon", horizon)
        self.flip()

    def _render_frame_obs_format(self, obs, lst):
        c = 0
        for i, j in lst.items():
            if i != "fixation":
                obs[c] = obs[c] * 128 + np.any(obs[c] > 0) * 127
            c += 1
        return obs
