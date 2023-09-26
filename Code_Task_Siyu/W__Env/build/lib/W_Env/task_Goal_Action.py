from W_Gym.W_Gym_Grid2D import W_Gym_grid2D
from gym import spaces
import numpy as np

class task_Goal_Action(W_Gym_grid2D):
    p_reward = None
    def __init__(self, p_reward = 1, n_maxTrialsPerBlock = 30, **kwarg):
        super().__init__(3,3,6, n_maxTrialsPerBlock = n_maxTrialsPerBlock, **kwarg)
        self.p_reward = p_reward
        # set action space
        self.action_space = spaces.Discrete(5) # stay, left, up, right, down
        # set rendering dimension names
        self.setup_obs_Name2DimNumber({'fixation':0, \
                                       'square':1, 'imageA':2, 'imageB':3, \
                                        "dot1":4, "dot2":5})
        # set stages
        stage_names = ["fix0", "flash1", "post1", \
                       "flash2", "post2", "choice1", "hold1", \
                        "choice2", "hold2"]
        stage_advanceuponaction = ["choice1", "choice2"]
        self.setW_stage(stage_names = stage_names, stage_advanceuponaction = stage_advanceuponaction)
        # task constants 
        self.cccs = np.array([6,8,2,0])
        self.ccs = np.array([3,7,5,1])
    
    def _setup_render(self):
        plottypes = ["circle", "square", "square", "square", "square", "square"]
        colors = [(255,255,255), (255,255,255), (0, 255, 0), (0, 0, 255), (255,255,255), (255,255,255)]
        radius = [0.02, 0.04, 0.08, 0.08, 0.04, 0.08]
        self._render_setplotparams('obs', plottypes, colors, radius)
        plottypes = ["circle"]
        colors = [(255,0,0)]
        radius = [0.01]
        self._render_setplotparams('action', plottypes, colors, radius)
        
    def _reset_block(self):
        self.oracle = np.random.randint(2)
    
    def _reset_trial(self):
        self.gaze.set_pos(1,1)
        cccs = self.cccs 
        ccs = self.ccs 
        ccc_id = np.random.choice(4,2, replace = False)
        cc_id = np.array([0,2]) + np.random.choice(2,1)
        param = {'ccc_id': ccc_id, 'ccc_pos': cccs[ccc_id], 'cc_id':cc_id, 'cc_pos':ccs[cc_id]}
        self.param_trial = param
        self.pos_choice1 = None
        self.pos_choice2 = None
    
    def get_valid_option3(self, action):
        action = np.where(action == self.ccs)[0][0]
        te = np.array([action, action-1])
        te[te == -1] = 3
        cccs = self.cccs
        return cccs[te].tolist()

    def _step(self, action):
        R_ext = 0
        R_int = 0
        # register pos choice 1 and choice 2
        if self.metadata_stage['stage_names'][self.stage] == "choice1" and \
            self.pos_choice1 is None:
            self.pos_choice1 = action
        if self.metadata_stage['stage_names'][self.stage] == "choice2" and \
            self.pos_choice2 is None:
            self.pos_choice2 = action      
        return R_ext, R_int
    
    def _step_after(self, action):
        R_ext = 0
        R_int = 0
        # get reward
        if not self.trial_is_error and self.metadata_stage['stage_names'][self.stage] == "ITI":
            if self.pos_choice2 == self.param_trial['ccc_pos'][self.oracle]: # correct choice
                treward = self.probabilistic_reward(self.p_reward)
            else:
                treward = self.probabilistic_reward(1 - self.p_reward)
            R_ext += treward     
        return R_ext, R_int
    
    def _step_set_validactions(self):
        if self.metadata_stage['stage_names'][self.stage] in ["fix0","post1", "post2","flash1","flash2"]:
            self.valid_actions = 4
        elif self.metadata_stage['stage_names'][self.stage] == "choice1":
            self.valid_actions = [1,3,5,7]
        elif self.metadata_stage['stage_names'][self.stage] == "choice2":
            self.valid_actions = self.get_valid_option3(self.pos_choice1)
        elif self.metadata_stage['stage_names'][self.stage] in ["hold1"]:
            self.valid_actions = np.intersect1d(self.pos_choice1, self.param_trial['cc_pos'])
        elif self.metadata_stage['stage_names'][self.stage] in ["hold2"]:
            self.valid_actions = np.intersect1d(self.pos_choice2, self.param_trial['ccc_pos'])
    
    def _draw_obs(self):
        # calculate new observation
        if self.metadata_stage['stage_names'][self.stage] in ["fix0","post1", "post2"]:
            self.draw((1,1),"fixation")
        elif self.metadata_stage['stage_names'][self.stage] in ["flash1"]:
            self.draw((1,1),"fixation")
            tpos = self.param_trial['ccc_pos']
            self.draw(self.vec2mat(tpos[0]), "imageA")
            self.draw(self.vec2mat(tpos[1]), "imageB")
        elif self.metadata_stage['stage_names'][self.stage] in ["flash2"]:
            self.draw((1,1),"fixation")
            tpos = self.param_trial['cc_pos']
            self.draw(self.vec2mat(tpos[0]), "square")
            self.draw(self.vec2mat(tpos[1]), "square")
        elif self.metadata_stage['stage_names'][self.stage] == "choice1":
            self.draw((1,0),"dot1")
            self.draw((2,1),"dot1")
            self.draw((0,1),"dot1")
            self.draw((1,2),"dot1")
        elif self.metadata_stage['stage_names'][self.stage] in ["hold1"]: 
            (gx, gy) = self.vec2mat(self.pos_choice1)
            self.draw((gx, gy), "square")
        elif self.metadata_stage['stage_names'][self.stage] == "choice2": 
            for tp in self.valid_actions:
                (x,y) = self.vec2mat(tp)
                self.draw((x,y), "dot2")
        elif self.metadata_stage['stage_names'][self.stage] in ["hold2"]: 
            (gx, gy) = self.vec2mat(self.pos_choice2)
            if self.pos_choice2 == self.param_trial['ccc_pos'][0]:
                self.draw((gx, gy), "imageA")
            elif self.pos_choice2 == self.param_trial['ccc_pos'][1]:
                self.draw((gx, gy), "imageB")
        self.flip()
