from gym import spaces
from W_Gym.W_Gym import W_Gym
from W_Python import W_tools as W
import numpy as np

class task_TwoStep_2frame_full(W_Gym):
    task_param = {'p_switch': 0.025, 'reward_safe': 0.1}
    high_state = None
    is_fixed = False
    def __init__(self, is_fixed = [0, 0], is_flip_ptrans = False, is_flip = True, is_guess = False,  *arg, **kwarg):
        super().__init__(is_ITI = False, *arg, **kwarg)

        self.observation_space = spaces.Discrete(9)
        # set action space
        self.action_space = spaces.Discrete(5) # shuttle1,2, planet 1,2, takereward
        self.is_flip_ptrans = is_flip_ptrans
        self.is_fixed = is_fixed
        self.is_flip = is_flip
        self.is_guess = is_guess
        # set rendering dimension names
        self.setup_obs_Name2DimNumber({'planet0':0, 'planet1':1, 'planet2':2,\
                                       'shuttle1':3, 'shuttle2':4, \
                                       'displayreward':5 ,\
                                       'question_shuttle':6, 'question_planet':7, 'question_reward':8,\
                                       })
        # set stages
        stage_names = ["stage1", "stage2"]
        stage_advanceuponaction = ["stage1", "stage2"]
        self.setW_stage(stage_names = stage_names, stage_advanceuponaction = stage_advanceuponaction)

        self.display_reward = 0
        
        print(f"fix:{self.is_fixed}, flip:{self.is_flip_ptrans}")

    

    def _reset_block(self):
        if self.is_fixed[0] == 1:
            p = 0.9 
        elif self.is_fixed[0] == 2:
            p = np.random.choice(5,1)[0]/10 + 0.6
        else:
            p = np.random.rand()
            p = np.max((p, 1-p))
        if self.is_flip_ptrans and np.random.rand() < 0.5:
            p = 1 - p
        self.task_param['p_trans'] = [p,p]
        self.high_state = np.random.choice(2,1)[0]
        if self.is_fixed[1] == 1:
            p = 0.9
        elif self.is_fixed[1] == 2:
            p = np.random.choice(5,1)[0]/10 + 0.6
        else:
            p = np.random.rand()
            p = np.max((p, 1-p))
            
        # print(f"phigh:{p}")
        self.task_param['p_reward_high'] = p
        self.task_param['p_reward_low'] = 1-p

    def _reset_trial(self):
        if self.is_flip and np.random.rand() < self.task_param['p_switch']: # flip reward
            self.high_state = 1- self.high_state
        if self.is_flip and np.random.rand() < self.task_param['p_switch']: # flip transition
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
        param = {'transition':trans.astype(int), 'reward':r}
        self.param_trial = param
        self.planet = None

    def _step_set_validactions(self):
        if self.metadata_stage['stage_names'][self.stage] in ["stage1"]:
            self.valid_actions = [0,1]
        elif self.metadata_stage['stage_names'][self.stage] in ["stage2"]:
            self.valid_actions = [2]
    
    def _step(self, action, guess = -1):
            R_ext = 0
            R_int = 0

            if self.metadata_stage['stage_names'][self.stage] == "stage1":
                self.shuttle = action
                self.planet = self.param_trial['transition'][self.shuttle]
            # if self.metadata_stage['stage_names'][self.stage] == "guessplanet":
                if self.is_guess:
                    if self.planet == guess:
                        R_int += 1
                    else:
                        R_int -= 0
                    # R_int += np.array(self.planet == guess).astype(int)
            if self.metadata_stage['stage_names'][self.stage] == "stage2":
                self.display_reward = self.param_trial['reward'][self.planet]
                R_ext += self.display_reward
            return R_ext, R_int

    def _draw_obs(self):
        if self.display_reward > 0: # display reward from previous trial
            self.draw('displayreward', self.display_reward)
            self.display_reward = 0
        if self.metadata_stage['stage_names'][self.stage] == "stage1":
            self.draw('planet0', 1)
            self.draw('question_shuttle', 1)
        elif self.metadata_stage['stage_names'][self.stage] == "guessplanet":
            self.draw('planet0', 1)
            self.draw('question_planet',1)
            if self.shuttle == 0:
                self.draw('shuttle1',1)
            elif self.shuttle == 1:
                self.draw('shuttle2',1)
        elif self.metadata_stage['stage_names'][self.stage] == "stage2":
            if self.planet == 0:
                self.draw('planet1',1)
            elif self.planet == 1:
                self.draw('planet2',1)
            self.draw('question_reward', 1)
        self.flip()

    def _setup_render(self):
        plottypes = ["circle", "circle", "circle", "circle", "circle", "circle", "square", "square", "square"]
        colors = [(100,100,100), (0,255,0), (0,0,255), (0,125,0), (0,0,125), (255,0,0), (255,255,255), (255,255,255), (255,255,255)]
        radius = [0.25, 0.25, 0.25, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05]
        position = [None, None, None, [0.3,0.5], [0.7, 0.5], [0.5, 0.3], [0.2,0.8],[0.5,0.8],[0.8,0.8]]
        self._render_setplotparams('obs', plottypes, colors, radius, position)
        plottypes = ["arrowsplus"]
        self._render_setplotparams('action', plottypes, plotparams = [0,2,1,3,4])


    def step(self, action, guess= -1):
        reward_E = 0
        reward_I = 0
        # advance time
        self.tot_t += self.dt
        self.t = self.t + self.dt
        self.timer = self.timer + self.dt
        if hasattr(self, "_action_transform"):
            action = self._action_transform(action)
        # check valid actions
        # guess = action[1].item()
        # action = action[0].item()
        is_error = not (self.check_isvalidaction(action, self.valid_actions) or \
            self.metadata_stage['stage_names'][self.stage] == "ITI")
        self.is_effective_action = self.check_isvalidaction(action, self.effective_actions)
        # record current choice
        self.last_action = action
        self.last_stage = self.stage
        # get consequences of actions
        if not is_error and hasattr(self, '_step'):    
            tR_ext, tR_int = self._step(action, guess)
            reward_E += tR_ext
            rguess = tR_int      
        
        # move on to the next time point
        # check is advance stage
        is_advance = False
        # self.action_immediateadvance = None
        if not is_error:
            if self.metadata_stage['stage_advanceuponaction'][self.stage] == 1:
                if self.effective_actions is None:
                    effective_actions = self.valid_actions
                else:
                    effective_actions = self.effective_actions
                if self.check_isvalidaction(action, effective_actions):
                    is_advance = True
                    # self.action_immediateadvance = action
            if self.timer >= self.metadata_stage['stage_timings'][self.stage]:
                is_advance = True
        # move on to the next stage  
        if is_error or is_advance:
            tR_ext, tR_int, is_done = self.advance_stage(is_error)
            reward_E += tR_ext
            reward_I += tR_int
        else: 
            is_done = False

        if self.tot_t >= self.n_maxT:
            is_done = True
        # get consequences of actions (after)
        if not is_error and hasattr(self, '_step_after'):    
            tR_ext, tR_int = self._step_after(action)
            reward_E += tR_ext
            reward_I += tR_int

        # set valid actions for the new observation
        if hasattr(self, '_step_set_validactions'):
            self._step_set_validactions()
        if hasattr(self, '_draw_obs'):
            self._draw_obs()

        self.last_reward = reward_E + reward_I
        self.render(option = ["obs","action","reward","time"])
        
        obs = self._get_obs()
        info = self._get_info()
        outr = np.stack((self.last_reward, rguess))
        return obs, outr, is_done, self.tot_t, info
    

   
    
