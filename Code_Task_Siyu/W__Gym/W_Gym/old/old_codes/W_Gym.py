from W_Python import W_tools as W
import gym
import numpy as np
from collections import namedtuple 
import pandas


class W_Gym_render(gym.Env):
    dt = None
    obs_Name2DimNumber = None
    plot_params = dict()
    currentscreen = None
    window = None
    clock = None
    font = None
    canvas = None
    text_T = None
    text_R = None
    image_lst = []
    metadata_render = {'render_mode': None, 'window_size': [512, 512], 'render_fps': None}
    def __init__(self, render_mode = None, window_size = [512, 512], \
                 render_fps = None, dt = 1,\
                 **kwarg):
        self.dt = dt
        metadata_render = W.W_dict_kwargs()
        W.W_dict_updateonly(self.metadata_render, metadata_render)
        if self.metadata_render['render_fps'] is None:
            self.metadata_render['render_fps'] = 1/self.dt
        self.metadata_render['window_size'] = np.array(self.metadata_render['window_size'])
        self.setup_rendermode()
        if hasattr(self, '_setup_render'):
            self._setup_render()
        # print(f"render mode: {self.metadata_render['render_mode']}")

    # draw obs
    def draw(self, channelname, val):
        if channelname == "ITI":
            return
        idx = self.obs_Name2DimNumber[channelname]
        if self.currentscreen is None:
            self.currentscreen = np.zeros(self.observation_space_size())
        self.currentscreen[idx] = val 

    def observation_space_size(self): # deal with inconsistencies in gym output
        if self.observation_space.shape == ():
            return self.observation_space.n
        else:
            return self.observation_space.shape

    def _render_frame_1D(self, canvas, data, dictname):
        import pygame
        params = self.plot_params[dictname]
        n_channel = len(data) 
        for ci in range(n_channel):
            tcol = params['colors'][ci]
            tplottype = params['plottypes'][ci]
            tradius = params['radius'][ci]
            tpos = None
            if params['position'] is not None:
                tpos = params['position'][ci]
            if tpos is None:
                tpos = np.array(self.window.get_size()) * [0.5,0.5]
            tval = data[ci]
            if np.any(tval > 0): # show
                canvas = self._render_draw(canvas, tplottype, tcol, tpos, tradius, tval)
        return canvas
    
    def _render_draw(self, canvas, tplottype = None, tcol = None, tpos = None, tradius = None, tval = None):
        import pygame
        if tplottype == "circle":
            pygame.draw.circle(canvas, tcol, tpos, np.mean(tradius))
        elif tplottype == "square":
            pygame.draw.rect(canvas, tcol, \
                np.concatenate((-tradius + tpos, tradius * 2), axis = None), 0)
        elif tplottype == "image":
            self._render_array(tval, tpos, tradius)
        return canvas
    
    def _render_frame_action(self, canvas, *arg, **kwarg):
        if self.plot_params['action']['plottypes'] == ['binary']:
            canvas = self._render_frame_binarychoice(canvas, np.array(self.plot_params['action']['plotparams']) == self.last_action)
        elif self.plot_params['action']['plottypes'] == ["arrows"]:
            canvas = self._render_frame_arrowchoice(canvas, np.array(self.plot_params['action']['plotparams']) == self.last_action)
        elif self.plot_params['action']['plottypes'] == ["arrowsplus"]:
            canvas = self._render_frame_arrowchoice(canvas, np.array(self.plot_params['action']['plotparams']) == self.last_action)
        else:
            canvas = self._render_frame_1D(canvas, W.enlist(self.last_action), 'action')
        return canvas
    
    def _render_frame_binarychoice(self, canvas, action):
        tradius = np.array(self.window.get_size()) * [0.05, 0.05]
        if action[0]:
            self._render_draw(canvas, 'square', (255,0,0), np.array(self.window.get_size()) * [0.1,0.5], tradius)
        if action[1]:
            self._render_draw(canvas, 'square', (255,0,0), np.array(self.window.get_size()) * [0.9,0.5], tradius)
        return canvas

    def _render_frame_arrowchoice(self, canvas, action):
        tradius = np.array(self.window.get_size()) * [0.05, 0.05]
        if action[0]:
            self._render_draw(canvas, 'square', (255,0,0), np.array(self.window.get_size()) * [0.1,0.5], tradius)
        if action[2]:
            self._render_draw(canvas, 'square', (255,0,0), np.array(self.window.get_size()) * [0.9,0.5], tradius)
        if action[1]:
            self._render_draw(canvas, 'square', (255,0,0), np.array(self.window.get_size()) * [0.5,0.1], tradius)
        if action[3]:
            self._render_draw(canvas, 'square', (255,0,0), np.array(self.window.get_size()) * [0.5,0.9], tradius)
        if len(action) > 4 and action[4]:
            self._render_draw(canvas, 'square', (255,0,0), np.array(self.window.get_size()) * [0.5,0.5], tradius)    
        return canvas
    
    def _render_frame_obs(self, canvas, *arg, **kwarg):
        obs = list()
        lst = self.obs_Name2DimNumber
        for i, j in lst.items():
            obs.append(self.obs[np.array(j)])
        if hasattr(self, '_render_frame_obs_format'):
            obs = self._render_frame_obs_format(obs, lst)
        canvas = self._render_frame_1D(canvas, obs, 'obs')
        return canvas

    def flip(self, is_clear = True):
        self.obs = self.currentscreen
        if is_clear:
            self.blankscreen()

    def blankscreen(self):
        assert hasattr(self, 'observation_space')
        self.currentscreen = np.zeros(self.observation_space_size())

    def setup_rendermode(self, render_mode = None):
        if render_mode is None:
            render_mode = self.metadata_render['render_mode']
        else:
            self.metadata_render['render_mode'] = render_mode
        if render_mode == "human":
            import pygame  # import here to avoid pygame dependency with no render
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(self.metadata_render['window_size'])
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font('freesansbold.ttf', 32)
            self.set_text('text_T', f"t = {0}")

    def setup_obs_Name2DimNumber(self, mydict):
        self.obs_Name2DimNumber = mydict

    def _render_setplotparams(self, dictname, plottypes = None, colors = None, radius = None, position = None, plotparams = None):
        params = W.W_dict_kwargs()
        del params['dictname']
        if dictname == "action" and position is None:
            position = np.array([0.1, 0.1])
            position = np.repeat(position[np.newaxis,:], len(params['plottypes']), axis = 0)
        for i in range(len(params['plottypes'])):
            if params['radius'] is not None:
                params['radius'][i] = params['radius'][i] * self.metadata_render['window_size']
            if params['position'] is not None and params['position'][i] is not None:
                params['position'][i] = params['position'][i] * self.metadata_render['window_size']
        self.plot_params.update({dictname: params})
        
    def set_text(self, attr, str):
        black = (0, 0, 0)
        red = (255, 0, 0)
        setattr(self, attr, self.font.render(str, True, red, black))


    def _render_array(self, z, pos, tradius = [1,1]):
        tradius = np.array(tradius)
        if len(z.shape) < 2:
            z = z.reshape(z.shape.__add__((1,)))
        if len(z.shape) == 2:
            z = np.stack([z,z,z], axis = 2)
        rx = np.ceil(tradius[0]/z.shape[0])
        ry = np.ceil(tradius[1]/z.shape[1])
        r = np.min((rx, ry))
        if r > 1:
            r = r.astype('int')
            z = np.kron(z, np.ones((r,r,1)))
        import pygame
        surf = pygame.surfarray.make_surface(z)
        image = {'image':surf, 'pos':pos - np.array(surf.get_size())/2}
        self.image_lst.append(image)

    def _render_frame_update(self):
        canvas = self.canvas
        if self.metadata_render['render_mode'] == "human":
            import pygame
            assert self.window is not None
            self.window.blit(canvas, canvas.get_rect())
            if self.image_lst != []:
                for x in self.image_lst:
                    self.window.blit(x['image'], x['pos'])
                self.image_lst = []
            if self.text_T is not None:
                trect = self.text_T.get_rect()
                wsize = pygame.display.get_window_size()
                trect.center = tuple(map(lambda i, j: i-j, wsize, trect.center))
                self.window.blit(self.text_T, trect)
            if self.text_R is not None:
                self.window.blit(self.text_R, self.text_R.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata_render["render_fps"])
        else:  # rgb_array or single_rgb_array
            import numpy as np
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
    
    def _render_frame_create(self):
        import pygame 
        canvas = pygame.Surface(self.metadata_render['window_size'])
        canvas.fill((0, 0, 0))
        return canvas
    
    def close(self):
        if self.window is not None:
            import pygame 
            pygame.display.quit()
            pygame.quit()

    def render(self, option = None, *arg, **kwarg):
        if self.metadata_render['render_mode'] is None:
            return
        self.canvas = self._render_frame_create()
        option = W.enlist(option)
        for x in option:
            self.render_frame(x, *arg, **kwarg)
        self._render_frame_update()

    def render_frame(self, option = None, *arg, **kwarg):
        canvas = self.canvas
        if option is None:
            assert hasattr(self, '_render_frame')
            self.canvas = self._render_frame(canvas, *arg, **kwarg)    
        elif option == "action":
            self.canvas = self._render_frame_action(canvas, *arg, **kwarg)
        elif option == "obs":
            self.canvas = self._render_frame_obs(canvas, *arg, **kwarg)
        elif option == "reward":
            self._render_frame_reward(*arg, **kwarg)
        elif option == "time":
            self._render_frame_time(*arg, **kwarg)
        else:
            assert hasattr(self, '_render_frame')
            self.canvas = self._render_frame(canvas, option, *arg, **kwarg)

    def _render_frame_reward(self, *arg, **kwarg):
        if hasattr(self, 'last_reward'):
            self.set_text('text_R', f"R = {self.last_reward}")

    def _render_frame_time(self, *arg, **kwarg):
        self.set_text('text_T', f"game#{self.tot_trials}, { self.metadata_stage['stage_names'][self.stage] }, t = {self.t}")
        
    def get_window_relative2absolute(self, pos):
        import numpy as np
        pos = np.array(pos)
        return pos * self.metadata_render['window_size']

class W_Gym(W_Gym_render):
    Rewards = {"R_advance":0, "R_error": -1, "R_reward": 1}
    obs = None
    obs_augment = None
    is_augment_obs = False
    is_faltten_obs = True
    info = None
    n_maxTrials = None
    tot_t = 0
    t = 0 # total time from beginning of a trial
    timer = 0 # timer
    stage = 0 # task stage
    tot_trials = 0 # total trials from beginning
    trial_counter = 0 # can be reset
    metadata_episode = {"n_maxTrialsPerBlock": np.Inf, "n_maxBlocks": np.Inf}
    metadata_stage = {'stage_names':None, 'stage_timings': None, 'stage_advanceuponaction': None}
    last_reward = 0
    last_action = None
    last_stage = None
    is_oracle = False
    # action_immediateadvance = None
    def __init__(self, n_maxT = np.Inf, n_maxTrials = np.Inf, \
                    n_maxTrialsPerBlock = np.Inf, n_maxBlocks = np.Inf, \
                    is_augment_obs = True, is_faltten_obs = True, \
                    is_ITI = True, is_oracle = False, **kwarg):
        super().__init__(**kwarg)
        self.n_maxT = n_maxT
        self.n_maxTrials = n_maxTrials
        self.is_augment_obs = is_augment_obs
        self.is_faltten_obs = is_faltten_obs or self.is_augment_obs
        self.is_ITI = is_ITI
        self.valid_actions = None
        self.effective_actions = None

        self.info_task = None
        self.info_block = None
        self.info_trial = None        

        self.info = {'info_task':[], 'info_block':[], 'info_trial':[], 'info_step': []}
        metadata = W.W_dict_kwargs()
        W.W_dict_updateonly(self.metadata_episode, metadata)
        self.setW_stage(["stages"], [np.Inf])

    def _len_observation(self):
        len = self.observation_space_size()
        if self.is_augment_obs:
            len += self._len_actions() + 1
        return len
    
    def _len_actions(self):
        return self.action_space.n
    
    # flow
    def reset(self, return_info = False):
        self.info['info_task'] = None
        self.tot_trials = 0
        self.tot_blocks = 0
        self.tot_t = 0
        self.last_action = None
        self.last_reward = 0
        if hasattr(self, '_reset'):
            self._reset()
        self.reset_block()
        self.reset_trial()
        if hasattr(self, '_step_set_validactions'):
            self._step_set_validactions()
        if hasattr(self, '_draw_obs'):
            self._draw_obs()
        self.render(option = ['obs', 'time'])

        self.task_info()
        obs = self._get_obs()
        self.info_step = {'obs':self._get_obs(False)}
        info = self._get_info()
        return obs if not return_info else (obs, info)

    def reset_block(self):
        self.tot_blocks += 1
        self.trial_counter = 0
        if hasattr(self, '_reset_block'):
            self._reset_block()
        self.block_info()

    def reset_trial(self):
        self.param_trial = None
        self.t = 0
        self.timer = 0
        self.stage = 0
        self.trial_is_error = False
        last_trial = self.trial_counter
        if hasattr(self, '_reset_trial'):
            self._reset_trial()
        self.trial_info(last_trial)
    
    def step_info(self):
        pass

    def _step_info(self, obs, action, reward, is_done, tot_t):
        self.info_step.update({'action':action, 'reward':reward, 'tot_t': tot_t, 'is_done':is_done, 'obs_next': obs})
        self.info_step.update({'blockID': self.tot_blocks, 'trialID': self.trial_counter, 't':self.t, 'stage': self.stage})
        self.step_info()
        self.info['info_step'].append(self.info_step)
        self.info_step = {'obs':obs}

    def block_info(self):
        self.info_block = {'blockID': self.tot_blocks}
        if hasattr(self, '_block_info'):
            self._block_info()
        self.info['info_block'].append(self.info_block)

    def trial_info(self, last_trial):
        self.info_trial = {'blockID': self.tot_blocks, 'trialID': last_trial}
        if hasattr(self, '_trial_info'):
            self._trial_info()
        if hasattr(self, '_get_oracle_trial'):
            self.info_trial['oracle'] = self._get_oracle_trial()
        self.info['info_trial'].append(self.info_trial)

    def task_info(self):
        self.info['info_task'] = self.info_task

    def get_oracle_action(self):
        return self.action_oracle.pop(0)
    
    def step(self, action):
        reward_E = 0
        reward_I = 0
        # advance time
        self.tot_t += self.dt
        self.t = self.t + self.dt
        self.timer = self.timer + self.dt
        if hasattr(self, "_action_transform"):
            action = self._action_transform(action)
        # check valid actions
        is_error = not (self.check_isvalidaction(action, self.valid_actions) or \
            self.metadata_stage['stage_names'][self.stage] == "ITI")
        self.is_effective_action = self.check_isvalidaction(action, self.effective_actions)
        # record current choice
        self.last_action = action
        self.last_stage = self.stage
        # get consequences of actions
        if not is_error and hasattr(self, '_step'):    
            tR_ext, tR_int = self._step(action)
            reward_E += tR_ext
            reward_I += tR_int       
        

        last_t = self.tot_t



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
        if hasattr(self, '_get_oracle_step'):
            self._get_oracle_step()
        self._step_info(self._get_obs(False), action, self.last_reward, is_done, last_t)
        info = self._get_info()
        return obs, self.last_reward, is_done, self.tot_t, info

    def find_stage(self, stagename):
        return np.where([j == stagename for j in self.metadata_stage['stage_names']])[0][0]

    def advance_stage(self, is_error = False):
        is_done = False
        R_ext = 0
        R_int = 0
        is_nexttrial = 0
        if not is_error:
            self.stage, is_error = self._advance_stage()
            if not is_error:
                R_int = self.Rewards['R_advance']
                self.timer = 0 # auto reset timer 
                if self.stage == len(self.metadata_stage['stage_names']):
                    is_nexttrial = 1
        
        if is_error:
            self.trial_is_error = True
            R_int = self.Rewards['R_error']
            if "ITI" in self.metadata_stage['stage_names']:
                self.stage = self.find_stage('ITI')
                self.timer = 0 # auto reset timer 
            else:
                is_nexttrial = 1
        
        if is_nexttrial == 1:
            self.tot_trials += 1
            self.trial_counter += 1
            if self.tot_trials >= self.n_maxTrials:
                is_done = True
            if self.trial_counter >= self.metadata_episode['n_maxTrialsPerBlock']:
                self.reset_block()
            self.reset_trial()   

        return R_ext, R_int, is_done

    def _advance_stage(self):
        is_error = False
        return self.stage + 1, is_error

    # rewards
    def setW_reward(self, **kwarg):
        W.W_dict_updateonly(self.Rewards, kwarg)

    def setW_stage(self, stage_names, stage_timings_user = None, \
                   stage_advanceuponaction = None):
        if self.is_ITI and not "ITI" in stage_names:
            stage_names.append("ITI")
        self.metadata_stage['stage_names'] = stage_names
        nstage = len(self.metadata_stage['stage_names'])
        stage_timings = np.ones(nstage) * self.dt           
        if stage_timings_user is not None:
            stage_timings[np.arange(0, len(stage_timings_user))] = stage_timings_user
        # if stage_advanceuponaction is None:
        #     stage_advanceuponaction = "ITI"
        # elif not "ITI" in stage_advanceuponaction and "ITI" in self.metadata_stage['stage_names']:
        #     stage_advanceuponaction.append("ITI")

        c = np.zeros(nstage)
        if stage_advanceuponaction is not None:
            tid = np.array([np.where([j == i for j in self.metadata_stage['stage_names']]) for i in iter(stage_advanceuponaction)]).squeeze()
            c[tid] = 1
        stage_advanceuponaction = c
        
        self.metadata_stage['stage_timings'] = stage_timings
        self.metadata_stage['stage_advanceuponaction'] = stage_advanceuponaction

    def probabilistic_reward(self, p):
        if np.random.uniform() <= p:
            reward = self.Rewards['R_reward']
        else:
            reward = 0
        return reward
    # action
    def check_isvalidaction(self, action, valid_actions = None):
        if valid_actions is None:
            return True
        else:
            valid_actions = W.enlist(valid_actions)
            is_valid = action in valid_actions
            return is_valid

    # get obs
    def _get_obs(self, is_augment = None):
        if self.is_faltten_obs or self.is_augment_obs:
            obs = self.obs.flatten()
        else:
            obs = self.obs
        if (is_augment is None and self.is_augment_obs) or is_augment:
            action = self._action_flattened()
            r = np.array(self.last_reward)
            r = r.reshape((1,))
            obs = np.concatenate((obs, r, action))
        return obs
    
    def _get_info(self):
        return self.info

    def _action_flattened(self):
        if not hasattr(self, '_action_transform'):
            action = np.zeros(self.action_space.n)
        else:
            assert hasattr(self, '_action_dimension')
            action = np.zeros(self._action_dimension)
        if self.last_action is not None:
            action[self.last_action] = 1
        return action
    
    def format4save(self):
        info = self._get_info()
        d1 = self.info2pandas(info['info_step'])
        d2 = self.info2pandas(info['info_trial'])
        d2 = pandas.concat([d2.drop(columns = 'params'), self.info2pandas(list(d2.params))], axis = 1)
        d3 = self.info2pandas(info['info_block'])
        d3 = pandas.concat([d3.drop(columns = 'params'), self.info2pandas(list(d3.params))], axis = 1)

        d23 = pandas.merge(d2, d3, on  = "blockID")

        data = pandas.merge(d1, d23, on = ["blockID", "trialID"])
        return data
    
    def info2pandas(self, tinfo):
        namestep = tuple(tinfo[0].keys())
        steptp = namedtuple('step', namestep)
        step = [steptp(*v.values()) for v in tinfo]
        step = steptp(*zip(*step))
        step = [list(np.stack(x)) for x in step]
        step = {k:v for k,v in zip(namestep, step)}
        data = pandas.DataFrame.from_dict({k:step[k] for k in list(set(step.keys()))})
        return data