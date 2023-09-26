from W_Python.W import W
import numpy as np
import pandas as pd
# import gym # try without gym

class W_Gym_task():
    """
    It implements a class of episodic task consists of blocks, trials, states and steps
    """
    # meta parameters
    # gym variables 
    _metadata_gym = {"n_maxTime": np.Inf, "n_maxTrials": np.Inf, "n_maxBlocks": np.Inf, \
                    "block_n_maxTrials": np.Inf, 'trial_n_maxTime': np.Inf,  \
                    "option_obs_augment": ["action", "reward"], \
                    "option_obs_is_flatten": True, "is_ITI": True}
    # state variables
    _metadata_state = {'n_state ': 0, 'statenames': None, 'timelimits': None, \
                    'is_immediate_advance': None, 'is_terminating_state': None, \
                    'matrix_transition': None} 
    # reward setting
    _param_rewards = {"R_error": -1, "R_reward": 1}
    # parameters for task, block and trial
    _param_task = {}
    _param_block = {}
    _param_trial = {}
    # timing variables (unit ms)
    _time_unit = 1 # ms
    _time_task = 0 # time since task start 
    _time_block = 0 # time since block start
    _time_trial = 0 # time since trial start
    _time_state = 0 # time since state start
    # counting variables (number of steps)
    _count_block = 0 # number of blocks completed
    _count_block_trial = 0 # number of trials completed since block start
    _count_task_trial = 0 # number of trials completed since task start
    # gym variables
    _obs = None # observation
    _last_action = None
    _last_action_motor = None
    _last_reward = 0
    _env_vars = {} # environment moment-by-moment variables (saved)
    _env_vars_after = {} # environment variables after update (saved)
    _temp_vars = {} # temp moment-by-moment variables (not saved)
    # draw observation
    _next_screen = None # to draw next screen
    _obs_channel_namedict = None # input dimension vs name, for easy drawing
    # trial variables
    _trial_is_error = False
    # state variables
    _state = 0 # current state
    _state_valid_actions = None # valid actions for a state
    _state_effective_actions = None # effective actions for a state
    # data variable (recorded behavior)
    _data_issave = False
    _data_istable = True
    _data = None
    # names 
    env_name = 'env'
    dim_obs = None
    n_actions = None
    n_obs = None

    def __init__(self, n_maxTime = np.Inf, n_maxTrials = np.Inf, n_maxBlocks = np.Inf, \
                    block_n_maxTrials = np.Inf, option_obs_augment = ["reward", "action"], \
                    option_obs_is_flatten = True, is_ITI = True, dt = 1, is_save = False, is_save_table = True, \
                    param_task = None, param_metadata = None, *arg, **kwarg):
        self._param_task = W.W_dict_updateonly(self._param_task, kwarg)
        if param_task is not None:
            self._param_task = W.W_dict_updateonly(self._param_task, param_task)
        _param_inputs = W.W_dict_function_arguments()
        W.W_dict_updateonly(self._metadata_gym, _param_inputs)
        if param_metadata is not None:
            W.W_dict_updateonly(self._metadata_gym, param_metadata)
        if self._metadata_gym['option_obs_augment'] is not None:
            self._metadata_gym['option_obs_is_flatten'] = True
        self._time_unit = dt
        self._data_issave = is_save
        self._data_istable = is_save_table
        self._reset_task()

    def get_auto_savename(self):
        if hasattr(self, 'custom_savename'):
            tstr = self.custom_savename()
            savename = f"{self.env_name}_{tstr}"
        else:
            savename = f"{self.env_name}"
        return savename

    def setup_obs_dim(self, *arg):
        self.dim_obs = np.array(arg)
        self.n_obs = np.prod(self.dim_obs)

    def setup_action_dim(self, n_actions, n_motor = None):
        self.n_actions = n_actions
        if n_motor is None:
            n_motor = n_actions
        self.n_motor = n_motor

    def setup_obs_channel_namedict(self, mydict):
        self._obs_channel_namedict = mydict

    def setup_state_parameters(self, state_names, state_timelimits = None, \
                    state_immediateadvance = None, \
                    matrix_state_transition = "next", \
                    terminating_state = None):
        if self._metadata_gym['is_ITI'] and not "ITI" in state_names:
            state_names.append("ITI")
        self._metadata_state['statenames'] = state_names
        self._metadata_state['n_state'] = len(self._metadata_state['statenames'])
        # set time limits for each state
        if state_timelimits is None:
            state_timelimits = np.ones(self._metadata_state['n_state']) * np.Inf
        elif state_timelimits == "ones":
            state_timelimits = np.ones(self._metadata_state['n_state']) * self._time_unit
        self._metadata_state['timelimits'] = state_timelimits
        # set state flag for advancing upon effective actions
        c = np.zeros(self._metadata_state['n_state'])
        if state_immediateadvance is not None:
            tid = W.W_list_findidx(state_immediateadvance, self._metadata_state['statenames'])
            c[tid] = 1
        self._metadata_state['is_immediate_advance'] = c
        # set state flag for terminating state
        if terminating_state is None:
            if "ITI" in self._metadata_state['statenames']:
                terminating_state = "ITI"
            else:
                terminating_state = state_names[-1] # default to be last state
        terminating_state = W.W_enlist(terminating_state)
        c = np.zeros(self._metadata_state['n_state'])
        tid = W.W_list_findidx(terminating_state, self._metadata_state['statenames'])
        c[tid] = 1
        self._metadata_state['is_terminating_state'] = c
        # set state transition
        # if matrix_state_transition == "next":
        #     matrix_state_transition = np.zeros([self._metadata_state['n_state'], self._metadata_state['n_state']])
        #     for i in range(1, self._metadata_state['n_state']):
        #         matrix_state_transition[i-1, i] = 1
        self._metadata_state['matrix_transition'] = matrix_state_transition 

    def setup_reward(self, **kwarg):
        self._param_rewards.update(kwarg)

    def saveon(self):
        self._data_issave = True
        if self._data is None:
            self._data_initialize()

    def saveoff(self):
        self._data_issave = False

    def _data_initialize(self):
        self._data = None if self._data_istable else  {'task': None, 'block': None, 'trial': None, 'data': None}

    def reset(self, reset_task = True, clear_data = True):
        if reset_task: # reset everything 
            self._reset_task()
        elif clear_data: # reset data
            self._data_initialize()
        self._reset_block()
        # draw new observation
        if hasattr(self, 'draw_observation'):
            self.draw_observation()
        # render
        if hasattr(self, 'render'):
            self.render(option = ['obs', 'time'])
        return self._get_obs()

    def _reset_task(self):
        self._time_task = 0
        self._count_block = 0
        self._count_task_trial = 0
        if self._data_issave:
            self._data_initialize()
        if hasattr(self, 'custom_reset'):
            self.custom_reset()
        self._record('task', self._param_task) # _param_task is not reset

    def _reset_block(self):
        self._param_block = {} # reset _param_block
        self._count_block_trial = 0
        self._time_block = 0
        if hasattr(self, 'custom_reset_block'):
            self.custom_reset_block()
        self._record('block', self._param_block)
        self._reset_trial()

    def _reset_trial(self):
        self._param_trial = {} # reset _param_trial
        self._time_trial = 0
        self._trial_is_error = False
        if hasattr(self, 'custom_reset_trial'):
            self.custom_reset_trial()
        self._record('trial', self._param_trial)
        self._state = 0
        self._reset_state()
        
    def _reset_state(self):
        self._time_state = 0
        self._state_valid_actions = None
        self._state_effective_actions = None
        # set valid actions for the new state
        if hasattr(self, 'custom_step_set_validactions'):
            self.custom_step_set_validactions()

    def _advance_time(self):
        is_done = False
        self._time_state += self._time_unit
        self._time_trial += self._time_unit
        self._time_block += self._time_unit
        self._time_task += self._time_unit
        if self._time_task >= self._metadata_gym['n_maxTime']:
            is_done = True
        return is_done
    
    def _advance_trial(self):
        is_done = False
        # update number of completed trials
        self._count_task_trial += 1
        if self._count_task_trial >= self._metadata_gym['n_maxTrials']:
            return True # return if is_done = True
        # update number of completed trials within the block
        self._count_block_trial += 1
        if self._count_block_trial >= self._metadata_gym['block_n_maxTrials']:
            is_done = self._advance_block()
        if not is_done:
            self._reset_trial()
        return is_done
        
    def _advance_block(self):
        is_done = False
        self._count_block += 1
        if self._count_block >= self._metadata_gym['n_maxBlocks']:
            is_done = True
        if not is_done:
            self._reset_block()
        return is_done

    def _abort(self):
        is_done = False
        self._trial_is_error = True
        reward = self._param_rewards['R_error']
        if "ITI" in self._metadata_state['statenames']:
            self._go_to_state("ITI")
        else:
            is_done = self._advance_trial()
        return reward, is_done
    
    def _go_to_state(self, newstate):
        if isinstance(newstate, str):
            newstate = W.W_list_findidx(newstate, self._metadata_state['statenames'])
        self._state = newstate
        self._reset_state()

    def step(self, action_motor, is_record = True, render_options = ["obs","action","reward","time"]):
        reward = 0
        # advance time
        is_done = self._advance_time()
        tdata = {'time_task': self._time_task, 'time_trial': self._time_trial, \
                 'state': self._metadata_state['statenames'][self._state], \
                 'obs': self.format_obs_for_save(self._obs), \
                 'count_trial': self._count_block_trial, 'count_block': self._count_block}
        tdata.update(self._env_vars)
        if self._data_istable:
            tdata.update(self._param_trial)
            tdata.update(self._param_block)
            tdata.update(self._param_task)
        # transform actions
        # if len(action_motor) == 1:
        #     action_motor = int(action_motor)
        if hasattr(self, "transform_actions"):
            action = self.transform_actions(action_motor)
        else:
            action = action_motor
        # check valid actions
        is_error = not W.is_in_list(action, self._state_valid_actions)
        is_effective = W.is_in_list(action, self._state_effective_actions)
        # get consequences of actions (reward)
        if not is_error and hasattr(self, 'custom_step_reward'):
            treward = self.custom_step_reward(action)
            reward += treward
        # get consequences of actions (state transition)
        # determine if state-transition occurs
        is_transition = False
        if not is_error:
            if self._metadata_state['is_immediate_advance'][self._state] == 1 and is_effective:
                is_transition = True
            if self._time_state >= self._metadata_state['timelimits'][self._state]:
                is_transition = True
        # state transition
        if is_error:
            treward, t_is_done = self._abort()
        elif is_transition:
            treward, t_is_done = self._state_transition()
        else:
            treward = 0
            t_is_done = False
        reward += treward
        is_done = is_done or t_is_done
        # get consequences of actions (after possible state transition)
        if hasattr(self, 'custom_step_reward_newstate'):
            treward = self.custom_step_reward_newstate(action)
            reward += treward
        
        # recursive component: may take multiple steps (collapse some states)
        if self._metadata_state['timelimits'][self._state] == 0:
            _, treward, t_is_done = self.step(action_motor, is_record = False, \
                                              render_options = None)
            reward += treward
            is_done = is_done or t_is_done
        
        self._last_action_motor = action_motor
        self._last_action = action
        self._last_reward = reward
        tdata.update(self._env_vars_after)
        tdata.update({'action': action, 'is_error': is_error, 'reward': reward})
        # record current action
        if is_record:
            self._record('data', tdata)
        # draw new observation
        if hasattr(self, 'draw_observation'):
            self.draw_observation()
        # render
        obs_renderer = None
        if hasattr(self, 'render'):
            obs_renderer = self.render(option = render_options)
        if hasattr(self, 'metadata_render') and self.metadata_render['render_mode'] == "rgb_array":
            obs = obs_renderer
        else:
            obs = self._get_obs()
        return obs, reward, is_done, self._time_task

    def _state_transition(self):
        reward = 0
        is_done = False
        if self._metadata_state['is_terminating_state'][self._state]:
            is_done = self._advance_trial()
        else:
            if hasattr(self, 'custom_state_transition'):
                newstate = self.custom_state_transition()
            elif self._metadata_state['matrix_transition'] == "next":
                newstate = self._state + 1
            else:
                transprob = self._metadata_state['matrix_transition'][self._state]
                newstate = np.random.choice(np.arange(0, self._metadata_state['n_state']), 1, p=transprob)
            self._go_to_state(newstate)
        return reward, is_done
    
    def get_probabilistic_reward(self, p):
        if np.random.uniform() <= p:
            reward = self._param_rewards['R_reward']
        else:
            reward = 0
        return reward

    def _get_obs(self):
        obs = self._obs.flatten() if self._metadata_gym['option_obs_is_flatten'] else self._obs
        option_obs_augment = self._metadata_gym['option_obs_augment']
        if option_obs_augment is not None:
            for opt_name in iter(option_obs_augment):
                if opt_name == "action": 
                    tval = W.W_onehot(self._last_action, self.get_n_actions(is_motor=False))
                elif opt_name == "motor":
                    tval = W.W_onehot(self._last_action_motor, self.get_n_actions(is_motor=True))
                elif opt_name == "reward":
                    tval = np.array(self._last_reward)
                    tval = tval.reshape((1,))
                else:
                    tval = np.array(self._env_vars[opt_name])
                    tval = tval.reshape((1,))
                obs = np.concatenate((obs, tval))
        return obs
    
    def format_obs_for_save(self, obs):
        if hasattr(self, 'custom_format_obs_for_save'):
            obs = self.custom_format_obs_for_save(obs)
        return obs.flatten()
    
    def get_n_actions(self, is_motor = False):
        if is_motor:
            assert hasattr(self, 'n_motor')
            return self.n_motor
        else:
            assert hasattr(self, 'n_actions')
            return self.n_actions
        #    return self.action_space.n

    def get_n_obs(self, is_count_augmented_dimensions = True):
        #if self.observation_space.shape == ():
        #    len = self.observation_space.n
        #else:
        #    len = self.observation_space.shape
        
        assert hasattr(self, 'n_obs')
        len = self.n_obs
        
        if is_count_augmented_dimensions and self._metadata_gym['option_obs_augment'] is not None:
            for opt_name in iter(self._metadata_gym['option_obs_augment']):
                if opt_name == "action": 
                    len += self.get_n_actions(is_motor=False)
                elif opt_name == "motor":
                    len += self.get_n_actions(is_motor=True)
                elif opt_name == "reward":
                    len += 1
                else:
                    len += np.array(self._env_vars[opt_name]).size
        return len                
    
    def get_dim_obs(self):
        assert hasattr(self, 'dim_obs')
        return self.dim_obs

    def flip(self, is_clear = True):
        self._obs = self._next_screen
        if is_clear:
            self.blankscreen()

    def blankscreen(self):
        # assert hasattr(self, 'observation_space')
        self._next_screen = np.zeros(self.get_dim_obs())

    # draw obs
    def draw_onehot(self, channelname, val):
        if self._next_screen is None:
            self.blankscreen()
        if channelname == "ITI":
            return
        assert self._obs_channel_namedict is not None
        idx = self._obs_channel_namedict[channelname]
        self._next_screen[idx] = val
    
    def _record(self, datatype, datadict = None):
        if not self._data_issave:
            return
        if self._data_istable and not datatype == "data":
            return
        newdata = pd.DataFrame.from_dict(datadict, orient = "index").T
        if self._data_istable: # must have datatype == "data"
            df = self._data 
            if df is None:
                df = pd.DataFrame()
            self._data = pd.concat((df, newdata))
        else:        
            df = self._data[datatype] 
            if df is None:
                df = pd.DataFrame()
            self._data[datatype] = pd.concat((df, newdata))
    
