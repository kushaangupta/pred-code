from collections import namedtuple
import pandas


class W_Gym_render(gym.Env):
    dt = None
    obs_Name2DimNumber = None
    plot_params = dict()
    image_lst = []
  
        if hasattr(self, 'custom_setup_render'):
            self.custom_setup_render()
        # print(f"render mode: {self.metadata_render['render_mode']}")



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






class W_Gym(W_Gym_render):
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
    metadata_stage = {'stage_names':None, 'stage_timings': None, 'stage_advanceuponaction': None}
    last_reward = 0
    last_action = None
    last_stage = None
    is_oracle = False
    # action_immediateadvance = None
    def __init__(self, is_oracle = False):


        self.info = {'info_task':[], 'info_block':[], 'info_trial':[], 'info_step': []}
        self.setW_stage(["stages"], [np.Inf])

    def _len_observation(self):
        len = self.observation_space_size()
        if self.is_augment_obs:
            len += self._len_actions() + 1
        return len

    def _len_actions(self):
        return self.action_space.n

    # flow
    def reset(self):
        self.info['info_task'] = None

        self.last_action = None
        self.last_reward = 0


        self.task_info()
        self.info_step = {'obs':self._get_obs(False)}

    def reset_trial():
        last_trial = self.trial_counter


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
        # record current choice
        self.last_stage = self.stage


        last_t = self.tot_t






        if hasattr(self, '_get_oracle_step'):
            self._get_oracle_step()
        self._step_info(self._get_obs(False), action, self.last_reward, is_done, last_t)
        info = self._get_info()
        return obs, self.last_reward, is_done, self.tot_t, info


    def _advance_stage(self):
        is_error = False
        return self.stage + 1, is_error


    # action

    # get obs
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
