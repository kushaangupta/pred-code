from W_Gym.W_Gym import W_Gym
import W_Python.W_tools as W
import numpy as np
from gym import spaces

class grid2D():
    xy_range = None
    pos_grid2D = None
    def __init__(self, x1 = -np.Inf, x2 = np.Inf, y1 = -np.Inf, y2 = np.Inf, **kwarg):
        self.xy_range = np.array([[x1, x2],[y1, y2]])
        self.set_pos(**kwarg)

    def set_pos(self, x0 = None, y0 = None):
        if x0 is not None and y0 is not None:
            self.pos_grid2D = self.restrict2range(np.array([x0, y0]))
    
    def move(self, dx, dy, **kwarg):
        d = np.array([dx, dy])
        pos = d + self.pos_grid2D
        self.pos_grid2D = self.restrict2range(pos)
        return self._get_pos(**kwarg)

    def restrict2range(self, pos):
        for i in range(2):
            if pos[i] < self.xy_range[i][0]:
                pos[i] = self.xy_range[i][0]
            if pos[i] > self.xy_range[i][1]:
                pos[i] = self.xy_range[i][1]
        return pos
    
    def _get_pos(self): 
        x, y = (self.pos_grid2D[0], self.pos_grid2D[1])
        return x, y
    
class W_Gym_grid2D(W_Gym):
    plot_position = None
    def __init__(self, nx, ny, ndim_obs = 1, **kwarg):
        # ndim_obs is the dimension of observation (number of input channels)
        super().__init__(**kwarg)
        self.observation_space = spaces.Box(low = 0, high = 1, shape = (nx, ny, ndim_obs), dtype = np.int8)
        self.obs = np.zeros(self.observation_space.shape)
        self.gaze = grid2D(0,2,0,2) # action object
        self._action_dimension = nx * ny
        self.plot_position = self.pos_grid(nx, ny) * self.metadata_render['window_size']
        
    def pos_grid(self, nx, ny):
        x = np.linspace(0, 1, nx, endpoint=False) + 1/nx/2
        y = np.linspace(0, 1, ny, endpoint=False) + 1/ny/2
        y = np.flip(y)
        xv, yv = np.meshgrid(x, y)
        pos = np.stack((xv, yv))
        pos = np.moveaxis(pos, [0,1,2],[2,0,1])
        return pos
        
    def draw(self, xy, channel_name):
        if self.currentscreen is None:
            self.blankscreen()
        x, y = xy
        channelID = self.obs_Name2DimNumber[channel_name]
        self.currentscreen[x, y, channelID] = 1

    def _render_frame_grid2D(self, canvas, data, dictname):
        import pygame
        params = self.plot_params[dictname]
        if len(data.shape) == 1:
            data = self.one_hot_2D(data[0], data[1])
        n_channel = data.shape[2]            
        for ci in range(n_channel):
            tcol = params['colors'][ci]
            tplottype = params['plottypes'][ci]
            tradius = params['radius'][ci]
            for xi in range(self.observation_space.shape[0]):
                for yi in range(self.observation_space.shape[1]):
                    tval = data[xi, yi, ci]
                    tpos = self.plot_position[xi, yi]
                    if tval > 0: # show
                        if tplottype == "circle":
                            pygame.draw.circle(canvas, tcol, tpos, np.mean(tradius))
                        elif tplottype == "square":
                            pygame.draw.rect(canvas, tcol, 
                                np.concatenate((-tradius + tpos, tradius * 2), axis = None), 0)
        return canvas
            
    def one_hot_2D(self, x, y):
        data = np.zeros(self.observation_space.shape)
        data = data[:,:,0]
        data[x,y] = 1
        data = np.expand_dims(data,axis = 2)
        return data
 
    def vec2mat(self, gaze):
        ny = self.observation_space.shape[1]
        y = gaze % ny
        x = int((gaze - y)/ny)
        return x, y

    def mat2vec(self, x, y):
        ny = self.observation_space.shape[1]
        gaze = x * ny + y
        return gaze

    def action2dir(self, action):
        if action > 0:
            action = 5 - action
        if action % 2 == 0:
            dx = 0
        else:
            dx = action - 2
        if action % 2 == 1 or action == 0:
            dy = 0
        else:
            dy = 3 - action                        
        # mapping 
        # action:0, 1,  2,  3,  4
        # dx:    0, -1, 0,  1,  0
        # dy:    0, 0,  1,  0, -1
        return dx, dy
    
    def _action_transform(self, action):
        dx, dy = self.action2dir(action)
        gx, gy = self.gaze.move(dx, dy)
        gz = self.mat2vec(gx, gy)
        return gz

    def _reset_trial(self):
        self.gaze.set_pos(0,0)

    def _render_frame_action(self, canvas):
        canvas = self._render_frame_grid2D(canvas, self.gaze.pos_grid2D, 'action')
        return canvas
    
    def _render_frame_obs(self, canvas):
        canvas = self._render_frame_grid2D(canvas, self.obs, 'obs')
        return canvas
