from .W_Gym import W_Gym
import numpy as np
# from gym import spaces

class Grid2D():
    xy_range = None
    pos_grid2D = None
    lastpos_grid2D = None
    def __init__(self, x1 = -np.Inf, x2 = np.Inf, y1 = -np.Inf, y2 = np.Inf, **kwarg):
        self.xy_range = np.array([[x1, x2],[y1, y2]])
        self.set_pos(**kwarg)

    def set_pos(self, x0 = None, y0 = None):
        if x0 is not None and y0 is not None:
            self.lastpos_grid2D = self.pos_grid2D
            self.pos_grid2D = self.restrict2range(np.array([x0, y0]))
    
    def move(self, dx, dy, **kwarg):
        d = np.array([dx, dy])
        pos = d + self.pos_grid2D
        self.set_pos(pos[0], pos[1])
        return self._get_pos(**kwarg)

    def restrict2range(self, pos):
        for i in range(2):
            if pos[i] < self.xy_range[i][0]:
                pos[i] = self.xy_range[i][0]
            if pos[i] > self.xy_range[i][1]:
                pos[i] = self.xy_range[i][1]
        return pos
    
    def _get_pos(self): 
        return np.array([self.pos_grid2D[0], self.pos_grid2D[1]])
        
class W_Gym_Grid2D(W_Gym):
    size_grid = None
    key_preset = "arrow+"
    def __init__(self, nx, ny, ndim_obs = 1, key_preset = "array+", **kwarg):
        # ndim_obs is the dimension of observation (number of input channels)
        super().__init__(**kwarg)
        #self.observation_space = spaces.Box(low = 0, high = 1, shape = (nx, ny, ndim_obs), dtype = np.int8)
        self.size_grid = np.array([nx, ny])
        self._obs = np.zeros([nx, ny, ndim_obs])
        self.setup_obs_dim(nx, ny, ndim_obs)
        n_motor = self.setup_key_Grid2D(key_preset)
        self.setup_action_dim(nx * ny, n_motor)
        self.gaze = Grid2D(0,nx-1,0,ny-1) # action object
        self.plot_position = self.pos_grid(nx, ny) * self._metadata_render['window_size'] # x is row (which is dimension 1), y is column (which is dimension 0)
        
    def setup_key_Grid2D(self, key_preset):
        if key_preset == "arrow+":
            n_motor = 5
        elif key_preset == "arrow":
            n_motor = 4
        elif key_preset == "binary+":
            n_motor = 3
        elif key_preset == "binary":
            n_motor = 2
        self.key_preset = key_preset
        return n_motor
        
    def pos_grid(self, nx, ny): # x is x, y is y (descartes coordinate)
        x = np.linspace(0, 1, ny, endpoint=False) + 1/ny/2
        y = np.linspace(0, 1, nx, endpoint=False) + 1/nx/2 # x is -y, y is x
#        y = np.flip(y)
        xv, yv = np.meshgrid(x, y)
        pos = np.stack((xv, yv))
        pos = np.moveaxis(pos, [0,1,2],[2,0,1])
        return pos
        
    def draw_onehot_2D(self, xy, channel_name):
        if self._next_screen is None:
            self.blankscreen()
        x, y = xy
        assert self._obs_channel_namedict is not None
        channelID = self._obs_channel_namedict[channel_name]
        self._next_screen[x, y, channelID] = 1
            
    def one_hot_2Dc(self, x, y):
        data = np.zeros(self.size_grid)
        # data = data[:,:,0]
        data[x,y] = 1
        data = np.expand_dims(data,axis = 2)
        return data
 
    def vec2mat(self, gaze): # x, y start from 0
        ny = self.size_grid[1]
        y = gaze % ny
        x = int((gaze - y)/ny)
        return x, y

    def mat2vec(self, x, y): # x, y start from 0
        ny = self.size_grid[1]
        gaze = x * ny + y
        return gaze

    def action2dir(self, action): 
        action = int(action)
        if self.key_preset == "arrow+": # action: 0, 1, 2, 3, 4
            pass
        elif self.key_preset == "arrow": # action = 0,1,2,3
            action = action + 1
            # action: 1,2,3,4
        elif self.key_preset == "binary+": # action: 0,1,2
            if action == 2:
                action = 3
            # action: 0,1,3
        elif self.key_preset == "binary": # action: 0, 1
            action = action * 2 + 1
            # action: 1, 3

        #if action > 0:
        #    action = 5 - action # action = 0, 4, 3, 2, 1
        if action % 2 == 0: # dx = 0, 0, 1, 0, -1
            dx = 0
        else:
            dx = action - 2
        if action % 2 == 1 or action == 0: # dy = 0, -1, 0, 1, 0
            dy = 0
        else:
            dy = 3 - action                        
        # mapping 
        # action:0, 1,  2,  3,  4
        # dx:    0, -1, 0,  1,  0
        # dy:    0, 0,  1,  0, -1
        
        return dx, dy
    
    def transform_actions(self, action):
        dx, dy = self.action2dir(action) # dx, dy axis (x is x, y is y, descartes coordinate)
        gz = self.gaze.move(-dy, dx) # gaze axis (x is -y, y is x)
        gz = self.mat2vec(gz[0], gz[1])
        return gz

    def custom_reset_trial(self):
        self.gaze.set_pos(0,0)

    def custom_render_frame_action(self, canvas):
        if self.gaze.pos_grid2D is not None:
            canvas = self._render_frame_grid2D(canvas, self.gaze.pos_grid2D, 'action')
        if self.gaze.lastpos_grid2D is not None:
            canvas = self._render_frame_grid2D(canvas, self.gaze.lastpos_grid2D, 'action', pointscale = 0.5)
        return canvas
    
    def custom_render_frame_obs(self, canvas):
        canvas = self._render_frame_grid2D(canvas, self._obs, 'obs')
        return canvas
    
    def _render_frame_grid2D(self, canvas, data, dictname, pointscale = 1):
        import pygame
        params = self._renderer_auto_params[dictname]
        if len(data.shape) == 1:
            data = self.one_hot_2Dc(data[0], data[1])
        n_channel = data.shape[2]            
        for ci in range(n_channel):
            tcol = params['colors'][ci]
            tplottype = params['plottypes'][ci]
            tradius = params['radius'][ci]
            for xi in range(self.dim_obs[0]):
                for yi in range(self.dim_obs[1]):
                    tval = data[xi, yi, ci]
                    tpos = self.plot_position[xi, yi]
                    if tval > 0: # show
                        if tplottype == "circle":
                            pygame.draw.circle(canvas, tcol, tpos, np.mean(tradius) * pointscale)
                        elif tplottype == "square":
                            pygame.draw.rect(canvas, tcol, 
                                np.concatenate((-tradius + tpos, tradius * 2), axis = None), 0)
        return canvas
