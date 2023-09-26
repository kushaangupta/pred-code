from W_Env.task_Goal_Action import task_Goal_Action
from W_Env.task_Temporal_Discounting import task_Temporal_Discounting
from W_Env.task_Horizon import task_Horizon
from W_Env.task_TwoStep import task_TwoStep
from W_Env.task_TwoStep_Confidence import task_TwoStep_Confidence
from W_Env.task_TwoStep_simple import task_TwoStep_simple
from W_Env.task_TwoStep_1frame import task_TwoStep_1frame
from W_Env.task_TwoStep_Confidence_mini import task_TwoStep_Confidence_mini
from W_Env.task_TwoStep_Confidence_2frame import task_TwoStep_Confidence_2frame
# from W_Env.task_TwoStep_2frame_full import task_TwoStep_2frame_full
from W_Gym.W_Gym_simulator import W_env_simulator

def W_Env(envname, *arg, **kwarg):
    envnames  = ["MC", "WV", "Horizon", "TwoStep", "TwoStep_Confidence", \
                 "TwoStep_simple","TwoStep_1frame", "TwoStep_Confidence_mini", \
                 "TwoStep_Confidence_2frame", "TwoStep_2frame_full"]
    fullnames = ["task_Goal_Action", "task_Temporal_Discounting", "task_Horizon", \
                 'task_TwoStep', 'task_TwoStep_Confidence', 'task_TwoStep_Confidence_mini', \
                    'TwoStep_Confidence_2frame','task_TwoStep_2frame_full']
    if not envname in envnames:
        raise Exception("env not defined")
    if envname == "MC":
        env = task_Goal_Action(*arg, **kwarg)
    if envname == "WV":
        env = task_Temporal_Discounting(*arg, **kwarg)
    if envname == "Horizon":
        env = task_Horizon(*arg, **kwarg)
    if envname == "TwoStep":
        env = task_TwoStep(*arg, **kwarg)    
    if envname == "TwoStep_Confidence":
        env = task_TwoStep_Confidence(*arg, **kwarg)
    if envname == "TwoStep_simple":
        env = task_TwoStep_simple(*arg, **kwarg)
    if envname == "TwoStep_1frame":
        env = task_TwoStep_1frame(*arg, **kwarg)
    if envname == "TwoStep_Confidence_mini":
        env = task_TwoStep_Confidence_mini(*arg, **kwarg)
    if envname == "TwoStep_Confidence_2frame":
        env = task_TwoStep_Confidence_2frame(*arg, **kwarg)
    # if envname == "TwoStep_2frame_full":
    #     env = task_TwoStep_2frame_full(*arg, **kwarg)
    return env

class W_Env_player():
    env = None
    player = None
    def __init__(self, envname, *arg, **kwarg):
        self.env = W_Env(envname, *arg, **kwarg)
        self.envname = envname

    def get_env(self):
        return self.env
    
    def get_player(self):
        if self.player is not None:
            return self.player
        player = W_env_simulator(self.env)
        if self.envname == "MC":
            player.set_keys(keys = ['space', 'left', 'up', 'right','down'], actions = [0,1,2,3,4])
        if self.envname == "WV":
            player.set_keys(keys = ['space', 'a','b'], actions = [0,1,2])
        if self.envname in ["Horizon", "TwoStep", "TwoStep_simple", "TwoStep_Confidence_2frame"]:
            player.set_keys(keys = ['space', 'left', 'right'], actions = [0,1,2])
        if self.envname in ["TwoStep_1frame"]:
            player.set_keys(keys = ['left', 'right'], actions = [0,1])
        if self.envname in ["TwoStep_Confidence"]:
            player.set_keys(keys = ['space', 'left', 'right', 'up'], actions = [0,1,2,3])
        if self.envname in ["TwoStep_Confidence_mini"]:
            player.set_keys(keys = ['space', 'left', 'up', 'right','down'], actions = [4,0,2,1,3])
        self.player = player
        return player

    def play(self):
        if self.player is None:
            self.get_player()
        self.player.play()