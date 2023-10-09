from task_TwoStep_Confidence_2frame import task_TwoStep_Confidence_2frame
from W_Gym.W_Gym_simulator import W_env_simulator
render_mode = "human"
n_maxTrials = 100
env = task_TwoStep_Confidence_2frame(render_mode = render_mode, \
                        n_maxTrials = n_maxTrials)
player = W_env_simulator(env)
player.set_keys(keys = ['space', 'left', 'up', 'right','down'], actions = [4,0,2,1,3])
player.play()