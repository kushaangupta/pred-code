from W_Env.W_Env import W_Env
from W_Env.W_Env_Simulator import W_Env_Simulator

if __name__ == "__main__": 
    render_mode = "human"
    n_maxTrials = 100
    # envname = input("input environment name:")
    # envname = "Pavlovian"
    envname = "Tokens"
    # envname = "GoalsActions"
    # envname = "TwoStep"
    env = W_Env(envname = envname, render_mode = render_mode, \
                            n_maxTrials = n_maxTrials, \
                            is_save = True, version = 'default')
    player = W_Env_Simulator(env)
    player.play(mode = "human")