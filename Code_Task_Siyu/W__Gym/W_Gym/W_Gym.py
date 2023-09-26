
from .W_Gym_renderer import W_Gym_renderer
# from .W_Gym_task import W_Gym_task

class W_Gym(W_Gym_renderer):
    # interactive component
    human_key_action = None
    # instance = None
    
    def setup_human_keys_auto(self, option = None):
        if option == "binary":
            keys = ['left', 'right']
            actions = [0,1]
        elif option == "binary_plus":
            keys = ['space', 'left', 'right']
            actions = [0,1,2]
        elif option == "arrows":
            keys = ['left', 'up', 'right', 'down']
            actions = [0,1,2,3]
        elif option == "arrows_plus":
            keys = ['space', 'left', 'up', 'right','down']
            actions = [0,1,2,3,4]
        self.set_human_keys(keys, actions)

    def set_human_keys(self, keys, actions):
        self.human_key_action = {'keys':keys, 'actions': actions}
    
    # def __getattr__(self, name):
    #     return self.instance.__getattribute__(name)







