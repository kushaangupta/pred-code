from typing import Any
from gym.wrappers import RecordVideo

def W_Env_creater(envname, *arg, **kwarg):
    tsk_name = "task_" + envname
    ldic = locals()
    exec(f"from W_Env.{tsk_name} import {tsk_name} as W_tsk", globals(), ldic)
    W_tsk = ldic['W_tsk']
    env = W_tsk(*arg, **kwarg)
    return env

class W_Env(object):
    def __new__(cls, envname = None, is_record = False, log_dir = './video', \
                 *arg, **kwarg):
        if envname is None:
            return None
        env = W_Env_creater(envname, *arg, **kwarg)
        if is_record:
            env = RecordVideo(env, log_dir)
        return env
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        self.__init__(*args, *kwds)