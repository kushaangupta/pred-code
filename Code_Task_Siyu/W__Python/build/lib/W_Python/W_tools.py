import time
import numpy as np
import os

def W_tic():
    global W_tic_toc_time
    W_tic_toc_time = time.time()

def W_toc(str = "elapse time is "):
    global W_tic_toc_time
    elapsetime =  time.time() - W_tic_toc_time
    print(f"{str} {elapsetime}")
    return elapsetime

def W_dict_deleteNone(**kwargs):
    res = dict((k,v) for k,v in kwargs.items() if v is not None)
    return res

def W_dict_kwargs():
    import inspect
    frame = inspect.currentframe().f_back
    keys, _, _, values = inspect.getargvalues(frame)
    kwargs = {}
    for key in keys:
        if key != 'self':
            kwargs[key] = values[key]
    return kwargs

def enlist(a):
    if type(a) == type(np.array([])):
        a = a.tolist()
    if not type(a) == type([]):
        a = [a]
    return a

def W_dict_updateonly(dict1, dict2):
    dict1.update((k, dict2[k]) for k in dict1.keys() & dict2.keys())
    return dict1

def W_onehot(x, n):
    return np.squeeze(np.eye(n)[x])

def W_mkdir(folder):
    if not os.path.exists(folder):
        os.mkdir(folder)
    return folder