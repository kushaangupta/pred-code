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
    if not type(a) == type([]):
        a = [a]
    return a