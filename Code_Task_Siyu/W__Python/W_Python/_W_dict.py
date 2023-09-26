import inspect

class W_dict():
    def W_dict_function_arguments():
        frame = inspect.currentframe().f_back
        keys, _, _, values = inspect.getargvalues(frame)
        kwargs = {}
        for key in keys:
            if key != 'self':
                kwargs[key] = values[key]
        return kwargs

    def W_dict_updateonly(dict1, dict2):
        dict1.update((k, dict2[k]) for k in dict1.keys() & dict2.keys())
        return dict1

    def W_dict_deleteNone(dict1):
        dict1 = dict((k,v) for k,v in dict1.items() if v is not None)
        return dict1