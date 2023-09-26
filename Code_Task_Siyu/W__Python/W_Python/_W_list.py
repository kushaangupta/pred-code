import numpy as np

class W_list():
    def W_enlist(a):
        if type(a) == type(np.array([])):
            a = a.tolist()
        if not type(a) == type([]):
            a = [a]
        return a
    
    def is_in_list(target, sourcelist = None):
        if sourcelist is None:
            is_valid = True
        else:
            sourcelist = W_list.W_enlist(sourcelist)
            is_valid = target in sourcelist
        return is_valid
        
    def W_list_findidx(targetstrs, sourcestrs):
        targetstrs = W_list.W_enlist(targetstrs)
        sourcestrs = W_list.W_enlist(sourcestrs)
        idx = np.array([np.where([j == i for j in iter(sourcestrs)]) for i in iter(targetstrs)]).squeeze()
        if idx.size == 1:
            idx = int(idx)
        return idx