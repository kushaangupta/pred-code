import torch
from collections import namedtuple 
from ._W_list import W_list

class W_namedtuple():
    def W_list_of_dict_to_namedtuple(d):
        d = W_list.W_enlist(d)
        keys = list(d[0].keys())
        tuple_buffer = namedtuple('Buffer', keys)
        
        buffer = []
        for i in keys:
            buffer += [torch.concat([x[i].unsqueeze(0) for x in d])]

        out = tuple_buffer(*buffer)
        return out        
