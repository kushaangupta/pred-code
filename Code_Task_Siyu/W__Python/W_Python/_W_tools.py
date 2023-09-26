
import numpy as np
import itertools
import torch

class W_tools():
    def W_counter_balance(params):
        names = [k for k in params]
        pars = [params[k] for k in params]
        lst = list(itertools.product(*pars))  
        lst = np.array(lst)

        dct = dict(((names[k], list(lst[:,k])) for k in range(lst.shape[1])))
        dct['n'] = lst.shape[0]
        return dct

    def W_onehot_array(x, n):
        if torch.is_tensor(x):
            if x is None:
                return torch.zeros(n)
            else:
                x = x.cpu().numpy()
                shape = x.shape
                x = x.reshape((1,-1))
                out = torch.squeeze(torch.eye(n)[x])
                out = out.reshape(shape + (n,))
                return out
        else: 
            if x is None:
                return np.zeros(n)
            else:
                shape = x.shape
                x = x.reshape((1,-1))
                out = np.squeeze(np.eye(n)[x])
                out = out.reshape(shape + (n,))
                return out
            
    def W_onehot(x, n):
        if x is None:
            return np.zeros(n)
        else:
            out = np.squeeze(np.eye(n)[x])
            return out
