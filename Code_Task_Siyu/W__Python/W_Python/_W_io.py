import os

class W_io():
    def W_foldernames(path):
        return os.path.abspath(os.path.join(path, os.pardir))
    
    def W_mkdir(folder0):
        folder = [folder0]
        while not os.path.exists(folder[-1]):
            folder += [W_io.W_foldernames(folder[-1])]           
        for i in reversed(range(len(folder)-1)):
            if not os.path.exists(folder[i]):
                os.mkdir(folder[i])
        return folder0

