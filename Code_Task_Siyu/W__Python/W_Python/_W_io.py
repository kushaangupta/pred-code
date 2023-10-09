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
    
    def W_filewrap(file):
        path, _ = os.path.split(file)
        W_io.W_mkdir(path)
        return file

    def W_enext(file, extname):
        return os.path.splitext(file)[0] + "." + extname
    
    def W_file_prefix(file, pfx):
        p, n = os.path.split(file)
        return os.path.join(p, f"{pfx}_{n}")
    
    def W_file_suffix(file, sfx):
        p, n = os.path.split(file)
        n, e = os.path.splitext(n)
        return os.path.join(p, f"{n}_{sfx}" + e)
