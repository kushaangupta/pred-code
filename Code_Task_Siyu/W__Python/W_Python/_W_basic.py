import time

class W_basic():
    def W_tic():
        global W_tic_toc_time
        W_tic_toc_time = time.time()

    def W_toc(str = "elapse time is "):
        global W_tic_toc_time
        elapsetime =  time.time() - W_tic_toc_time
        if str is not None:
            print(f"{str} {elapsetime}")
        return elapsetime
