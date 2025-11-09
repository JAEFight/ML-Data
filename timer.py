import time
import datetime

def start_timer(min = 0):
    a = int(time.time())
    time.sleep(min)
    return a

def stop_timer(a):
    b = int(time.time())
    t = b-a
    return t
