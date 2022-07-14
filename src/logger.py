from datetime import datetime
from multiprocessing import Lock

_lock = Lock()

def get_now_tm_str():
    now = datetime.now()
    return '%04d-%02d-%02d %02d:%02d:%02d.%06d' % (now.year, now.month, now.day,
                                                   now.hour, now.minute, now.second, now.microsecond)

def create_or_clear_file(filename):
    f = open(filename, 'w')

def log_in_file(filename, text):
    global _lock
    _lock.acquire()
    now_tm_str = get_now_tm_str()
    f = open(filename, 'a')
    f.write('%s;%s\n' % (now_tm_str, text))
    f.close()
    _lock.release()
