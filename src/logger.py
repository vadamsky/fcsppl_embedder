import os
import time
from datetime import datetime
from multiprocessing import Lock

_lock = Lock()
_upd_dct = {}

def get_now_tm_str():
    now = datetime.now()
    return '%04d-%02d-%02d %02d:%02d:%02d.%06d' % (now.year, now.month, now.day,
                                                   now.hour, now.minute, now.second, now.microsecond)

def create_or_clear_file(filename):
    if os.path.exists(filename):
        os.system(f'mv {filename} old_{filename}')
    f = open(filename, 'w')

def log_in_file(filename, text, typ='INFO'):
    global _lock
    global _upd_dct
    #
    if filename[-4:] != '.csv':
        filename += '.csv'
    #
    _lock.acquire()
    now_tm_str = get_now_tm_str()
    f = open(filename, 'a')
    f.write('%s;%s;%s\n' % (now_tm_str, typ, text))
    f.close()
    #
    if _upd_dct.get(filename) is None:
        _upd_dct[filename] = time.time()
    #if time.time() - _upd_dct[filename] > 12 * 3600:
    tm_secs = int(time.time())
    #if tm_secs % (12 * 3600) == 0:
    #    os.system(f'mv {filename} old_{filename}')
    #    f = open(filename, 'w')
    #    f.write('%s;%s\n' % (now_tm_str, ''))
    #    f.close()
    #    #_upd_dct[filename] = time.time()
    #
    _lock.release()


