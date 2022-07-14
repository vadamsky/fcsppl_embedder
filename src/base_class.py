#!/usr/bin/env python3

import time
import sys
import os
import pickle
#import shared_memory
#from named_atomic_lock import NamedAtomicLock


class BaseClass():
    def __init__(self):
        #try:
        #    self.shm_stop = shared_memory.SharedMemory2(name='shm_stop', create=True, size=8 * 64)
        #except Exception:
        #    # if exist now
        #    shm_stop = shared_memory.SharedMemory2(name='shm_stop')
        pass
