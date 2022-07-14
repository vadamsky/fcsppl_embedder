# -*- coding: utf-8 -*-
"""Wrapper of shared memory part.
"""

import os
import sys

import numpy as np # type: ignore

from multiprocessing import shared_memory#, Lock
from NamedAtomicLock import NamedAtomicLock


class ShmWrapper():

    def __init__(self, name, np_shape, np_dtype=np.float64, raw_size=0):
        self.raw_size = raw_size
        self.arr = np.zeros(np_shape, dtype=np_dtype)
        self.np_size = np.dtype(np.uint64).itemsize * np.prod(np.array(np_shape))
        size = self.np_size + raw_size
        #
        self.shm = None
        try:
            self.shm = shared_memory.SharedMemory(name=name, create=True, size=size)
        except Exception:
            # if exist now
            self.shm = shared_memory.SharedMemory(name=name)
        self.shm_lock = NamedAtomicLock(name + '_lock')
        #
        self.arr_shm = np.ndarray(np_shape, dtype=np_dtype, buffer=self.shm.buf[0:self.np_size]) # this is array in shared mem


    def get_arr_from_shm(self, lock=True):
        if lock:
            self.shm_lock.acquire()

        self.arr[:] = self.arr_shm[:] # copy from shm to self.arr

        if lock:
            self.shm_lock.release()

    def put_arr_to_shm(self, lock=True):
        if lock:
            self.shm_lock.acquire()

        #self.shm.buf[0:self.np_size] = self.arr.tobytes()
        self.arr_shm[:] = self.arr[:] # copy to shm from self.arr

        if lock:
            self.shm_lock.release()

    def get_int_from_shm(self, shm_shift=0, int_size=8, lock=True):
        if shm_shift + int_size > self.raw_size:
            raise ValueError('Space error!  shm_shift + int_size > raw_size')
        if lock:
            self.shm_lock.acquire()

        shift = self.np_size + shm_shift
        value = int.from_bytes(self.shm.buf[shift:shift + int_size], 'big')

        if lock:
            self.shm_lock.release()
        return value

    def put_int_to_shm(self, value, shm_shift=0, int_size=8, lock=True):
        if shm_shift + int_size > self.raw_size:
            raise ValueError('Space error!  shm_shift + int_size > raw_size')
        if lock:
            self.shm_lock.acquire()

        shift = self.np_size + shm_shift
        self.shm.buf[shift:shift + int_size] = value.to_bytes(int_size, 'big')

        if lock:
            self.shm_lock.release()
