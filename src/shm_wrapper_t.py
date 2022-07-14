# -*- coding: utf-8 -*-
"""Class for sending messages 2 monitor and recv they from monitor.
"""

#import zmq
import math
import time
import random
import pickle
from collections import defaultdict
from threading import Thread, Lock

from shared_memory import SharedMemory2

MAX_VALUE = 256 * 256 * 256 * 256 * 256 * 256 * 256 * 256 - 1

class ShmWrapperT:
    def __init__(self, in_name, out_name, shm_sz=4096, timeout=0.003, common_lck_of_several_shms=None):
        self.in_name = in_name
        self.out_name = out_name
        self.shm_sz = shm_sz
        self.timeout = timeout
        #
        self.serv_running = True
        self.shm = None
        shm_name = f'transport_{in_name}_{out_name}'
        try:
            self.shm = SharedMemory2(name=shm_name)
        except Exception:
            # first 8 - parts count of message
            # second 8 - part of message
            # third 8 - length of message part
            self.shm = SharedMemory2(name=shm_name, create=True, size=24 + shm_sz)
        self.shm_sz = self.shm.size - 24
        fill = 0
        for i in range(int(self.shm.size / 8)):
            self.shm.buf[i*8:i*8+8] = fill.to_bytes(8, 'big')
        self.lck = Lock()
        self.common_lck_of_several_shms = common_lck_of_several_shms

    def __del__(self):
        pass


    def common_lck_of_several_shms_acquire(self):
        if self.common_lck_of_several_shms:
            self.common_lck_of_several_shms.acquire()

    def common_lck_of_several_shms_release(self):
        if self.common_lck_of_several_shms:
            self.common_lck_of_several_shms.release()

    def send_func(self, data, msg_type=None, ):
        identifier = self.in_name
        verse = pickle.dumps((identifier, time.time(), msg_type, data))
        lenv = len(verse)
        parts_cnt = math.ceil(lenv / self.shm_sz)
        #print(f'lenv, parts_cnt: {lenv}, {parts_cnt}')
        self.common_lck_of_several_shms_acquire()
        self.lck.acquire()
        time_start = time.time()
        for part in range(parts_cnt):
            while True:
                if time.time() - time_start >= 5:  # ERROR SENDING
                    length_ = MAX_VALUE
                    self.shm.buf[16:24] = length_.to_bytes(8, 'big')
                    self.lck.release()
                    self.common_lck_of_several_shms_release()
                    return False
                length = int.from_bytes(self.shm.buf[16:24], 'big')
                #print(f'part, length: {part}, {length}')
                if length:
                    time.sleep(self.timeout)
                    continue
                break
            shft = part * self.shm_sz
            length = self.shm_sz if part < (parts_cnt - 1) else lenv - self.shm_sz * (parts_cnt - 1)
            #print(f'shft, length: {shft}, {length}')
            try:
                self.shm.buf[24:24+length] = verse[shft:shft+length]
            except Exception as e:
                print(f'Traceback in shm_wrapper_t: send_func:  lenv={lenv}  parts_cnt={parts_cnt}  part={part}  self.shm_sz={self.shm_sz}  shft={shft}  length={length}  len(self.shm.buf[16:16+length])={len(self.shm.buf[16:16+length])}  len(verse[shft:shft+length])={len(verse[shft:shft+length])}  type(self.shm.buf[shft:shft+length])={type(self.shm.buf[shft:shft+length])}  type(verse[shft:shft+length])={type(verse[shft:shft+length])}')
            prt = part  # parts_cnt - 1 - part
            self.shm.buf[0:8] = parts_cnt.to_bytes(8, 'big')
            self.shm.buf[8:16] = prt.to_bytes(8, 'big')
            self.shm.buf[16:24] = length.to_bytes(8, 'big')
        self.lck.release()
        self.common_lck_of_several_shms_release()
        return True

    def send(self, data, msg_type=None):
        return self.send_func(data, msg_type)


    def read(self, blocked=True):
        self.lck.acquire()
        while True:
            self.common_lck_of_several_shms_acquire()
            length = int.from_bytes(self.shm.buf[16:24], 'big')
            if length == 0:
                self.common_lck_of_several_shms_release()
                if not blocked:
                    self.lck.release()
                    return (None, time.time(), None, None)
                time.sleep(self.timeout)
                continue
            break
        #
        bad_respawn = False
        verse = b''
        while True:
            parts_cnt = int.from_bytes(self.shm.buf[0:8], 'big')
            prt = int.from_bytes(self.shm.buf[8:16], 'big')
            length = int.from_bytes(self.shm.buf[16:24], 'big')
            #print(f'parts_cnt={parts_cnt}  prt={prt}  length={length}')
            if verse == b'' and prt != 0:  # receiver was killed before read all, and respawned
                bad_respawn = True
            if length == 0:  # and prt > 0:
                time.sleep(self.timeout)
                continue
            length_ = 0
            if length == MAX_VALUE:  # ERROR SENDING
                self.shm.buf[16:24] = length_.to_bytes(8, 'big')
                self.lck.release()
                self.common_lck_of_several_shms_release()
                print('Traceback in shm_wrapper_t:read: bad_sending', f'  parts_cnt={parts_cnt}  length={length}  len(verse)={len(verse)}  names: {self.in_name} {self.out_name}')
                return (None, time.time(), None, None)
            #print(f'prt, length: {prt}, {length}')
            verse += bytes(self.shm.buf[24:24+length])
            time.sleep(self.timeout)
            self.shm.buf[16:24] = length_.to_bytes(8, 'big')
            if prt == parts_cnt - 1:
                break
            time.sleep(self.timeout)
        parts_cnt_ = 0
        self.shm.buf[0:8] = parts_cnt_.to_bytes(8, 'big')
        #print(f'len(verse): {len(verse)}')
        if bad_respawn:
            self.lck.release()
            self.common_lck_of_several_shms_release()
            print('Traceback in shm_wrapper_t:read: bad_respawn', f'  parts_cnt={parts_cnt}  length={length}  len(verse)={len(verse)}  names: {self.in_name} {self.out_name}')
            return (None, time.time(), None, None)
        try:
            (identifier, tm, msg_type, data) = pickle.loads(verse)
        except Exception as e:
            self.lck.release()
            self.common_lck_of_several_shms_release()
            print('Traceback in shm_wrapper_t:read:', e, f'  parts_cnt={parts_cnt}  length={length}  len(verse)={len(verse)}  names: {self.in_name} {self.out_name}')
            return (None, time.time(), None, None)
        self.lck.release()
        self.common_lck_of_several_shms_release()
        return (identifier, tm, msg_type, data)

    def run_serv(self, callback_func):
        while self.serv_running:
            (identifier, tm, msg_type, data) = self.read()
            callback_func((identifier, tm, msg_type, data))
        print('ShmWrapper.run_serv stopped')
