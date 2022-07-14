# -*- coding: utf-8 -*-
"""Select server (this or subserver) for detection.
"""

import os
import sys
import time
import json
import glob
import pickle
import shutil
import pathlib
import requests
import functools # for lists comparision

import contextlib

from threading import Thread, Semaphore, get_ident, Lock
#import threading, queue
#from multiprocessing import shared_memory#, Lock
import shared_memory
#from NamedAtomicLock import NamedAtomicLock
from named_atomic_lock import NamedAtomicLock
from constants import SHM_PRF

sys.path.append('../System')
from load_config import load_config

#from zmq_wrapper import ZmqWrapper, MON_HOST
from shm_wrapper_t import ShmWrapperT

SENTINEL = -1


class Controller():
    def __init__(self, json_name):
        #(DET_PROCS, EMB_PROCS, subservers, IMAGE_SIZE, S_MIN, PADDING, NEED_EMBED, EMBEDDER_BATCH) = load_config(json_name)
        CONF_DCT = load_config(json_name)
        self.DET_PROCS = CONF_DCT['DET_PROCS']
        self.EMBEDDER_BATCH = CONF_DCT['EMBEDDER_BATCH']
        self.IMAGE_SIZE = CONF_DCT['IMAGE_SIZE']
        self.QUEUES_DIR = CONF_DCT['QUEUES_DIR']
        self.TIMEOUT_TO_REMOVE_FROM_QUEUE = CONF_DCT['TIMEOUT_TO_REMOVE_FROM_QUEUE']

        self.shm_stop = None
        try:
            self.shm_stop = shared_memory.SharedMemory2(name=SHM_PRF+'stop')
        except Exception:
            self.shm_stop = shared_memory.SharedMemory2(name=SHM_PRF+'stop', create=True, size=8 * 128)
        self.shm_a = shared_memory.SharedMemory2(name=SHM_PRF+'a') # last_handled_id, last_saved_id, reserve
        self.shm_b = shared_memory.SharedMemory2(name=SHM_PRF+'b')
        self.shm_c = shared_memory.SharedMemory2(name=SHM_PRF+'c')
        self.shm_a_lock = NamedAtomicLock(SHM_PRF+'a_lock')
        self.shm_b_lock = NamedAtomicLock(SHM_PRF+'b_lock')
        self.shm_c_lock = NamedAtomicLock(SHM_PRF+'c_lock')
        #self.zmq_mon = ZmqWrapper(identifier='controller', addr=MON_HOST+':%d'%CONF_DCT['PORT_MON_IN'], tp='client', resp_type='with_response')
        self.zmq_mon = ShmWrapperT('controller', 'monitor', 4096)
        self.start_string = 'python3 src/controller.py %s' % (json_name)


    def need_stop_all_processes(self):
        stop_proc_flag = int.from_bytes(self.shm_stop.buf[0:8], 'big')
        return bool(stop_proc_flag)

    def get_root_home_loading(self):
        block_usage_pct_root, block_usage_pct_home = 0, 0
        with contextlib.closing(open('/etc/mtab')) as fp:
            for m in fp:
                fs_spec, fs_file, fs_vfstype, fs_mntops, fs_freq, fs_passno = m.split()
                if fs_file == '/':
                    r = os.statvfs(fs_file)
                    block_usage_pct_root = 100.0 - (float(r.f_bavail) / float(r.f_blocks) * 100)
                if fs_file == '/home':
                    r = os.statvfs(fs_file)
                    block_usage_pct_home = 100.0 - (float(r.f_bavail) / float(r.f_blocks) * 100)
        return block_usage_pct_root, block_usage_pct_home

    def run(self):
        # Main thread
        iii = 0
        while True:
            if self.need_stop_all_processes():
                break

            # === SEND ===
            # lOADING
            if (iii % 10) == 0:
                root_loading, home_loading = self.get_root_home_loading()
                #print('Controller will send...')
                self.zmq_mon.send({'act': 'root_home_loading', 'root_loading': root_loading, 'home_loading': home_loading, 'start_string': self.start_string}, msg_type='run')
                #print('Controller sended')

            # FPATHS
            #fpaths_svd = glob.glob(self.QUEUES_DIR + 'saved/*.pkl')
            #fpaths_dtd = glob.glob(self.QUEUES_DIR + 'detected/*.pkl')
            #fpaths_rsd = glob.glob(self.QUEUES_DIR + 'resulted/*.pkl')
            #print('Controller will send...')
            #self.zmq_mon.send({'act': 'len_fpaths', 'len_saved': len(fpaths_svd), 'len_detected': len(fpaths_dtd), 'len_resulted': len(fpaths_rsd), 'start_string': self.start_string}, msg_type='run')
            len_runner_detect = int.from_bytes(self.shm_a.buf[3*8:3*8+8], 'big')
            len_runner_embed  = int.from_bytes(self.shm_a.buf[4*8:4*8+8], 'big')
            len_server        = int.from_bytes(self.shm_a.buf[5*8:5*8+8], 'big')
            self.zmq_mon.send({'act': 'len_fpaths', 'len_runner_detect': len_runner_detect, 'len_runner_embed': len_runner_embed, 'len_server': len_server, 'start_string': self.start_string}, msg_type='run')
            #print('Controller sended')

            # === CONTROL ===
            # SHMS
            self.shm_b.remove_old_values_from_shm()
            self.shm_c.remove_old_values_from_shm()
            #
            time.sleep(1)
            iii += 1

        print('STOP Controller...')
        print('STOPped Controller!')


def main():
    if len(sys.argv) != 2:
        print("""USAGE: controller.py cnf_file_name
EXAMPLE: controller.py config_video003.txt""")
        exit(1)

    r = Controller(sys.argv[1])
    r.run()

################################################################################

if __name__ == "__main__":
    main()

################################################################################
