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
import requests

import numpy as np # type: ignore
import cv2

from multiprocessing import Process#, Queue, Value # pylint: disable=W0611
from threading import Thread, Semaphore, get_ident, Lock
from queue import Queue
#import threading, queue
#from multiprocessing import shared_memory#, Lock
import shared_memory
#from NamedAtomicLock import NamedAtomicLock
from named_atomic_lock import NamedAtomicLock
from constants import SHM_PRF

from pipesworker import PipesWorker
from images_bytes_conversions import get_np_image_from_base64, get_base64_from_np_image, get_base64_from_np_array
sys.path.append('../System')
from conversions import int_to_bytes, int_from_bytes # pylint: disable=E0401, C0413
from load_config import load_config

from constants import RUNNED, NEED_STOP_PROC, NEED_STOP_THRD, STOPPED, WAIT_DET_PROC_RESP

#from zmq_wrapper import ZmqWrapper, MON_HOST, DET_HOST, EMB_HOST, SRV_HOST
from shm_wrapper_t import ShmWrapperT

SENTINEL = -1
QUARTER_QUEUE_TEST_CID = 9900
pipe_in_name = './pipes/pipe_detector_main_'
pipe_out_name = './pipes/pipe_main_detector_'


################################################################################

import logging

logger = logging.getLogger('RunnerDetect')
logger.setLevel(logging.INFO)

# Create file handler
logger_handler = logging.FileHandler('./logs/runner_detect.log')
logger_handler.setLevel(logging.INFO)
# Create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

#logger_formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
logger_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
logger_handler.setFormatter(logger_formatter)

logger.addHandler(logger_handler)
#logger.addHandler(ch)

################################################################################


"""class Lock():
    def __init__(self):
        self.flag = False

    def locked(self):
        return self.flag

    def release(self):
        self.flag = False

    def acquire(self):
        while self.flag:
            time.sleep(0.1 / 1000)
        self.flag = True
"""


class RunnerDetect(PipesWorker):
    def __init__(self, json_name):
        PipesWorker.__init__(self, '', '', blocked=False)

        self.json_name = json_name
        CONF_DCT = load_config(json_name)
        self.CONF_DCT = CONF_DCT
        self.DET_PROCS = CONF_DCT['DET_PROCS']
        self.EMBEDDER_BATCH = CONF_DCT['EMBEDDER_BATCH']
        self.IMAGE_SIZE = CONF_DCT['IMAGE_SIZE']
        self.N_GPUS = CONF_DCT['N_GPUS']

        self.last_loaded_id = 0

        self.imgs_dt_lst = []
        self.imgs_dt_len = 0
        self.imgs_dt_lock = Lock()

        self.shm_stop = shared_memory.SharedMemory2(name=SHM_PRF+'stop') # last_handled_id, last_saved_id, reserve
        self.shm_a = shared_memory.SharedMemory2(name=SHM_PRF+'a') # last_handled_id, last_saved_id, reserve
        self.shm_b = shared_memory.SharedMemory2(name=SHM_PRF+'b') # 4*EMBEDDER_BATCH of 160*160*3 images + 13 rect+keypoints+agegender
        self.shm_c = shared_memory.SharedMemory2(name=SHM_PRF+'c') # 4*EMBEDDER_BATCH of 512 embeddeings
        #self.shm_a_lock = NamedAtomicLock(SHM_PRF+'a_lock')
        self.shm_b_lock = Lock()  # NamedAtomicLock(SHM_PRF+'b_lock')
        #self.shm_c_lock = NamedAtomicLock(SHM_PRF+'c_lock')
        self.start_lock = Lock()
        self.stop_resp_thrd = False

        self.zmq_mon = ShmWrapperT('runner_v3_detect', 'monitor', 4096)
        self.zmq_dets = []
        for detector_id in range(self.DET_PROCS):
            zmq_detector = ShmWrapperT(f'detect_thread_{detector_id}', f'detector_{detector_id}', 4*1024*1024)
            self.zmq_dets.append(zmq_detector)
        self.last_worked_detector_id = self.DET_PROCS - 1
        self.start_string = 'python3 src/runner_v3_detect.py %s' % json_name


    def push_quarter_queue_for_all_detectors_test(self):
        fname = './saved_for_internal_tests/000000000000000000000001_9900.pkl'
        f = open(fname, 'rb')
        (req_id, client_id, img_id, nparr, status, dict_stat) = pickle.load(f)
        req_id, client_id, img_id = 0, QUARTER_QUEUE_TEST_CID, ''
        dict_stat['det_load_from_saved_tm'] = time.time()
        dict_stat['saved_queue_len'] = 100500
        #dict_stat['qsize'] = self.q.qsize()
        dict_stat['det_put_to_internal_queue_tm'] = time.time()
        data = (req_id, client_id, img_id, nparr, status, dict_stat)
        identifier, tm, msg_type = 'server3_mt', time.time(), 'send_image_test'
        dt = (identifier, tm, msg_type, data)
        for i in range(self.DET_PROCS * 2):
            self.zmq_callback_func(dt)


    def zmq_callback_func(self, dt):
        #print(f'runner_v3_detect: Runner_V3_Detect: zmq_callback_func')
        (identifier, tm, msg_type, data) = dt
        if data:
            (req_id, client_id, img_id, nparr, status, dict_stat) = data
            print(f'runner_detect: RunnerDetect: zmq_callback_func: req_id={req_id}  self.imgs_dt_len={self.imgs_dt_len}')
            self.imgs_dt_lock.acquire()
            self.imgs_dt_lst.append((req_id, client_id, img_id, nparr, status, dict_stat))
            self.imgs_dt_len += 1
            self.shm_a.buf[3*8:3*8+8] = self.imgs_dt_len.to_bytes(8, 'big')
            self.imgs_dt_lock.release()
        else:
            print(f'runner_detect: RunnerDetect: zmq_callback_func: req_id={None}  self.imgs_dt_len={self.imgs_dt_len}')

    def run(self):
        # Main thread
        print('RunnerDetect main thread started')
        id_to_load = 1 # first id to load
        last_quarter_filled_queue_tm = time.time()
        while True:
            if self.need_stop_all_processes():
                print('RunnerDetect main thread exited by need_stop_all_processes')
                break

            self.zmq_mon.send({'act': 'running', 'start_string': self.start_string}, msg_type='run')
            if self.imgs_dt_len >= 1:
                last_quarter_filled_queue_tm = time.time()
            if time.time() - last_quarter_filled_queue_tm >= 30:  # All detectors test once in 30 seconds
                self.push_quarter_queue_for_all_detectors_test()

            start_tm = time.time()
            # Load images and detect
            # # load images
            last_svd_req_id = int.from_bytes(self.shm_a.buf[0:8], 'big')

            if self.imgs_dt_len == 0:
                time.sleep(0.05)
                continue

            gogogo = False
            for _ in range(self.DET_PROCS):
                self.last_worked_detector_id += 1
                if self.last_worked_detector_id >= self.DET_PROCS:
                    self.last_worked_detector_id = 0
                shft = 8 * (6 + self.last_worked_detector_id)
                is_detector_busy_flag = int.from_bytes(self.shm_a.buf[shft:shft+8], 'big')
                if is_detector_busy_flag == 0:
                    one = 1
                    self.shm_a.buf[shft:shft+8] = one.to_bytes(8, 'big')
                    gogogo = True
                    break
            if not gogogo:
                time.sleep(0.025)
                continue

            self.imgs_dt_lock.acquire()
            (req_id, client_id, img_id, nparr, status, dict_stat) = self.imgs_dt_lst[0]
            dict_stat['det_load_from_saved_tm'] = time.time()
            dict_stat['saved_queue_len'] = self.imgs_dt_len
            self.imgs_dt_lst = self.imgs_dt_lst[1:]
            self.imgs_dt_len -= 1
            self.imgs_dt_lock.release()

            #print('Runner_V3_Detect  shm_b_empty=', shm_b_empty, '  shm_b_filled=', shm_b_filled, '  self.q.qsize()=', self.q.qsize())
            np_image = nparr #cv2.imdecode(nparr, -1)[..., ::-1]
            precisedetection = False
            dict_stat['qsize'] = 0  # self.q.qsize()
            dict_stat['det_put_to_internal_queue_tm'] = time.time()
            dict_stat['det_get_from_internal_queue_tm'] = time.time()
            #self.q.put((req_id, client_id, img_id, np_image, status, dict_stat), block=False)
            self.zmq_dets[self.last_worked_detector_id].send((req_id, client_id, img_id, np_image, dict_stat, precisedetection), msg_type='run')

        #self.stop_resp_thrd = True
        logger.error('Main thread stopped')
        logger.error('RunnerDetect: main thread stopped')
        # Join resp thread
        #resp_thr.join()


def main():
    if len(sys.argv) != 2:
        print("""USAGE: runner_detect.py cnf_file_name
EXAMPLE: runner_detect.py config_all.txt""")
        exit(1)

    r = RunnerDetect(sys.argv[1])
    r.run()

################################################################################

if __name__ == "__main__":
    main()

################################################################################
