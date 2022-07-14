#!/usr/bin/env python3

import time
import sys
import os
import os.path
import json
import pathlib
import shutil
import pickle
from shutil import copyfile
#import shared_memory
from constants import SHM_PRF

import numpy as np
import time

from queue import Queue
from threading import Thread

from numpy.linalg import norm
from Double import loadModel

from pipesworker import PipesWorker
#from shm_wrapper_t import ShmWrapperT
from load_config import load_config
from conversions import int_to_bytes, int_from_bytes

pipe_out_name = './pipes/pipe_embedder_main_'
pipe_in_name = './pipes/pipe_main_embedder_'


################################################################################

import logging

logger = logging.getLogger('Embedder')
logger.setLevel(logging.INFO)

# Create file handler
logger_handler = logging.FileHandler('./logs/common.log')
logger_handler.setLevel(logging.DEBUG)
# Create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

#logger_formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
logger_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
logger_handler.setFormatter(logger_formatter)

logger.addHandler(logger_handler)
#logger.addHandler(ch)

################################################################################


class Embedder(PipesWorker):
    def __init__(self, json_name, number_of_embedder, videocard_number=0):
        global pipe_in_name, pipe_out_name
        CONF_DCT = load_config(json_name)
        self.size = CONF_DCT['IMAGE_SIZE']
        self.QUEUES_DIR = CONF_DCT['QUEUES_DIR']
        pipe_in_name  = pipe_in_name.replace('./pipes', self.QUEUES_DIR+'pipes')
        pipe_out_name = pipe_out_name.replace('./pipes', self.QUEUES_DIR+'pipes')
        PipesWorker.__init__(self, pipe_in_name + '%d' % number_of_embedder,
                                   pipe_out_name + '%d' % number_of_embedder,
                                   blocked=False)

        ####self.shm_stop = shared_memory.SharedMemory2(name=SHM_PRF+'stop') # stop_flag, reserve
        ####print('shm_stop:', self.shm_stop)
        #self.zmq_mon = ShmWrapperT('embedder', 'monitor', 4096)
        self.start_string = 'python3 src/embedder.py %s' % (json_name)

        device = 'cpu:0'
        device = 'gpu:%d' % videocard_number
        self.device = device
        print('embedder device = ', device)
        self.model = loadModel('./weights/glint360k_cosface_r100_fp16_0.1.h5')

    def run(self):
        while True:
            if self.need_stop_all_processes():
                break
            #self.zmq_mon.send({'act': 'running', 'start_string': self.start_string}, msg_type='run')

            verse = self.read_from_pipein_strct()
            if verse is None: # need stop process
                break
            if verse == '': # length == 0
                continue #break
            print('Embedder input was readed')

            (images, req_id) = verse # list of numpy arrays

            imgs = np.zeros([len(images), 160, 160, 3])
            embeddings = []
            i = 0
            for img in images:
                img = np.reshape(img,  [1, 160, 160, 3])
                imgs[i] = img
                i += 1
            embeddings = self.model.predict([imgs, imgs])
            embs = [emb for emb in embeddings]

            rt = self.write_in_pipeout_strct(embs)
            if rt is None: # need stop process
                break
            print('Embedder ret was writed')
            #logger.info('Images written in output pipe num=%d  len=%d' % (num, len(verse)))

def main():
    if len(sys.argv) != 2:
        print("""USAGE: embedder.py cnf_file_name
EXAMPLE: embedder.py config_all.txt""")
        exit(1)

    e = Embedder(sys.argv[1], 0, 0)
    e.run()

################################################################################

if __name__ == "__main__":
    main()

################################################################################
