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
import functools # for lists comparision

import numpy as np # type: ignore
import cv2

from multiprocessing import Process, Pool#, Queue, Value # pylint: disable=W0611
from threading import Thread, Semaphore, get_ident, Lock
#from queue import Queue
from kombu import Connection, Exchange, Queue, Producer
#import shared_memory
#from named_atomic_lock import NamedAtomicLock

from pipesworker import PipesWorker
from images_bytes_conversions import get_np_image_from_base64, get_base64_from_np_image, get_base64_from_np_array
sys.path.append('../System')
from conversions import int_to_bytes, int_from_bytes # pylint: disable=E0401, C0413
from load_config import load_config

from logger import log_in_file
from constants import SHM_PRF

#from zmq_wrapper import ZmqWrapper, MON_HOST, DET_HOST, EMB_HOST, SRV_HOST
#from shm_wrapper_t import ShmWrapperT

SENTINEL = -1
pipe_in_name = '/cache/pipes/pipe_embedder_main_'
pipe_out_name = '/cache/pipes/pipe_main_embedder_'


################################################################################

import logging

logger = logging.getLogger('RunnerEmbed')
logger.setLevel(logging.INFO)

# Create file handler
logger_handler = logging.FileHandler('./logs/runner_embed.log')
logger_handler.setLevel(logging.INFO)
# Create console handler and set level to debug
#ch = logging.StreamHandler()
#ch.setLevel(logging.INFO)

#logger_formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
logger_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
logger_handler.setFormatter(logger_formatter)

logger.addHandler(logger_handler)
#logger.addHandler(ch)

################################################################################


class RunnerEmbed(PipesWorker):
    def __init__(self, json_name):
        global pipe_in_name, pipe_out_name
        embedder_id = 0
        CONF_DCT = load_config(json_name)
        self.EMBEDDER_BATCH = CONF_DCT['EMBEDDER_BATCH']
        self.IMAGE_SIZE = CONF_DCT['IMAGE_SIZE']
        #self.TEST_QUEUE = CONF_DCT['TEST_QUEUE']
        self.BASE_QUEUE = CONF_DCT['BASE_QUEUE']
        PipesWorker.__init__(self, pipe_in_name + '%d' % embedder_id,
                                   pipe_out_name + '%d' % embedder_id,
                                   blocked=False)

        self.last_loaded_id = 0

        self.imgs_dt_lst = []
        self.imgs_dt_len = 0
        self.imgs_dt_lock = Lock()

        ####self.shm_a = shared_memory.SharedMemory2(name=SHM_PRF+'a') # last_handled_id, last_saved_id, reserve
        ####self.zmq_mon = ShmWrapperT('runner_embed', 'monitor', 4096)
        self.start_string = 'python3 src/server_in.py %s' % (json_name)

        ####self.zmq_srv = ShmWrapperT('runner_embed', 'client_out', 1024*1024)
        ####self.zmq_embs = []
        rabbit_url = f"amqp://guest2:guest2@{self.BASE_QUEUE}:5672/"
        self.conn = Connection(rabbit_url)
        self.channel = self.conn.channel()
        self.exchange = Exchange("base_exchange", type="direct")
        self.producer = Producer(exchange=self.exchange, channel=self.channel, routing_key="BOB")
        self.queue = Queue(name="base_queue", exchange=self.exchange, routing_key="BOB")
        self.queue.maybe_bind(self.conn)
        self.queue.declare()

        self.ids = np.zeros((self.EMBEDDER_BATCH), dtype=np.int64) # ids to copy from shm_b to shm_c
        self.cids = np.zeros((self.EMBEDDER_BATCH), dtype=np.int64) # client ids to copy from shm_b to shm_c
        self.sids = [b''] * self.EMBEDDER_BATCH # image string ids to copy from shm_b to shm_c
        self.imgs = np.zeros((self.EMBEDDER_BATCH, self.IMAGE_SIZE, self.IMAGE_SIZE, 3), dtype=np.uint8) # imgs to embed
        self.rect_kps = np.zeros((self.EMBEDDER_BATCH, 13), dtype=np.float64) # rect4 + keypoint6 + agegender3
        #
        embs = self.warm_up()
        self.flag = 1
        os.utime('./send_ok_check', (time.time(), time.time()))

    def get_embs(self, np_imgs, id_=0):
        tm_lst = [time.time(), None, None, None]
        if self.write_in_pipeout_strct((np_imgs, id_, tm_lst)) is None:
            print('runner_embed: error with write_in_pipeout_strct: None returned')
            return None
        (embs, tm_lst) = self.read_from_pipein_strct()
        tm_lst[3] = time.time()
        log_in_file('runner_embed_tms.csv', f'begin={tm_lst[0]} end={tm_lst[1]} emb_tm={tm_lst[2]-tm_lst[1]} trans_tm={tm_lst[3]-tm_lst[0]-tm_lst[2]+tm_lst[1]}')
        return embs

    def warm_up(self):
        np_img = np.ones((self.IMAGE_SIZE, self.IMAGE_SIZE, 3), dtype=np.uint8) * 255
        embs = self.get_embs([np_img], 0)
        print(embs)

        np_imgs = np.ones((self.EMBEDDER_BATCH, self.IMAGE_SIZE, self.IMAGE_SIZE, 3), dtype=np.uint8) * 255
        embs = self.get_embs(np_imgs, 0)

        print('np_imgs:', np_imgs.shape, np_imgs.dtype)
        #print('Embedder warm-up is ok! ', embs.shape)
        print('Embedder warm-up is ok! ', len(embs))
        return embs

    def need_stop_all_processes(self):
        ####stop_proc_flag = int.from_bytes(self.shm_stop.buf[0:8], 'big')
        return False  # bool(stop_proc_flag)

    def zmq_callback_func(self, dt):
        (identifier, tm, msg_type, data) = dt
        if data:
            (req_id, client_id, img_id, faces_imgs, frames, faces_keypoints, status, message_to_ack, dict_stat) = data
            log_in_file('runner_embed.csv', f'zmq_callback_func: req_id={req_id}, img_id={img_id}, client_id={client_id}  self.imgs_dt_len={self.imgs_dt_len}')
            print('RunnerEmbed: zmq_callback_func:  self.imgs_dt_len=', self.imgs_dt_len, '  req_id, client_id, img_id:', req_id, client_id, img_id, '  len(frames):', len(frames))
            self.imgs_dt_lock.acquire()
            self.imgs_dt_lst.append((req_id, client_id, img_id, faces_imgs, frames, faces_keypoints, status, message_to_ack, dict_stat))
            self.imgs_dt_len += 1
            ####self.shm_a.buf[4*8:4*8+8] = self.imgs_dt_len.to_bytes(8, 'big')
            self.imgs_dt_lock.release()

    def send_data_to_base(self, tpl):
        # tpl is: (req_id, client_id, img_id, ret_dicts, status, dict_stat)
        # each of ret_dicts has members:
        # 'req_id', 'cid', 'sid', 'np_img_160', 'embedding', 'rect', 'keypoints'
        # keypoints has members: 'left_eye', 'right_eye', 'nose'
        try:
            post_body = pickle.dumps(tpl)
            self.producer.publish(post_body)
        except Exception as e:
            log_in_file('runner_embed.csv', f'send_data_to_base: Error:  e={e}')
            return False
        else:
            log_in_file('runner_embed.csv', f'send_data_to_base: ok')
        return True

    def run(self):
        # Main thread
        while True:
            ####self.zmq_mon.send({'act': 'running', 'start_string': self.start_string}, msg_type='run')
            # load files
            loads = []
            statuses = []
            self.imgs = []

            #if self.imgs_dt_len:
            while self.imgs_dt_len:
                self.imgs_dt_lock.acquire()
                (req_id, client_id, img_id, faces_imgs, frames, faces_keypoints, status, message_to_ack, dict_stat) = self.imgs_dt_lst[0]
                dict_stat['emb_load_from_detected_tm'] = time.time()
                dict_stat['detected_queue_len'] = self.imgs_dt_len
                self.imgs_dt_lst = self.imgs_dt_lst[1:]
                self.imgs_dt_len -= 1
                self.imgs_dt_lock.release()

                """
                #if status == 'OK':
                #print('len(self.imgs), len(faces_imgs), self.EMBEDDER_BATCH:', len(self.imgs), len(faces_imgs), self.EMBEDDER_BATCH)
                if len(self.imgs) + len(faces_imgs) <= self.EMBEDDER_BATCH or len(self.imgs) <= (self.EMBEDDER_BATCH / 2):
                    if len(self.imgs) + len(faces_imgs) <= self.EMBEDDER_BATCH:
                        for img in faces_imgs:
                            self.imgs.append(img)
                        loads.append((req_id, client_id, img_id, faces_imgs, frames, faces_keypoints, dict_stat))
                    else:
                        shift = self.EMBEDDER_BATCH - len(self.imgs)
                        for img in faces_imgs[:shift]:
                            self.imgs.append(img)
                        loads.append((req_id, client_id, img_id, faces_imgs[:shift], frames[:shift], faces_keypoints[:shift], dict_stat))
                    statuses.append(status)
                else:
                    break  # while self.imgs_dt_len:
                """
                for img in faces_imgs:
                    self.imgs.append(img)
                loads.append((req_id, client_id, img_id, faces_imgs, frames, faces_keypoints, message_to_ack, dict_stat))
                statuses.append(status)
                #    else:
                #        shift = self.EMBEDDER_BATCH - len(self.imgs)
                #        for img in faces_imgs[:shift]:
                #            self.imgs.append(img)
                #        loads.append((req_id, client_id, img_id, faces_imgs[:shift], frames[:shift], faces_keypoints[:shift], dict_stat))
                #    statuses.append(status)
                #else:
                #    self.imgs_dt_lock.release()
                #    time.sleep(0.05)
                #    continue  # while self.imgs_dt_len:
                if len(self.imgs) >= self.EMBEDDER_BATCH:
                    break


            embs = None
            if len(self.imgs) == 0:
                #print('RunnerEmbed:  len(self.imgs) == 0')
                #print('RunnerEmbed:  len(loads)=', len(loads))
                time.sleep(0.5)
                continue
            else:
                # get embs
                #self.zmq_mon.send({'act': 'pre get_embs', 'len(self.imgs)': len(self.imgs)}, msg_type='run')
                try:
                    embs = self.get_embs(self.imgs, 0)
                except Exception as e:
                    log_in_file('runner_embed.csv', f'run: get_embs Error:  e={e}')
                    print('Error with  embs = self.get_embs(self.imgs, 0)')
                    logger.error('Embedder get_embs err')
                if not (embs is None):
                    log_in_file('runner_embed.csv', f'run: embs was calculated: len(embs)={len(embs)}')
                    print(f'embs was calculated: len(embs)={len(embs)}')

            print(f'len(loads)={len(loads)}')
            # save
            shift = 0
            for load, status in zip(loads, statuses):
                (req_id, client_id, img_id, faces_imgs, frames, faces_keypoints, message_to_ack, dict_stat) = load
                embs_ = embs[shift:shift+len(faces_imgs)]
                ret_dicts = []
                for face_img, frame, keypoints, emb in zip(faces_imgs, frames, faces_keypoints, embs_):
                    ret_dict = {'req_id': req_id}
                    ret_dict['cid'] = client_id
                    ret_dict['sid'] = img_id
                    ret_dict['np_img_160'] = face_img
                    ret_dict['embedding'] = emb
                    ret_dict['rect'] = (frame[0], frame[1], frame[2], frame[3])
                    ret_dict['keypoints'] = {}
                    ret_dict['keypoints']['left_eye'] = (keypoints[0][0], keypoints[0][1])
                    ret_dict['keypoints']['right_eye'] = (keypoints[1][0], keypoints[1][1])
                    ret_dict['keypoints']['nose'] = (keypoints[2][0], keypoints[2][1])
                    ret_dicts.append(ret_dict)

                dict_stat['emb_begin_save_in_resulted_tm'] = time.time()
                dict_stat['emb_save_in_resulted_tm'] = time.time()

                #self.send_data_to_base( (req_id, client_id, img_id, ret_dicts, status, dict_stat) )
                self.flag += 1
                print(f'self.flag: {self.flag}')
                send_ok = self.send_data_to_base( (req_id, client_id, img_id, ret_dicts, status, dict_stat) )
                if send_ok:
                    #self.zmq_mon.send({'act': 'ok'}, msg_type='run')
                    os.utime('./send_ok_check', (time.time(), time.time()))
                #self.zmq_srv.send((req_id, client_id, img_id, ret_dicts, status, dict_stat), msg_type='send_embedded')
                print('RunnerEmbed: zmq_srv sended: req_id, client_id, img_id=', req_id, client_id, img_id, '  len(ret_dicts):', len(ret_dicts))

                shift += len(faces_imgs)
                if not (message_to_ack is None):
                    message_to_ack.ack()

        logger.error('Embedder runner stopped')
        print('STOP RunnerEmbed...')
        print('STOPped RunnerEmbed!')


def main():
    if len(sys.argv) != 2:
        print("""USAGE: runner_embed.py cnf_file_name
EXAMPLE: runner_embed.py config_all.txt""")
        exit(1)

    r = RunnerEmbed(sys.argv[1])
    r.run()

################################################################################

if __name__ == "__main__":
    main()

################################################################################
