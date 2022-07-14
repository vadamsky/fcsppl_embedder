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

from threading import Thread, Semaphore, get_ident, Lock
from queue import Queue
import shared_memory
from named_atomic_lock import NamedAtomicLock

from pipesworker import PipesWorker
from images_bytes_conversions import get_np_image_from_base64, get_base64_from_np_image, get_base64_from_np_array
from conversions import int_to_bytes, int_from_bytes # pylint: disable=E0401, C0413
from load_config import load_config

from constants import SHM_PRF
from shm_wrapper_t import ShmWrapperT


################################################################################

import logging

logger = logging.getLogger('ClientOut')
logger.setLevel(logging.INFO)

# Create file handler
logger_handler = logging.FileHandler('./logs/client_out.log')
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

def make_request(EMBEDDER_HOST, post_body, headers):
    response = None
    try:
        # self.EMBEDDER_HOST = '95.216.44.199:8002'
        response = requests.post(EMBEDDER_HOST, headers=headers, data=post_body, timeout=5)
        response.raise_for_status()
    except requests.exceptions.HTTPError as errh:
        print('errh:', errh)
        return False
    except requests.exceptions.ConnectionError as errc:
        print('errc:', errc)
        return False
    except requests.exceptions.Timeout as errt:
        print('errt:', errt)
        return False
    except requests.exceptions.RequestException as err:
        print('err:', err)
        return False
    #
    jsn = None
    if not (response is None):
        if response.status_code != 200:
            print("Result not found!")
            return False
        else:  # 200
            answer = response.content
            jsn = json.loads(answer)
            return True
    return False

def send_thread_func(q, EMBEDDER_HOST, imgs_dt_lock, imgs_dt_flags):
    while True:
        try:
            item = q.get(block=False)
            (req_id, client_id, img_id, faces_imgs, frames, faces_keypoints, status, dict_stat) = item
            dict_stat['det_get_from_internal_queue_tm'] = time.time()

            if status == 'OK':
                post_body = pickle.dumps( (req_id, client_id, img_id, faces_imgs, frames, faces_keypoints, status, dict_stat) )
                headers = requests.structures.CaseInsensitiveDict([
                    ('UNO-DETECTION-OPTION', 'yes')
                    ])
                ok = make_request(EMBEDDER_HOST, post_body, headers)
                if ok:
                    q.task_done()
        except Exception as e:
            #print('ClientOut: send_thread_func:', e)
            time.sleep(1)
        #dict_stat['det_detected_tm'] = time.time()


class ClientOut(PipesWorker):
    def __init__(self, json_name):
        global pipe_in_name, pipe_out_name
        embedder_id = 0
        CONF_DCT = load_config(json_name)
        self.DET_PROCS = CONF_DCT['DET_PROCS']
        self.IMAGE_SIZE = CONF_DCT['IMAGE_SIZE']
        self.QUEUES_DIR = CONF_DCT['QUEUES_DIR']
        self.EMBEDDER_BATCH = CONF_DCT['EMBEDDER_BATCH']
        self.EMBEDDER_HOST = CONF_DCT['EMBEDDER_HOST']
        self.SEND_THREADS_CNT = 5
        self.send_threads = []
        PipesWorker.__init__(self, '', '', blocked=False)

        self.last_loaded_id = 0

        self.qlen = self.SEND_THREADS_CNT * 2
        self.q = Queue(self.qlen)
        self.imgs_dt_flags = []
        self.imgs_dt_lst = []
        self.imgs_dt_len = 0
        self.imgs_dt_lock = Lock()

        self.shm_a = shared_memory.SharedMemory2(name=SHM_PRF+'a') # last_handled_id, last_saved_id, reserve
        self.shm_a_lock = NamedAtomicLock(SHM_PRF+'a_lock')
        self.zmq_mon = ShmWrapperT('client_out', 'monitor', 4096)
        self.start_string = 'python3 src/client_out.py %s' % (json_name)

        #self.zmq_srv = ShmWrapperT('client_out', 'server3_in', 1024*1024)
        self.zmq_embs = []
        for detector_id in range(self.DET_PROCS):
            zmq_emb = ShmWrapperT('detect_thread_%d' % detector_id, 'client_out', 256*1024)
            self.zmq_embs.append(zmq_emb)

        for _ in range(self.SEND_THREADS_CNT):
            send_thread = Thread(target=send_thread_func, args=(self.q, self.EMBEDDER_HOST, self.imgs_dt_lock, self.imgs_dt_flags, ))
            send_thread.start()
            self.send_threads.append(send_thread)

        self.ids = np.zeros((self.EMBEDDER_BATCH), dtype=np.int64) # ids to copy from shm_b to shm_c
        self.cids = np.zeros((self.EMBEDDER_BATCH), dtype=np.int64) # client ids to copy from shm_b to shm_c
        self.sids = [b''] * self.EMBEDDER_BATCH # image string ids to copy from shm_b to shm_c
        self.imgs = np.zeros((self.EMBEDDER_BATCH, self.IMAGE_SIZE, self.IMAGE_SIZE, 3), dtype=np.uint8) # imgs to embed
        self.rect_kps = np.zeros((self.EMBEDDER_BATCH, 13), dtype=np.float64) # rect4 + keypoint6 + agegender3


    def need_stop_all_processes(self):
        stop_proc_flag = int.from_bytes(self.shm_stop.buf[0:8], 'big')
        return bool(stop_proc_flag)

    def zmq_callback_func(self, dt):
        (identifier, tm, msg_type, data) = dt
        if data:
            (req_id, client_id, img_id, faces_imgs, frames, faces_keypoints, status, dict_stat) = data
            print('ClientOut: zmq_callback_func:  self.imgs_dt_len=', self.imgs_dt_len, '  req_id, client_id, img_id:', req_id, client_id, img_id, '  len(frames):', len(frames))
            self.imgs_dt_lock.acquire()
            self.imgs_dt_flags.append('new')
            self.imgs_dt_lst.append((req_id, client_id, img_id, faces_imgs, frames, faces_keypoints, status, dict_stat))
            self.imgs_dt_len += 1
            self.shm_a.buf[4*8:4*8+8] = self.imgs_dt_len.to_bytes(8, 'big')
            self.imgs_dt_lock.release()

    #def send_data_to_embedder(self, tpl):
    #    # tpl is: (req_id, client_id, img_id, faces_imgs, frames, faces_keypoints, status, dict_stat)
    #    post_body = pickle.dumps(tpl)
    #    headers = requests.structures.CaseInsensitiveDict([
    #        ('UNO-DETECTION-OPTION', 'yes')
    #        ])
    #    ret = make_request(post_body, headers)

    def run(self):
        for zmq_emb in self.zmq_embs:
            zmq_in_thr = Thread(target=zmq_emb.run_serv, args=(self.zmq_callback_func, ))
            zmq_in_thr.start()
        # Main thread
        while True:
            if self.need_stop_all_processes():
                break

            self.zmq_mon.send({'act': 'running', 'start_string': self.start_string}, msg_type='run')
            # load files
            loads = []
            statuses = []
            self.imgs = []

            if self.imgs_dt_len and (self.q.qsize() < self.qlen):
                self.imgs_dt_lock.acquire()
                (req_id, client_id, img_id, faces_imgs, frames, faces_keypoints, status, dict_stat) = self.imgs_dt_lst[0]
                dict_stat['emb_load_from_detected_tm'] = time.time()
                dict_stat['detected_queue_len'] = self.imgs_dt_len
                self.imgs_dt_lst = self.imgs_dt_lst[1:]
                self.imgs_dt_len -= 1
                self.imgs_dt_lock.release()
                try:
                    self.q.put((req_id, client_id, img_id, faces_imgs, frames, faces_keypoints, status, dict_stat), block=False)
                except Exception as e:
                    print('ClientOut: run:', e)
                else:
                    pass
            
            """
            #if self.imgs_dt_len:
            while self.imgs_dt_len:
                self.imgs_dt_lock.acquire()
                (req_id, client_id, img_id, faces_imgs, frames, faces_keypoints, status, dict_stat) = self.imgs_dt_lst[0]
                dict_stat['emb_load_from_detected_tm'] = time.time()
                dict_stat['detected_queue_len'] = self.imgs_dt_len
                self.imgs_dt_lst = self.imgs_dt_lst[1:]
                self.imgs_dt_len -= 1
                self.imgs_dt_lock.release()

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

            embs = None
            if len(self.imgs) == 0:
                print('RunnerEmbed:  len(self.imgs) == 0')
                print('RunnerEmbed:  len(loads)=', len(loads))
                time.sleep(0.5)
                continue
            else:
                # get embs
                #self.zmq_mon.send({'act': 'pre get_embs', 'len(self.imgs)': len(self.imgs)}, msg_type='run')
                try:
                    embs = self.get_embs(self.imgs, 0)
                except Exception:
                    print('Error with  embs = self.get_embs(self.imgs, 0)')
                    logger.error('Embedder get_embs err')
                #self.zmq_mon.send({'act': 'post get_embs', 'embs is None': (embs is None)}, msg_type='run')

            # save
            shift = 0
            for load, status in zip(loads, statuses):
                (req_id, client_id, img_id, faces_imgs, frames, faces_keypoints, dict_stat) = load
                embs_ = embs[shift:shift+len(faces_imgs)]
                ret_dicts_ = ret_dict_s[shift:shift+len(faces_imgs)]
                ret_dicts = []
                for face_img, frame, keypoints, emb, ret_dict_ in zip(faces_imgs, frames, faces_keypoints, embs_, ret_dicts_):
                    ret_dict = ret_dict_.copy()
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
                #os.remove(fpath_in)

                fpath = self.QUEUES_DIR + 'resulted/%.24ld_%.4d.pkl' % (req_id, client_id)
                dict_stat['emb_save_in_resulted_tm'] = time.time()
                self.zmq_srv.send((req_id, client_id, img_id, ret_dicts, status, dict_stat), msg_type='send_embedded')
                print('Runner_V3_Embed: zmq_srv sended: req_id, client_id, img_id=', req_id, client_id, img_id, '  len(ret_dicts):', len(ret_dicts))

                #with open(fpath, 'wb') as f:
                #    pickle.dump((ret_dicts, status, dict_stat), f)
                shift += len(faces_imgs)
            """

        logger.error('ClientOut stopped')
        print('STOP ClientOut...')
        print('STOPped ClientOut!')


def main():
    if len(sys.argv) != 2:
        print("""USAGE: client_out.py cnf_file_name
EXAMPLE: client_out.py config_all.txt""")
        exit(1)

    r = ClientOut(sys.argv[1])
    r.run()

################################################################################

if __name__ == "__main__":
    main()

################################################################################
