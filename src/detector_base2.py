#!/usr/bin/env python3

import time
import datetime
import sys
import os
import os.path
import json
import pathlib
import shutil
import pickle
from shutil import copyfile
import shared_memory
from constants import SHM_PRF

import numpy as np

from pipesworker import PipesWorker
from load_config import load_config
from images_bytes_conversions import get_np_image_from_base64, get_base64_from_np_image, get_base64_from_np_array
from conversions import int_to_bytes, int_from_bytes
from constants import RUNNED, NEED_STOP_PROC, NEED_STOP_THRD, STOPPED
from logger import log_in_file

#from zmq_wrapper import ZmqWrapper, MON_HOST
from shm_wrapper_t import ShmWrapperT

pipe_out_name = './pipes/pipe_detector_main_'
pipe_in_name = './pipes/pipe_main_detector_'


################################################################################

import logging

logger = logging.getLogger('DetectorBase2')
logger.setLevel(logging.INFO)

# Create file handler
logger_handler = logging.FileHandler('./logs/detector.log')
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


def detect_MTCNN(fcdet, imgs, Smin, size, padding):
    det_imgs = []
    det_xywhs = []
    det_inpaths = []
    for img in imgs:
        try:
            xywhs, faces, fo_detections, facepointss = fcdet.get_detected_faces_xywhs(img.copy())
            for xywh, face, fo_detection, facepoints in zip(xywhs, faces, fo_detections, facepointss):
                if xywh[2] * xywh[3] >= Smin:
                    det_imgs.append(fcdet.rotate(img, face, size, padding))
                    det_xywhs.append(xywh)
        except:
            logger.error('    Error detecting image')
    return det_imgs, det_xywhs

def detect_Blaze(fcdet, imgs, Smin, size, padding, precisedetection):
    det_imgs = []
    det_xywhs = []
    det_keypoints = []
    for img in imgs:
        try:
            a = datetime.datetime.now()
            #pickle.dump(img, open('img.pkl', 'wb'))
            xywhs, faces, fo_detections, facepointss = fcdet.get_detected_faces_xywhs(img.copy(), Smin, precisedetection)
            b = datetime.datetime.now()
            #try:
            (det_imgs_, keypoints_160_) = fcdet.rotate_pack(img, xywhs, faces, size, padding, Smin)
            c = datetime.datetime.now()
            for det_img, xywh, keypoints_160 in zip(det_imgs_, xywhs, keypoints_160_):
                #pickle.dump(det_img, open('det_img.pkl', 'wb'))
                if xywh[2] * xywh[3] >= Smin:
                    det_imgs.append(det_img)
                    det_xywhs.append(xywh)
                    det_keypoints.append(keypoints_160)
            d = datetime.datetime.now()
            print(b - a, c - b, d - c)
        except Exception as e:
            logger.error(f'    Error detecting image {e}')
            print('    Error detecting image ', e)
            print(det_imgs, det_xywhs, det_keypoints)
    return det_imgs, det_xywhs, det_keypoints


class DetectorBase(PipesWorker):
    def __init__(self, json_name, number_of_detector):
        global pipe_in_name, pipe_out_name
        CONF_DCT = load_config(json_name)
        self.Smin = CONF_DCT['S_MIN']
        self.size = CONF_DCT['IMAGE_SIZE']
        self.IMAGE_SIZE = self.size
        self.padding = CONF_DCT['PADDING']
        self.QUEUES_DIR = CONF_DCT['QUEUES_DIR']
        self.N_GPUS = CONF_DCT['N_GPUS']
        pipe_in_name  = pipe_in_name.replace('./pipes', self.QUEUES_DIR+'pipes')
        pipe_out_name = pipe_out_name.replace('./pipes', self.QUEUES_DIR+'pipes')
        PipesWorker.__init__(self, pipe_in_name + '%d' % number_of_detector,
                                   pipe_out_name + '%d' % number_of_detector,
                                   blocked=False, proc_num=number_of_detector)

        self.number_of_detector = number_of_detector
        detector_id = number_of_detector
        self.detector_id = detector_id

        #if self.detector_id == 0:
        #    log_in_file('detector0.csv', 'Detector0 base was pre-started')
        #    print('Detector0 was pre-started')
        self.shm_a = shared_memory.SharedMemory2(name=SHM_PRF+'a')
        self.zmq_mon = ShmWrapperT(f'detector_{detector_id}', 'monitor', 4096)
        self.zmq_detector  = ShmWrapperT(f'detect_thread_{detector_id}', f'detector_{detector_id}', 4*1024*1024)
        self.zmq_emb = ShmWrapperT(f'detect_thread_{detector_id}', f'client_out', 1024*1024)
        self.fcdet = None
        #device = videocard_number
        #if type_of_net == 'PT':
        #    device = 'cpu'
        #    if videocard_number >= 0:
        #        device = 'cuda:%d' % videocard_number
        #self.fcdet = FaceDet(device)
        gpu_num = 0 if self.N_GPUS <= 1 else number_of_detector % (self.N_GPUS - 1)
        self.start_string = 'python3 src/detector_GPUB.py %s %d %d' % (json_name, number_of_detector, gpu_num)
        #self.start_time = time.time()
        #if self.detector_id == 0:
        #    log_in_file('detector0.csv', 'Detector0 base was started')
        #    print('Detector0 was started')


    def run(self):
        #if self.detector_id == 0:
        #    log_in_file('detector0.csv', 'Detector0 run started')
        image_np = np.zeros((self.size, self.size, 3))
        status, det_imgs, det_xywhs = '', [], [] ####
        zero = 0
        self.shm_a.buf[(6+self.detector_id)*8:(6+self.detector_id)*8+8] = zero.to_bytes(8, 'big')
        #
        while True:
            if self.need_stop_all_processes() or self.need_stop_this_process():
                break
            #if self.detector_id == 0:
            #    log_in_file('detector0.csv', 'Detector0 run')
            #    print('Detector0 run')

            status = 'OK'
            log_in_file('detector.csv', '1:')

            # READ
            (identifier, tm, msg_type, ret) = self.zmq_detector.read()
            (req_id, client_id, img_id, image_np, dict_stat, precisedetection) = None, None, None, None, {}, False
            if ret:
                (req_id, client_id, img_id, image_np, dict_stat, precisedetection) = ret
                dict_stat['det_detector_loaded_tm'] = time.time()
            print(f'detector_base2: run:  self.detector_id={self.detector_id}  req_id={req_id}')
            #if self.detector_id == 0:
            #    log_in_file('detector0.csv', f'req_id={req_id}')
            #    print('Detector0 req_id={req_id}')
            if req_id is None:  # req_id may be 0
                self.shm_a.buf[(6+self.detector_id)*8:(6+self.detector_id)*8+8] = zero.to_bytes(8, 'big')
                continue  # self.zmq_detector2.send(None)

            # PING & DETECT
            det_imgs, det_xywhs, det_keypoints = [], [], []
            if image_np is None:
                self.zmq_mon.send({'act': 'img from pipe received', 'req_id': req_id, 'image_np.shape': (0,0,3), 'det_id': self.number_of_detector, 'start_string': self.start_string}, msg_type='run')
            else:
                self.zmq_mon.send({'act': 'img from pipe received', 'req_id': req_id, 'image_np.shape': image_np.shape, 'det_id': self.number_of_detector, 'start_string': self.start_string}, msg_type='run')
                try:
                    det_imgs, det_xywhs, det_keypoints = detect_Blaze(self.fcdet, [image_np], self.Smin, self.size, self.padding, precisedetection)
                except Exception as e:
                    log_in_file('detector.csv', f'{req_id};ERROR in detect_Blaze {e}')
                    print(f'detector_base2  req_id={req_id}  error in detect_Blaze {e}')
                #
                #if (time.time() - self.start_time) > 60 and self.detector_id == 0:
                #    break
            #if self.detector_id == 0:
            #    log_in_file('detector0.csv', f'len(det_imgs)={len(det_imgs)}')
            #    print('Detector0 len(det_imgs)={len(det_imgs)}')

            log_in_file('detector.csv', '3: req_id=%d' % req_id)
            self.zmq_mon.send({'act': 'img handled', 'req_id': req_id, 'det_id': self.number_of_detector, 'start_string': self.start_string}, msg_type='run')
            print(f'detector_base2: run:  self.detector_id={self.detector_id}  len(det_imgs)={len(det_imgs)}')
            dict_stat['det_detected_tm'] = time.time()

            # SEND
            faces_imgs, frames, faces_keypoints = det_imgs, det_xywhs, det_keypoints
            if ret != 'Previous status was bad' and len(faces_imgs) == 0:
                status = 'No detected faces'
            if ret == 'Previous status was bad' or len(faces_imgs) == 0:
                faces_imgs = [np.zeros((self.IMAGE_SIZE, self.IMAGE_SIZE, 3), dtype=np.uint8)]
                frames = [(0, 0, 0, 0)]
                faces_keypoints = [[[0, 0], [0, 0], [0, 0]]]
            start_tm = time.time()
            dict_stat['det_save_in_detected_tm'] = time.time()
            self.zmq_emb.send((req_id, client_id, img_id, faces_imgs, frames, faces_keypoints, status, dict_stat), msg_type='send_detected')
            self.zmq_mon.send({'act': 'img returned', 'req_id': req_id, 'det_id': self.number_of_detector, 'start_string': self.start_string}, msg_type='run')

            if ret is None:
                log_in_file('detector.csv', 'EXIT CYCLE')
                logger.error('    EXIT CYCLE')
                print('detector_base2  ret is None error')
                break
            self.shm_a.buf[(6+self.detector_id)*8:(6+self.detector_id)*8+8] = zero.to_bytes(8, 'big')

        # Stop detector
        log_in_file('detector0.csv', 'Detector 0 was stopped')
        log_in_file('detector.csv', 'Detector %d has stopped' % self.number_of_detector)
        print('detector.csv', 'Detector %d has stopped' % self.number_of_detector)
