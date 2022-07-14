import math
import numpy as np
import cv2

RUNNED, NEED_STOP_PROC, NEED_STOP_THRD, STOPPED = 4, 2, 1, 0
WAIT_DET_PROC_RESP = 6

STRID_SZ = 256
INT64_SZ = 8
FLOAT64_SZ = 8

IMAGE_WIDTH = 160 # 160*160 image
IMAGE_CHANNELS = 3 # RGB
IMAGE_SIZE = IMAGE_WIDTH * IMAGE_WIDTH * IMAGE_CHANNELS # uint8
EMBEDDING_SIZE = 512 * FLOAT64_SZ # 512-D vector

SHM_PRF = 'shm__'


BATCH_FACENET_EMB = 4
BATCH_INSIGHT_EMB = 4

EMB_ADDR = "95.216.44.199:43000"
#DETECTOR_HOST = "tcp://127.0.0.1"
#DETECTOR_PORT = 43003

# 'amqp://guest:guest@localhost:5672//'
#RABBIT_IN_ADDR = "amqp://guest2:guest2@95.216.44.199:5672/"
RABBIT_IN_ADDR = "amqp://guest2:guest2@95.217.34.26:5672/"
#RABBIT_OUT_ADDR = 
RABBIT_IN_EXCH = "in_exchange"
#RABBIT_OUT_EXCH = RABBIT_IN_EXCH # "detected-exchange"
RABBIT_IN_QNAME = "images"
#RABBIT_OUT_QNAME = "detected"
RABBIT_IN_RKEY = ""
#RABBIT_OUT_RKEY = RABBIT_IN_RKEY # "test"


proc_lst = \
[#'python3 detector.py 1 0',
'python3 detector.py 0 0',
'python3 runner_detect.py 1',
#'python3 facenet_embedder.py 0',
#'python3 insight_embedder.py 0',
#'python3 double_changer_embedder.py 0',
'python3 double_model_embedder.py 0', # 6.293Gi/7.929Gi with one detector
'python3 queue_out.py',
'python3 queue_in.py',
]


def shift_pad_and_resize_image(img, padding, shift_img, sz=112):
    pad = int(img.shape[0] * (0.25 - padding) / (1 + 0.25 * 2))
    new_sz = img.shape[0] - 2 * pad
    shf = int(new_sz * shift_img * (1 + 0.25 * 2) / (1 + padding * 2))
    img = img.copy()
    if shf < 0:
        line = img[-1 ,:,:].copy()
        img[0:shf ,:,:] = img[-shf: ,:,:]
        for ii in range(-shf):
            img[shf+ii ,:,:] = line
    if shf > 0:
        line = img[0 ,:,:].copy()
        img[shf: ,:,:] = img[0:-shf ,:,:]
        for ii in range(shf):
            img[ii ,:,:] = line

    if pad != 0:
        img = img[pad:-pad, pad:-pad]
    img = cv2.resize(img, (sz, sz))
    return img
