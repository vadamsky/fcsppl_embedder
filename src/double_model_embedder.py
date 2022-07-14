#!/usr/bin/env python3

from constants import INSIGHT_EMB_ADDR, QUEUE_OUT_ADDR, BATCH_INSIGHT_EMB
from constants import shift_pad_and_resize_image

import time
import glob
import sys
import os
import json
import pickle
import math

from threading import Thread, Lock
import multiprocessing

import numpy as np
from numpy.linalg import norm
from Double import loadModel

#from named_atomic_lock import NamedAtomicLock
from base_class import BaseClass
from zmq_wrapper import ZmqWrapper


class DoubleModelEmbedder(BaseClass):
    def __init__(self):
        self.save_id = 1

        #self.model = loadModel()
        self.model = loadModel('./weights/glint360k_cosface_r50_fp16_0.1.h5')
        self.zmq_this = ZmqWrapper('', addr=INSIGHT_EMB_ADDR, tp='server')
        self.zmq_out = ZmqWrapper('double_embedder', addr=QUEUE_OUT_ADDR, tp='client')
        self.zmq_this.run_serv(self.zmq_handle_func)


    def zmq_handle_func(self, dt):
        (identifier, tm, msg_type, data) = dt
        (img_id, det_imgs_271_025, faces) = data
        print('DoubleModelEmbedder:', img_id)
        #
        img_dct = {'img_id': img_id, 'detected': []}
        i = 0
        for det_271_025 in det_imgs_271_025:
            img_to_embed = shift_pad_and_resize_image(det_271_025, padding=0.06, shift_img=0.05, sz=160)
            img_dct['detected'].append((img_id, img_to_embed, faces[i]))
            i += 1
        #
        batch = img_dct['detected']
        imgs = np.zeros([len(batch), 160, 160, 3])
        embeddings = []
        # doing insight embed
        i = 0
        for batc in batch:
            img = batc[1]
            img = np.reshape(img,  [1, 160, 160, 3])
            imgs[i] = img
            #embedding = if_model.get_embedding(img).flatten()
            #embedding_norm = norm(embedding)
            #embedding = embedding / embedding_norm
            #embeddings.append(embedding)
            i += 1
        embeddings = self.model.predict([imgs, imgs])
        img_dct['embedded_double'] = [emb for emb in embeddings]
        #
        self.zmq_out.send(img_dct, msg_type='double_model')


def main():
    if len(sys.argv) != 2:
        print("""USAGE: double_model_embedder.py <vcard_number>""")
        exit(1)

    r = DoubleModelEmbedder()

################################################################################

if __name__ == "__main__":
    main()

################################################################################
