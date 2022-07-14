#!/usr/bin/env python3

import time
import sys
import os
import json
import pickle
import math

import numpy as np
from Facenet import loadModel

#from named_atomic_lock import NamedAtomicLock
from base_class import BaseClass
from zmq_wrapper import ZmqWrapper

from constants import FACENET_EMB_ADDR, QUEUE_OUT_ADDR, BATCH_FACENET_EMB
from constants import shift_pad_and_resize_image


class FacenetEmbedder(BaseClass):
    def __init__(self):
        self.pack_number = 0
        self.prebatch = []
        # init embedder
        self.df_model = loadModel()
        #
        self.zmq_this = ZmqWrapper('', addr=FACENET_EMB_ADDR, tp='server')
        self.zmq_out = ZmqWrapper('facenet_embedder', addr=QUEUE_OUT_ADDR, tp='client')
        self.zmq_this.run_serv(self.zmq_handle_func)


    def zmq_handle_func(self, dt):
        (identifier, tm, msg_type, data) = dt
        (img_id, det_imgs_271_025, faces) = data
        print('FacenetEmbedder:', img_id)

        i = 0
        for det_271_025 in det_imgs_271_025:
            img_to_embed = shift_pad_and_resize_image(det_271_025, padding=0.06, shift_img=0.05, sz=160)
            self.prebatch.append((img_id, img_to_embed, faces[i]))
            i += 1

        #if len(self.prebatch) >= INSIGHT_EMB_ADDR:
        self.embed_imgs()


    def embed_imgs(self):
        embeddings = []
        while len(self.prebatch):
            batch = self.prebatch #self.prebatch[:BATCH_FACENET_EMB] if len(self.prebatch)>=BATCH_FACENET_EMB else self.prebatch
            imgs = np.zeros((len(batch), 160, 160, 3))
            i = 0
            for batc in batch:
                img = batc[1]
                img = img / 255
                img = np.reshape(img,  [1, 160, 160, 3]) # return the image with shaping that TF wants.
                #img_pixels  = img#image.img_to_array(img) #what this line doing? must?
                imgs[i] = img
                i += 1
            embeddings = self.df_model.predict(imgs)

            #embed = {'img_ids': [], 'imgs_to_embed': [], 'faces': [], 'embeddings': []}
            #for preb, embedding in self.prebatch, embeddings:
            #    embed['img_ids'].append(preb[0])
            #    embed['imgs_to_embed'].append(preb[1])
            #    embed['faces'].append(preb[2])
            #    embed['embeddings'].append(embedding)

            embed = []
            for preb, embedding in self.prebatch, embeddings:
                embed.append((preb[0], preb[1], preb[2], embedding))
            
            #self.prebatch = self.prebatch[BATCH_FACENET_EMB:] if len(self.prebatch)>=BATCH_FACENET_EMB else []
            self.zmq_out.send(embed, msg_type='facenet')
            self.prebatch = []


def main():
    if len(sys.argv) != 2:
        print("""USAGE: facenet_embedder.py <vcard_number>""")
        exit(1)

    r = FacenetEmbedder()

################################################################################

if __name__ == "__main__":
    main()

################################################################################
