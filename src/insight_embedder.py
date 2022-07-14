#!/usr/bin/env python3

from constants import INSIGHT_EMB_ADDR, QUEUE_OUT_ADDR, BATCH_INSIGHT_EMB
from constants import shift_pad_and_resize_image

import time
import sys
import os
import json
import pickle
import math

import numpy as np
from numpy.linalg import norm
#import insightface
from model_zoo import get_model


#from named_atomic_lock import NamedAtomicLock
from base_class import BaseClass
from zmq_wrapper import ZmqWrapper


class InsightEmbedder(BaseClass):
    def __init__(self):
        self.pack_number = 0
        self.prebatch = []
        # init embedder
        #self.if_model = insightface.app.FaceAnalysis(ga_name=None) # det_name=None, rec_name=None
        self.if_model = get_model('arcface_r100_v1')
        #self.if_model.prepare(0, nms=0.4)
        self.if_model.prepare(0)
        #
        self.zmq_this = ZmqWrapper('', addr=INSIGHT_EMB_ADDR, tp='server')
        self.zmq_out = ZmqWrapper('insight_embedder', addr=QUEUE_OUT_ADDR, tp='client')
        self.zmq_this.run_serv(self.zmq_handle_func)


    def zmq_handle_func(self, dt):
        (identifier, tm, msg_type, data) = dt
        (img_id, det_imgs_271_025, faces) = data
        print('InsightEmbedder:', img_id)

        i = 0
        for det_271_025 in det_imgs_271_025:
            img_to_embed = shift_pad_and_resize_image(det_271_025, padding=0.06, shift_img=0.05, sz=112)
            self.prebatch.append((img_id, img_to_embed, faces[i]))
            i += 1

        #if len(self.prebatch) >= INSIGHT_EMB_ADDR:
        self.embed_imgs()


    def embed_imgs(self):
        embeddings = []
        while len(self.prebatch):
            batch = self.prebatch #self.prebatch[:INSIGHT_EMB_ADDR] if len(self.prebatch)>=INSIGHT_EMB_ADDR else self.prebatch
            for batc in batch:
                img = batc[1]
                # run batch embed
                #embedding = self.if_model.rec_model.get_embedding(img).flatten()
                embedding = self.if_model.get_embedding(img).flatten()
                embedding_norm = norm(embedding)
                embedding = embedding / embedding_norm
                embeddings.append(embedding)
            #

            #embed = {'img_ids': [], 'imgs_to_embed': [], 'faces': [], 'embeddings': []}
            #for preb, embedding in self.prebatch, embeddings:
            #    embed['img_ids'].append(preb[0])
            #    embed['imgs_to_embed'].append(preb[1])
            #    embed['faces'].append(preb[2])
            #    embed['embeddings'].append(embedding)

            embed = []
            for preb, embedding in self.prebatch, embeddings:
                embed.append((preb[0], preb[1], preb[2], embedding))

            #self.prebatch = self.prebatch[BATCH_INSIGHT_EMB:] if len(self.prebatch)>=BATCH_INSIGHT_EMB else []
            self.zmq_out.send(embed, msg_type='insight')
            self.prebatch = []


def main():
    if len(sys.argv) != 2:
        print("""USAGE: insight_embedder.py <vcard_number>""")
        exit(1)

    r = InsightEmbedder()

################################################################################

if __name__ == "__main__":
    main()

################################################################################
