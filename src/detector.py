#!/usr/bin/env python3

from constants import QUEUE_OUT_ADDR
from constants import FACENET_EMB_ADDR, INSIGHT_EMB_ADDR
from constants import calc_detector_addr

import time
import sys
import os
import json
import pickle
import math

import numpy as np
import cv2

from base_class import BaseClass
from zmq_wrapper import ZmqWrapper
from FaceDetectorInsight import FaceDetectorInsight


def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()


class Detector(BaseClass):
    def __init__(self, detector_num=0, vcard_num=0, batch=1):
        self.batch = batch
        self.zmq_this    = ZmqWrapper(identifier='',
                                      addr=calc_detector_addr(detector_num + 1), tp='server')
        self.zmq_facenet = ZmqWrapper(identifier='detector_%d' % detector_num,
                                      addr=FACENET_EMB_ADDR, tp='client')
        self.zmq_insight = ZmqWrapper(identifier='detector_%d' % detector_num,
                                      addr=INSIGHT_EMB_ADDR, tp='client')
        self.zmq_out     = ZmqWrapper(identifier='insight_embedder',
                                      addr=QUEUE_OUT_ADDR, tp='client')
        self.FaceDetectorInsight = FaceDetectorInsight()
        #
        self.zmq_this.run_serv(self.zmq_handle_func)


    def zmq_handle_func(self, dt):
        (identifier, tm, msg_type, data) = dt
        (img_id, np_image) = data
        print('Detector:', img_id)

        det_imgs_271_025_, faces_ = self.FaceDetectorInsight.get_one_img_result(np_image, 1200, 271, 0.25)

        # test on blur
        det_imgs_271_025, faces = [], []
        for det_img, face in zip(det_imgs_271_025_, faces_):
            if variance_of_laplacian(det_img) >= 80:
                det_imgs_271_025.append(det_img)
                faces.append(face)

        if len(faces):
            #self.zmq_facenet.send((img_id, det_imgs_271_025, faces))
            self.zmq_insight.send((img_id, det_imgs_271_025, faces))
        else:
            # data: {'img_id': img_id, 'detected':[('img_id, img_to_embed, face)], 'embedded_insight':[embedding], 'embedded_facenet':[embedding]}
            self.zmq_out.send( {'img_id': img_id, 'detected': [], 'embedded_insight':[], 'embedded_facenet':[]} , msg_type='detector')

    #def run(self):
    #    # MAIN CYCLE
    #    while True:
    #        if self.need_stop_all_processes():
    #            break



def main():
    if len(sys.argv) != 3:
        print("""USAGE: detector.py detector_num vcard_num
EXAMPLE: detector.py 1 0""")
        exit(1)

    r = Detector(int(sys.argv[1]), int(sys.argv[1]))
    r.run()

################################################################################

if __name__ == "__main__":
    main()

################################################################################
