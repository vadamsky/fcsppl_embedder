#!/usr/bin/env python3

import sys
import os
import os.path
import pathlib
import shutil
import pickle
from shutil import copyfile

import numpy as np

from conversions import int_to_bytes, int_from_bytes
from detector_base2 import detect_Blaze, DetectorBase

from FaceDetectorBlazeFacePytorchGPU import FaceDetectorBlazeFaceGPU # type: ignore # pylint: disable=E0401, C0413

#from zmq_wrapper import ZmqWrapper, MON_ADDR


class DetectorGPUB(DetectorBase):
    def __init__(self, json_name, number_of_detector, videocard_number=0):
        super(DetectorGPUB, self).__init__(json_name, number_of_detector)

        device = 'cpu'
        if videocard_number >= 0:
            device = 'cuda:%d' % videocard_number
        print('detector device = ', device)
        self.fcdet = FaceDetectorBlazeFaceGPU(device = device)


def main():
    if len(sys.argv) != 4:
        print("""USAGE: detector_GPUB.py cnf_file_name number_of_detector videocard_number
EXAMPLE: detector_GPUB.py 2 0""")
        exit(1)

    print('dGPU = DetectorGPUB(%s, %d, %d)' % (sys.argv[1], int(sys.argv[2]), int(sys.argv[3])))
    dGPU = DetectorGPUB(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
    dGPU.run()

################################################################################

if __name__ == "__main__":
    main()

################################################################################
