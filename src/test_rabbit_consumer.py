from rabbit_wrapper import RabbitWrapper

from constants import RABBIT_IN_ADDR, RABBIT_IN_QNAME, RABBIT_IN_EXCH, RABBIT_IN_RKEY
#from constants import RABBIT_OUT_ADDR, RABBIT_OUT_QNAME, RABBIT_OUT_EXCH, RABBIT_OUT_RKEY



import os
import base64
import pathlib
import time
import sys
import json
import pickle
import math
#import numpy as np # type: ignore
#import cv2 # type: ignore
#from io import BytesIO
#from PIL import Image # type: ignore


def get_np_image_from_bytes(byte_img: bytes):
    nparr = np.fromstring(byte_img, np.uint8)
    np_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)[..., ::-1]
    return np_image

def get_np_image_from_base64(base64_str: str):
    byte_img = base64.b64decode(base64_str)
    nparr = np.frombuffer(byte_img, np.uint8)
    np_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)[..., ::-1]
    return np_img

def get_base64_from_np_image(np_img, image_quality=90):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), image_quality]
    _, nparr = cv2.imencode('.jpg', np_img[..., ::-1], encode_param)
    return base64.b64encode(nparr.tobytes())

def get_base64_from_np_array(nparr):
    return base64.b64encode(nparr.tobytes())

def get_np_array_from_base64(base64_str: str):
    byte_ = base64.b64decode(base64_str)
    return np.frombuffer(byte_, np.uint8)

def load_image_in_bytes(path):
    """
    Loads image and return it bytes data.
    Args:
        path: Image path.
    Returns:
        Image byte array.
    """
    byte_img_io = BytesIO()
    byte_img = Image.open(path)
    byte_img.save(byte_img_io, "JPEG")
    byte_img_io.seek(0)
    byte_img = byte_img_io.read()
    return byte_img



def handle_rabbit_message(body):
    json_str = body
    jsn = json.loads(json_str)
    img_id = jsn['img_id']
    img_base64_str = jsn['img_base64']
    img_base64_bts = img_base64_str.encode('ascii')
    img_body = base64.b64decode(img_base64_bts)
    nparr = np.frombuffer(img_body, np.uint8)
    np_image = cv2.imdecode(nparr, -1)[..., ::-1]
    print('QueueIn:', img_id)
    #self.zmq_runner_detect.send((img_id, np_image))

rabbit = RabbitWrapper(tp='in', rabt_addr=RABBIT_IN_ADDR, exch_nm=RABBIT_IN_EXCH,
                            q_nm=RABBIT_IN_QNAME, r_key=RABBIT_IN_RKEY, handle_message=handle_rabbit_message)
print('QueueIn __init__ ok')
