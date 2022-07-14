# -*- coding: utf-8 -*-
"""Functions for working with bytes and images.
"""

import os
import base64
import pathlib

from io import BytesIO
from PIL import Image # type: ignore
import cv2 # type: ignore
import numpy as np # type: ignore

from conversions import int_from_bytes, int_to_bytes


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

def save_images_from_bytes(token: str, images_bytess: bytes, dir_path: str = './data/'):
    """
    Decodes several incoming images from client from bytes and saves their in token subdir.
    Args:
        token: Token string.
        images_bytess: Bytes with images data.
        dir_path: Parent directory to creating token subdir.
    Returns:
        None.
    """
    pathlib.Path(os.path.join(dir_path, token)).mkdir(parents=True, exist_ok=True)
    images_number = int_from_bytes(images_bytess[:4])
    shift = 4
    for num in range(images_number):
        img_len = int_from_bytes(images_bytess[shift:shift + 4])
        shift += 4
        byte_img = images_bytess[shift:shift + img_len]
        shift += img_len
        img = Image.open(BytesIO(byte_img))
        dir_ = os.path.join(dir_path, '%s/%d' % (token, num))
        pathlib.Path(dir_).mkdir(parents=True, exist_ok=True)
        img.save(os.path.join(dir_, 'init.jpg'), "JPEG")

def load_images_in_bytes(token: str, dir_path: str = './data/'):
    """
    Load all needed images from token subdir and return it bytes data.
    Args:
        path: Image path.
    Returns:
        Image byte array.
    """
    # Calculate init images number
    images_number: int = 0
    while True:
        dir_ = os.path.join(dir_path, '%s/%d' % (token, images_number))
        if not os.path.isdir(dir_):
            break
        images_number += 1
    print('images_number:', images_number)

    # Forming return data
    data: bytes = int_to_bytes(images_number) # return bytes data
    # Iterate init images
    for num in range(images_number):
        dir_ = os.path.join(dir_path, '%s/%d' % (token, num))
        img_bytes = load_image_in_bytes(os.path.join(dir_, 'init.jpg'))
        data += int_to_bytes(len(img_bytes))
        data += img_bytes

        # Calculate detected faces number in selected init image
        detected_number = 0
        while True:
            if not os.path.isdir(os.path.join(dir_, '%d' % detected_number)):
                break
            detected_number += 1
        print('detected_number:', detected_number)
        data += int_to_bytes(detected_number)

        # Iterate detected faces
        for det_num in range(detected_number):
            dir_det = os.path.join(dir_, '%d' % det_num)
            # Calculate similar faces number
            images_number_in_det_folder: int = 1
            while True:
                if not os.path.isfile(os.path.join(dir_det,
                                                   '%d.jpg' % images_number_in_det_folder)):
                    break
                images_number_in_det_folder += 1
            data += int_to_bytes(images_number_in_det_folder)

            # Iterate detected face and all similar faces
            pre_paths = ['detected.jpg'] + \
                        ['%d.jpg' % i for i in range(images_number_in_det_folder - 1)]
            for pre_path in pre_paths:
                img_bytes = load_image_in_bytes(os.path.join(dir_det, '%s' % pre_path))
                data += int_to_bytes(len(img_bytes))
                data += img_bytes
    return data
