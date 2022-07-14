# -*- coding: utf-8 -*- # pylint disable=C0103
"""Classes and functions for detecting faces.

"""

import os
import sys
from math import atan2, sin, cos, degrees, radians

from typing import List, Tuple, Dict, Any, NamedTuple

import numpy as np # type: ignore

from PIL import Image, ImageDraw # type: ignore
import cv2 # type: ignore
import insightface # type: ignore
import numpy as np # type: ignore
import torch # type: ignore


def transform_xy(point_x: int, point_y: int, # pylint: disable=R0913
                 center_x: int, center_y: int,
                 new_width2: int, new_height2: int,
                 angle: float) -> Tuple[int, int]:
    """
    Transforms coordinates from init coord system to
    coordinate system after rotating and cropping.
    Args:
        point_x: Point X coordinate in old system.
        point_y: Point Y coordinate in old system.
        center_x: Rotating center X coordinate in old system.
        center_y: Rotating center Y coordinate in old system.
        new_width2: Half of new image width.
        new_height2: Half of new image height.
        angle: Rotating angle in degrees.
    Returns:
        Tuple of point coordinates in new coordinate system.
    """
    # to cropped area (y in bottom)
    point_x = point_x + new_width2 - center_x
    point_y = point_y + new_height2 - center_y
    # to center (y in top)
    point_x = point_x - new_width2
    point_y = new_height2 - point_y
    # rotating (center coords, y in top)
    angle = radians(angle)
    new_x = point_x * cos(angle) - point_y * sin(angle)
    new_y = point_y * cos(angle) + point_x * sin(angle)
    # to cropped area (y in bottom)
    point_xf = new_x + new_width2
    point_yf = new_height2 - new_y
    return int(point_xf), int(point_yf)



class FaceDetectorInsight():
    """Class for detecting faces with MTCNN algorythm on GPU.
    """
    def __init__(self, device='cuda'):
        self._model = insightface.app.FaceAnalysis(rec_name=None, ga_name=None)#, ga_name rec_name=None)
        self._model.prepare(0, nms=0.4)

    def read_image(self, image_path: str) -> np.array: # pylint: disable=R0201
        """
        Reads image from file to numpy array.
        Args:
            image_path: Image path.
        Returns:
            Image in numpy array.
        """
        # img = dlib.load_rgb_image(filename)
        img = cv2.imread(image_path)
        if img is None:
            return img
        return img[..., ::-1] # pylint: disable=E1101

    def show_image(self, image_numpy: np.array): # pylint: disable=R0201
        """
        Shows image.
        Args:
            image_numpy: Image in numpy array.
        Returns:
            None.
        """
        pyplot.imshow(image_numpy)
        pyplot.show()

    def save_image(self, image_numpy: np.array, image_path: str): # pylint: disable=R0201
        """
        Saves image from numpy array to file.
        Args:
            image_numpy: Image in numpy array.
            image_path: Image path.
        Returns:
            None.
        """
        return cv2.imwrite(image_path, image_numpy[..., ::-1]) # pylint: disable=E1101

    def crop_image(self, image_numpy: np.array, # pylint: disable=R0913, R0201
                   im_x: int, im_y: int, im_w: int, im_h: int) -> np.array:
        """
        Crops image.
        Args:
            image_numpy: Image in numpy array.
            im_x: Crop rectangle X coordinate.
            im_y: Crop rectangle Y coordinate.
            im_w: Crop rectangle width.
            im_h: Crop rectangle height.
        Returns:
            Cropped image in numpy array.
        """
        img = Image.fromarray(image_numpy)
        img = img.crop((im_x, im_y, im_x + im_w, im_y + im_h))
        img = np.asarray(img)
        return img

    def _get_bbox(self, extractedModel):        
        boxes = []        
        for b, lst in enumerate(extractedModel):
            boxes.append(lst[0])
        return boxes

    def _get_scores(self, extractedModel):        
        scores = []        
        for b, lst in enumerate(extractedModel):
            scores.append(lst[2])
        return scores

    def _get_keypoints(self, extractedModel):
        keypoints = []        
        for b, lst in enumerate(extractedModel):
            keypoints.append(lst[1])
        return keypoints

    def _get_extracted_model_data(self, image: Image):
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        return self._model.get(img)

    def _get_embeddings(self, extractedModel):
        embeddings = []        
        for b, lst in enumerate(extractedModel):
            embeddings.append(lst[7])
        embeddings_np = np.zeros((len(embeddings), 1, 512))
        for j, item in enumerate(embeddings):
            embeddings_np[j] = np.asarray(item)
        embeddings_np = embeddings_np.reshape((len(embeddings_np), 512))
        return embeddings_np

    def get_detected_faces_xywhs(self,
                                 image_numpy: np.array, Smin: int, precisedetection: bool) -> Tuple[List[int], List[Tuple[int]],
                                                                 List[str], List[str]]:
        """
        Detects faces.
        Args:
            image_numpy: Image in numpy array.
        Returns:
            Tuple of lists with xywhs, faces, detections and points of faces.
        """
        boxes = []
        xywhs = []
        facepointss = []
        fo_detections = [] # etc
        faces = []
        #embs = []

        extractedModel = self._model.get(image_numpy)
        boxes = self._get_bbox(extractedModel)
        points = self._get_keypoints(extractedModel)
        probs = self._get_scores(extractedModel)
        #embs_ = self._get_embeddings(extractedModel)

        intvfunc = np.vectorize(lambda x: int(x))
        if boxes is None or len(boxes) == 0:
            return [], [], [], [], []

        #for box, prob, pnts, emb in zip(boxes, probs, points, embs_):
        for box, prob, pnts in zip(boxes, probs, points):
            if prob < 0.8:
                continue
            box = intvfunc(box)
            pnts = intvfunc(pnts)[:3]
            facepointss.append(pnts)
            fo_detections.append(prob)
            face = {}
            face['box'] = (box[0], box[1], box[2] - box[0], box[3] - box[1])
            face['keypoints'] = {'left_eye': pnts[0], 'right_eye': pnts[1], 'nose': pnts[2]}
            faces.append(face)
            xywhs.append(face['box'])
            #embs.append(emb)
        return xywhs, faces, fo_detections, facepointss#, embs

    def rotate(self, image_numpy: np.array, # pylint: disable=R0914
               face_or_fo_detection: Dict[str, Any],
               size: int, padding: float) -> np.array:
        """
        Rotate image.
        Args:
            image_numpy: Image in numpy array.
            face_or_fo_detection: Image detection.
            size: Face size.
            padding: Face padding.
        Returns:
            Numpy array with rotated image.
        """
        # here: face
        face = face_or_fo_detection

        # get coordinates
        fc_x, fc_y, width, height = face['box']
        fc_x2, fc_y2 = fc_x + width, fc_y + height
        fc_xc, fc_yc = int((fc_x + fc_x2) / 2), int((fc_y + fc_y2) / 2)
        # get keypoints
        keypoints = face['keypoints']
        left_eye, right_eye, nose = keypoints['left_eye'], keypoints['right_eye'], keypoints['nose']
        #print(nose, left_eye, right_eye)
        # draw first circles
        image_numpy = np.float32(image_numpy)
        #for p in [left_eye, right_eye, nose]:
        #    image_numpy = cv2.circle(image_numpy, tuple(p), radius=2, color=(0, 0, 255), thickness=1) # color is (BGR) or (RGB)?
        image_numpy = np.uint8(image_numpy)
        # create image for rotating (pre-cropping)
        img = Image.fromarray(image_numpy) # cv2 -> PIL
        rot_w_2 = int(width * (1 + 2 * padding) * 1.8 / 2) # 1.41 -> 1.8
        rot_h_2 = int(height * (1 + 2 * padding) * 1.8 / 2)
        im_rot = img.crop((fc_xc - rot_w_2, fc_yc - rot_h_2, fc_xc + rot_w_2, fc_yc + rot_h_2))
        ### Rotate image
        #   calc keypoints after pre-cropping
        left_eye_, right_eye_, nose_ = [], [], [] # keypoints after rotating
        for p, p2 in zip([left_eye, right_eye, nose], [left_eye_, right_eye_, nose_]):
            x = p[0] - (fc_xc - rot_w_2)
            y = p[1] - (fc_yc - rot_h_2)
            p2.append(x)
            p2.append(y)
        #   draw second circles
        #draw = ImageDraw.Draw(im_rot)
        #for p in [left_eye_, right_eye_, nose_]:
        #    draw.ellipse((p[0] - 2, p[1] - 2, p[0] + 2, p[1] + 2), outline=(255, 0, 0)) # RGB
        #   calc rotating angle
        angle_rad = atan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])
        angle_deg = degrees(angle_rad)
        #   rotating image
        im_rot = im_rot.rotate(angle_deg, resample=Image.BICUBIC, expand=True)
        w_rot, h_rot = im_rot.size  # width and height are increased after rotating (expand=True)
        rot_w_2_, rot_h_2_ = int(w_rot / 2), int(h_rot / 2)
        cos_ang = cos(angle_rad)
        sin_ang = sin(angle_rad)
        #   calc keypoints after rotating
        #print(nose_, left_eye_, right_eye_, rot_w_2_, rot_h_2_)
        for p in left_eye_, right_eye_, nose_:
            x = p[0] - rot_w_2
            y = rot_h_2 - p[1]
            p[0] = cos_ang * x - sin_ang * y
            p[1] = cos_ang * y + sin_ang * x
            p[0] += rot_w_2_
            p[1] = rot_h_2_ - p[1]

        #   calculating box (largest_face['box']) after rotate instead of new detection
        largest_face = {'box': [0, 0, 0, 0]}
        mid_x = int((left_eye_[0] + right_eye_[0]) / 2)
        mid_y = int((nose_[1] + (left_eye_[1] + right_eye_[1]) / 2) / 2)
        #print(mid_x, mid_y, width, height, mid_x - int(width / 2), mid_y - int(height / 2))
        #print('-------')
        largest_face['box'] = [mid_x - int(width / 2),
                               mid_y - int(height / 2),
                               width,
                               height]

        ### final cropping
        fc_x, fc_y, width, height = largest_face['box']
        fc_x2, fc_y2 = fc_x + width, fc_y + height
        fc_xc, fc_yc = int((fc_x + fc_x2) / 2), int((fc_y + fc_y2) / 2)
        wh_max = max(width, height)
        wh_2 = int(wh_max * (1 + 2 * padding) / 2)

        im_rot = im_rot.crop((fc_xc - wh_2, fc_yc - wh_2, fc_xc + wh_2, fc_yc + wh_2))
        #   calc keypoints after final cropping
        for p in left_eye_, right_eye_, nose_:
            p[0] -= (fc_xc - wh_2)
            p[1] -= (fc_yc - wh_2)
        #   draw fourth circles
        #draw = ImageDraw.Draw(im_rot)
        #for p in [left_eye_, right_eye_, nose_]:
        #    draw.ellipse((p[0] - 4, p[1] - 4, p[0] + 4, p[1] + 4), outline=(255, 255, 0)) # RGB

        #h, w, _ = im_rot.shape
        (oldw, oldh) = im_rot.size
        im_rot = np.asarray(im_rot) # PIL -> cv2
        im_rot = cv2.resize(im_rot, (size, size), interpolation=cv2.INTER_AREA) # pylint: disable=E1101

        #   calc final keypoints after resizing
        for p in left_eye_, right_eye_, nose_:
            p[0] = int(p[0] * size / oldw)
            p[1] = int(p[1] * size / oldw)
        #   draw fifth circles
        #for p in [left_eye_, right_eye_, nose_]:
        #    im_rot = cv2.circle(im_rot, tuple(p), radius=4, color=(255, 255, 255), thickness=1) # color is (BGR) or (RGB)?

        return im_rot, [left_eye_, right_eye_, nose_]

    def get_one_img_result(self, img, Smin, size, padding):
        det_imgs1 = []
        #det_imgs2 = []
        #keypoints = [] # For 1 only
        #try:
        xywhs, faces, fo_detections, facepointss = self.get_detected_faces_xywhs(img.copy(), Smin=Smin, precisedetection=True)
        #print(xywhs, faces, fo_detections, facepointss)
        for xywh, face, fo_detection, facepoints in zip(xywhs, faces, fo_detections, facepointss):
            #print(xywh, face, fo_detection, facepoints)
            if xywh[2] * xywh[3] >= Smin:
                im_rot, [left_eye_, right_eye_, nose_] = self.rotate(img, face, size * 2, padding * 4)
                det_imgs1.append(im_rot)
                #keypoints.append(face['keypoints'])
        #for img2 in det_imgs1:
        #    xywhs, faces, fo_detections, facepointss = self.get_detected_faces_xywhs(img2.copy(), Smin=Smin, precisedetection=True)
        #    for xywh, face, fo_detection, facepoints in zip(xywhs, faces, fo_detections, facepointss):
        #        im_rot, [left_eye_, right_eye_, nose_] = self.rotate(img2, face, size, padding)
        #        det_imgs2.append(im_rot)
        #        break
        #except:
        #    #logger.error('    Error detecting image')
        #    print('Err')
        #    pass
        return det_imgs1, faces#, keypoints
