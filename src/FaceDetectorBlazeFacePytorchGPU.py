# -*- coding: utf-8 -*- # pylint disable=C0103
"""Classes and functions for detecting faces.

"""

from math import atan2, sin, cos, degrees, radians, ceil, floor

from typing import List, Tuple, Dict, Any

import numpy as np # type: ignore

from PIL import Image, ImageDraw # type: ignore
import cv2 # type: ignore

from blazeface import BlazeFace # type: ignore
import torch # type: ignore
from tqdm.notebook import tqdm # type: ignore

#from matplotlib import pyplot # type: ignore

#from scipy import ndimage
#rotated = ndimage.rotate(image_to_rotate, 45)


EYES_AND_LIP = [0, 1, 3]
EYES_AND_NOSE = [0, 1, 2]
# 0 - right eye
# 1 - left eye
# 2 - nose
# 3 - mouth
# 4 - right ear
# 5 - left ear

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

def show_all_imgs(imgs: List[np.array]):
    """
    Show all images from list in one plot.
    Args:
        imgs: List of images np.arrays.
    Returns:
        None.
    """
    len_imgs = len(imgs)
    #for i in range(len_imgs):
    #    pyplot.subplot(1, len_imgs, i+1)
    #    pyplot.axis('off')
    #    pyplot.imshow(imgs[i])
    #pyplot.show()



def crop_img(img, x1, y1, x2, y2):
    if (x2 <= x1) | (y2 <= y1):
        return None
    
    ret = np.zeros((y2-y1, x2-x1, 3), dtype=np.uint8)
    if x1 >= img.shape[1] or y1 >= img.shape[0]:
        return ret
    if x2 <= 0 or y2 <= 0:
        return ret
    shx_src = 0 if x1 <= 0 else x1
    shy_src = 0 if y1 <= 0 else y1
    shx_dst = 0 if x1 >=0 else -x1
    shy_dst = 0 if y1 >=0 else -y1
    mxx = min(x2 - x1 - shx_dst + shx_src, img.shape[1])
    mxy = min(y2 - y1 - shy_dst + shy_src, img.shape[0])
    print(shy_dst, mxy-shy_src+shy_dst, shx_dst, mxx-shx_src+shx_dst, "|", 
          shy_src, mxy, shx_src, mxx)

    ret[shy_dst:mxy-shy_src+shy_dst, shx_dst:mxx-shx_src+shx_dst, :] = img[shy_src:mxy, shx_src:mxx, :]
    return ret

def crop_img_in(img, dst, x1, y1, x2, y2):
    dst.fill(0)
    if (x2 <= x1) | (y2 <= y1):
        return None

    #ret = np.zeros((y2-y1, x2-x1, 3), dtype=np.uint8)
    if x1 >= img.shape[1] or y1 >= img.shape[0]:
        return ret
    if x2 <= 0 or y2 <= 0:
        return ret
    shx_src = 0 if x1 <= 0 else x1
    shy_src = 0 if y1 <= 0 else y1
    shx_dst = 0 if x1 >=0 else -x1
    shy_dst = 0 if y1 >=0 else -y1
    mxx = min(x2 - x1 - shx_dst + shx_src, img.shape[1])
    mxy = min(y2 - y1 - shy_dst + shy_src, img.shape[0])
    #print(shy_dst, mxy-shy_src+shy_dst, shx_dst, mxx-shx_src+shx_dst, "|", 
    #      shy_src, mxy, shx_src, mxx)

    dst[shy_dst:mxy-shy_src+shy_dst, shx_dst:mxx-shx_src+shx_dst, :] = img[shy_src:mxy, shx_src:mxx, :]
    return mxx-shx_src+shx_dst, mxy-shy_src+shy_dst


def _overlapLine(p11, p12, p21, p22):
    """
    Gets length of interception of two 1-d sections.
    Args:
        p11: first point coordinate of the first section.
        p12: second point coordinate of the first section.
        p21: first point coordinate of the second section.
        p22: second point coordinate of the second section.
    Returns:
        Length of interception of two 1-d sections.
    """
    buf = p11
    p11 = min(p11, p12)
    p12 = max(buf, p12)

    buf = p21
    p21 = min(p21, p22)
    p22 = max(buf, p22)

    return  max(0, max(0, p12 - p21) - max(0, p11 - p21) - max(0, p12 - p22))


class FaceDetectorBlazeFaceGPU():
    """Class for detecting faces with BlazeFace algorythm on GPU.
    """
    def __init__(self, device='cuda:0'):
        gpu = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.back_net = BlazeFace(back_model=True).to(gpu)
        self.back_net.load_weights("src/blazefaceback.pth")
        self.back_net.load_anchors("src/anchorsback.npy")
        self.back_net.min_score_thresh = 0.5 #0.65
        self.back_net.min_suppression_threshold = 0.2 #0.3

        # for pyramid
        self.imposition = 128
        self.SzNet = 256
        self.MaxPyrK = 6
        self.maxImgSz = self.SzNet * self.MaxPyrK
        self.KInitResize = 1. # if one of image line size is larger ~2560, it resized
        self.KInitResize2 = 1. # if one of image line size is larger 256, it resized second time to adding as first image in batch
        self.imgRect = np.zeros((self.maxImgSz + self.SzNet, self.maxImgSz + self.SzNet, 3), dtype=np.uint8)
        self.imgRect.fill(127)


    def _get_pyramid_batch_from_image(self, img: np.array, precisedetection: bool) -> Tuple[int, int, np.array]:
        """
        Gets 256*256 images batch for detection.
        Args:
            img: Image in numpy array (init image).
        Returns:
            Tuple of first piramyd layer size and Numpy array of np.uint8 with shape (N, 256, 256, 3); first image in array is resized init image.
        """
        self.imgRect.fill(127)
        self.KInitResize = 1. # if one of image line size is larger ~2560, it resized
        self.KInitResize2 = 1. # if one of image line size is larger 256, it resized second time to adding as first image in batch

        (H, W, _) = img.shape
        maxWH = max(H, W)
        if maxWH > self.maxImgSz:
            self.KInitResize = self.maxImgSz / maxWH
            img = cv2.resize(img, (int(W * self.KInitResize), int(H * self.KInitResize)))
            (H, W, _) = img.shape
            maxWH = max(H, W)
        minWH = min(H, W)
        self.imgRect[:H, :W, :] = img

        if (maxWH <= self.SzNet):
            # only 1 layer and 1 image in "pyramid"
            ret = np.zeros((1, self.SzNet, self.SzNet, 3), dtype=np.uint8)
            ret[0] = self.imgRect[:self.SzNet, :self.SzNet, :]
            return (0, 0, ret)
        # 2 layers and several images in "pyramid" ?
        self.KInitResize2 = self.SzNet / maxWH
        initResizedTo256 = cv2.resize(self.imgRect[:maxWH, :maxWH, :], (self.SzNet, self.SzNet))
        if not precisedetection:
            # only 1 layer and 1 image in "pyramid"
            ret = np.zeros((1, self.SzNet, self.SzNet, 3), dtype=np.uint8)
            ret[0] = initResizedTo256[:self.SzNet, :self.SzNet, :]
            return (0, 0, ret)
        # 2 layers and several images in "pyramid"
        Nx = ceil((W - self.imposition) / (self.SzNet - self.imposition))
        Ny = ceil((H - self.imposition) / (self.SzNet - self.imposition))
        N = Nx * Ny + 1
        if N <= 1:
            print('ERROR PYRAMID: Nx, Ny=', Nx, Ny, '  W, H=', W, H)
        ret = np.zeros((N, self.SzNet, self.SzNet, 3), dtype=np.uint8)
        ret[0] = initResizedTo256
        k = 1
        delta = self.SzNet - self.imposition
        for j in range(Ny):
            for i in range(Nx):
                ret[k] = self.imgRect[delta * j:self.SzNet + delta * j, delta * i:self.SzNet + delta * i, :]
                k += 1
        return (Nx, Ny, ret)




    def _detect_from_pyramid(self, pyramid: Tuple[int, int, np.array], precisedetection: bool) -> List[List[Tuple[int, int]]]:
        """
        Gets list of all detected faces points in initial image coordinates.
        Args:
            pyramid: Tuple of first piramyd layer size and Numpy array of np.uint8 with shape (N, 256, 256, 3); first image in array is resized init image.
        Returns:
            List of all detected faces points in initial image coordinates.
        """
        FLAG = 1000000
        Nx, Ny, img_batch = pyramid
        self.back_net.min_score_thresh = 0.6 if precisedetection else 0.4
        detections_lst = self.back_net.predict_on_batch(img_batch)
        ##print('--dl--  detections_lst:', detections_lst) ####

        detectionss = []
        for detections in detections_lst:
            if isinstance(detections, torch.Tensor):
                detections = detections.cpu().numpy()
            if detections.ndim == 1:
                detections = np.expand_dims(detections, axis=0)
            detections *= self.SzNet
            detectionss.append(detections)

        #print('--0--  len(detectionss):', len(detectionss)) ####
        #print('--0--  detectionss:', detectionss) ####
        if len(detectionss) == 0:
            print('--1--', np.array(detectionss).shape) ####
            return None
        #if Nx == 0:
        #    # only one layer and only one not resized image
        #    print('--2--', detectionss[0].astype(np.uint8).shape) ####
        #    return detectionss[0].astype(np.uint8)#.tolist()

        #return detectionss
        # there are two layers
        detections0 = detectionss[0]
        detections0 /= (self.KInitResize * self.KInitResize2)
        #print('--2--  self.KInitResize, self.KInitResize2, detections0:', self.KInitResize, self.KInitResize2, detections0) ####
        k = 1
        for j in range(Ny):
            for i in range(Nx):
                dets = detectionss[k]
                # adding shifts
                shx = i * (self.SzNet - self.imposition)
                shy = j * (self.SzNet - self.imposition)
                for l in range(dets.shape[0]):
                    dets[l, 0] += shy # ymin
                    dets[l, 1] += shx # xmin
                    dets[l, 2] += shy # ymax
                    dets[l, 3] += shx # xmax
                    for m in range(6):
                        dets[l, 4 + m*2    ] += shx # mp_x
                        dets[l, 4 + m*2 + 1] += shy # kp_y
                dets /= self.KInitResize
                
                k += 1
        # End of low layer handling

        # handle face intersections with higher layer
        dets = detectionss[0]
        if Nx * Ny > 0:
            for dets2 in detectionss[1:]:
                for det in dets:
                    for det2 in dets2:
                        if det2[3] < 0:
                            continue
                        s = np.abs((det[3] - det[1]) * (det[2] - det[0]))
                        s2 = np.abs((det2[3] - det2[1]) * (det2[2] - det2[0]))
                        sint = _overlapLine(det[3], det[1], det2[3], det2[1]) * _overlapLine(det[2], det[0], det2[2], det2[0])
                        if s>100 and s2>100 and (np.abs(s2-s)/(s2+s))<0.2 and sint/s>0.7:
                            det[3] -= FLAG

        # Putting results in out array
        ret = []
        for dets in detectionss:
            dets = dets.astype(int)
            for face in dets:
                if face[3] < 0:
                    continue
                rt = np.zeros((8, 2))
                ymin = face[0]
                xmin = face[1]
                ymax = face[2]
                xmax = face[3]
                rt[0] = np.array([xmin, ymin])
                rt[1] = np.array([xmax, ymax])
                for k in range(6):
                    kp_x = face[4 + k*2    ]
                    kp_y = face[4 + k*2 + 1]
                    rt[k + 2] = np.array([kp_x, kp_y])
                ret.append(rt)
        #print('--3-- ret:', np.array(ret)) ####
        #print('--3--', np.array(ret).shape) ####
        return np.array(ret)

    def ss_from_detections(self, dets, i, j):
        try:
            s = np.abs((dets[i][0, 0] - dets[i][1, 0]) * (dets[i][0, 1] - dets[i][1, 1]))
        except:
            print('======================================')
            print(dets)
            print('======================================')
        s2 = np.abs((dets[j][0, 0] - dets[j][1, 0]) * (dets[j][0, 1] - dets[j][1, 1]))
        sint = _overlapLine(dets[i][0, 0], dets[i][1, 0], dets[j][0, 0], dets[j][1, 0]) * _overlapLine(dets[i][0, 1], dets[i][1, 1], dets[j][0, 1], dets[j][1, 1])
        flag = False
        if s > 100 and s2 > 100 and (np.abs(s2 - s)/(s2 + s)) < 0.3 and sint / s > 0.7:
            flag = True
        return s, s2, sint, flag

    def filter_intersections(self, detections):
        if len(detections) == 0:
            return None
        detections = np.array(detections)
        cluster_indexes = (np.ones(detections.shape[0]) * -1).astype(np.int32) # max count of clusters is detections.shape[0]
        cur_cluster_index = 0 # number of next cluster (from zero)
        for i in range(detections.shape[0] - 1):
            if cluster_indexes[i] >= 0: # already in any cluster
                continue
            cluster_indexes[i] = cur_cluster_index
            for j in range(i + 1, detections.shape[0]):
                if cluster_indexes[j] >= 0: # already in any cluster
                    continue
                # flag = are we gluing they in one cluster?
                _, _, _, flag = self.ss_from_detections(detections, i, j)
                if flag:
                    cluster_indexes[j] = cur_cluster_index
            cur_cluster_index += 1
            
        ret = np.zeros((np.max(cluster_indexes) + 1, 8, 2))
        for cluster_index in range(np.max(cluster_indexes) + 1):
            mask = (cluster_indexes == cluster_index)
            cnt = np.sum(mask)
            #rt = detections.dot(mask.reshape(-1,1)) / cnt
            #rt = mask.reshape(-1,1).dot(detections) / cnt
            rt = np.sum(detections[mask], axis=0) / cnt
            ret[cluster_index] = rt

        return ret

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
        pyramid = self._get_pyramid_batch_from_image(image_numpy, precisedetection)
        _, _, fcs = pyramid
        if len(fcs) == 0:
            return [], [], [], []
        detections = self._detect_from_pyramid(pyramid, precisedetection)
        if detections is None:
            return [], [], [], []
        # gluing detections in clusters and give they
        if np.array(detections).shape[0] > 1:
            detections: np.array = self.filter_intersections(detections)#.astype(np.int32)
        #print('--4--  detections:', detections) ####
        if detections is None:
            return [], [], [], []

        faces = []
        xywhs = []
        facepointss = []
        fo_detections = [] # etc
        faces = []

        for i in range(detections.shape[0]):
            fc: np.array = detections[i]
            fo_detections.append(1.)
            face = {}
            x, y  = fc[0, 0], fc[0, 1]
            width, height = fc[1, 0] - x, fc[1, 1] - y
            # fitting box size under MTCNN
            x -= width * 0.09
            width += width * 0.18
            y -= height * 0.09
            height += height * 0.18
            # checking Smin
            if width * height < Smin:
                continue
            # appending
            face['box'] = (x, y, width, height)
            face['keypoints'] = {'left_eye': (fc[2, 0], fc[2, 1]), 'right_eye': (fc[3, 0], fc[3, 1]),
                                 'nose': (fc[4, 0], fc[4, 1]), 'lip': (fc[5, 0], fc[5, 1])}
            faces.append(face)
            xywhs.append(face['box'])
            facepointss.append([(fc[2, 0], fc[2, 1]), (fc[3, 0], fc[3, 1]), (fc[4, 0], fc[4, 1]),
                                (fc[5, 0], fc[5, 1]), (fc[6, 0], fc[6, 1]), (fc[7, 0], fc[7, 1])])
        return xywhs, faces, fo_detections, facepointss

    def rotate(self, image_numpy: np.array, # pylint: disable=R0914
               face_or_fo_detection: Dict[str, Any],
               size: int, padding: int) -> np.array:
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

        img = Image.fromarray(image_numpy)
        # get coordinates
        fc_x, fc_y, width, height = face['box']
        fc_x2, fc_y2 = fc_x + width, fc_y + height
        fc_xc, fc_yc = int((fc_x + fc_x2) / 2), int((fc_y + fc_y2) / 2)
        # create image for rotating
        rot_w_2 = int(width * (1 + 2 * padding) * 1.8 / 2) # 1.41 -> 1.8
        rot_h_2 = int(height * (1 + 2 * padding) * 1.8 / 2)
        im_rot = img.crop((fc_xc - rot_w_2, fc_yc - rot_h_2, fc_xc + rot_w_2, fc_yc + rot_h_2))
        # rotate image
        keypoints = face['keypoints']
        left_eye, right_eye, nose = keypoints['left_eye'], keypoints['right_eye'], keypoints['nose']
        #print(nose, left_eye, right_eye)
        left_eye_, right_eye_, nose_ = [], [], []
        for p, p2 in zip([left_eye, right_eye, nose], [left_eye_, right_eye_, nose_]):
            x = p[0] - (fc_xc - rot_w_2)
            y = p[1] - (fc_yc - rot_h_2)
            p2.append(x)
            p2.append(y)
        angle_rad = atan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])
        angle_deg = degrees(angle_rad)
        im_rot = im_rot.rotate(angle_deg, resample=Image.BICUBIC, expand=True)
        w_rot, h_rot = im_rot.size  # width and height are increased after rotating (expand=True)
        rot_w_2_, rot_h_2_ = int(w_rot / 2), int(h_rot / 2)
        cos_ang = cos(angle_rad)
        sin_ang = sin(angle_rad)
        #print(nose_, left_eye_, right_eye_, rot_w_2_, rot_h_2_)
        for p in left_eye_, right_eye_, nose_:
            x = p[0] - rot_w_2
            y = rot_h_2 - p[1]
            p[0] = cos_ang * x - sin_ang * y
            p[1] = cos_ang * y + sin_ang * x
            p[0] += rot_w_2_
            p[1] = rot_h_2_ - p[1]
        
        # new detection
        #im_rot_np = np.asarray(im_rot)
        #
        #largest_face, largest_sq = None, 0
        #
        #boxes, probs, points = self.mtcnn_detector.detect(im_rot_np.copy(), landmarks=True)
        #faces = []
        #if not boxes is None:
        #    intvfunc = np.vectorize(lambda x: int(x))
        #    for box, prob, pnts in zip(boxes.tolist(), probs.tolist(), points.tolist()):
        #        if prob < 0.95:
        #            continue
        #        box = intvfunc(box)
        #        face = {}
        #        face['box'] = (box[0], box[1], box[2] - box[0], box[3] - box[1])
        #        face['keypoints'] = {'left_eye': pnts[0], 'right_eye': pnts[1]}
        #        faces.append(face)
        #
        #for fcr in faces:
        #    (_, _, fc_w, fc_h) = fcr['box']
        #    if fc_w * fc_h > largest_sq:
        #        largest_sq = fc_w * fc_h
        #        largest_face = fcr
        #if largest_face is None:
        #    largest_face = face
        #    fc_x, fc_y, fc_w, fc_h = largest_face['box']
        #    (fc_x, fc_y) = transform_xy(fc_x, fc_y, fc_xc, fc_yc, rot_w_2, rot_h_2, angle)
        #    largest_face['box'] = fc_x, fc_y, fc_w, fc_h
        
        # calculating box (largest_face['box']) after rotate instead of new detection
        largest_face = {'box': [0, 0, 0, 0]}
        mid_x = int((left_eye_[0] + right_eye_[0]) / 2)
        mid_y = int((nose_[1] + (left_eye_[1] + right_eye_[1]) / 2) / 2)
        #print(mid_x, mid_y, width, height, mid_x - int(width / 2), mid_y - int(height / 2))
        #print('-------')
        largest_face['box'] = [mid_x - int(width / 2),
                               mid_y - int(height / 2),
                               width,
                               height]
        
        # cropping
        fc_x, fc_y, width, height = largest_face['box']
        fc_x2, fc_y2 = fc_x + width, fc_y + height
        fc_xc, fc_yc = int((fc_x + fc_x2) / 2), int((fc_y + fc_y2) / 2)
        wh_max = max(width, height)
        wh_2 = int(wh_max * (1 + 2 * padding) / 2)
        im_rot = im_rot.crop((fc_xc - wh_2, fc_yc - wh_2, fc_xc + wh_2, fc_yc + wh_2))
        im_rot = np.asarray(im_rot)

        #h, w, _ = im_rot.shape
        im_rot = cv2.resize(im_rot, (size, size), interpolation=cv2.INTER_AREA) # pylint: disable=E1101

        return im_rot


    def rotate_pack(self, img, xywhs, faces, size, padding, Smin):
        """
        Rotate cropped faces.
        Args:
            img: Image in numpy array.
            Smin: Minimum faces square.
            xywhs: Faces boxes.
            faces: Faces detections.
            size: Face size.
            padding: Face padding.
        Returns:
            Numpy array with rotated image.
        """
        # return lists
        im_rots = []
        keypoints_160 = []

        # estimating of pre-resize image needing
        min_face_S = size * size * 1000
        max_face_S = 0
        for face in faces:
            fc_x, fc_y, width, height = face['box']
            S = width * height
            min_face_S = S if S < min_face_S else min_face_S
            max_face_S = S if S > max_face_S else max_face_S
        # if need to resize:
        if np.sqrt(min_face_S) > size * 1.2 * 1.2:
            k_resize = size * 1.2 / np.sqrt(min_face_S)
            h, w, _ = img.shape
            img = cv2.resize(img, (int(w * k_resize), int(h * k_resize)), interpolation=cv2.INTER_AREA)
            print('k_resize:', k_resize)
            #print('old coords:', faces[0]['box'], faces[0]['keypoints'])
            for face in faces:
                fc_x, fc_y, width, height = face['box']
                face['box'] = int(fc_x * k_resize), int(fc_y * k_resize), int(width * k_resize), int(height * k_resize)
                keypoints = face['keypoints']
                left_eye, right_eye, nose = keypoints['left_eye'], keypoints['right_eye'], keypoints['nose']
                left_eye = (int(left_eye[0] * k_resize), int(left_eye[1] * k_resize))
                right_eye = (int(right_eye[0] * k_resize), int(right_eye[1] * k_resize))
                nose = (int(nose[0] * k_resize), int(nose[1] * k_resize))
                face['keypoints'] = {'left_eye': left_eye, 'right_eye': right_eye, 'nose': nose}
            #print('new coords:', faces[0]['box'], faces[0]['keypoints'])

        # main handling...
        img = Image.fromarray(img)
        """
        max_crop_w = 0
        max_crop_h = 0
        auxiliary = []
        for xywh, face in zip(xywhs, faces):
            fc_x, fc_y, width, height = face['box']
            fc_x2, fc_y2 = fc_x + width, fc_y + height
            fc_xc, fc_yc = int((fc_x + fc_x2) / 2), int((fc_y + fc_y2) / 2)
            # for image for rotating
            rot_w_2 = int(width * (1 + 2 * padding) * 1.8 / 2) # 1.41 -> 1.8
            rot_h_2 = int(height * (1 + 2 * padding) * 1.8 / 2)

            if max_crop_w < rot_w_2 * 2:
                max_crop_w = rot_w_2 * 2
            if max_crop_h < rot_h_2 * 2:
                max_crop_h = rot_h_2 * 2
            auxiliary.append((fc_xc, fc_yc, rot_w_2, rot_h_2))

        img_for_crop = np.zeros((max_crop_h, max_crop_w, 3), dtype=np.uint8)"""
        ####for xywh, face, aux in zip(xywhs, faces, auxiliary):
        for face in faces:
            try:
                fc_x, fc_y, width, height = face['box']
                fc_x2, fc_y2 = fc_x + width, fc_y + height
                fc_xc, fc_yc = int((fc_x + fc_x2) / 2), int((fc_y + fc_y2) / 2)
                # get keypoints
                keypoints = face['keypoints']
                left_eye, right_eye, nose = keypoints['left_eye'], keypoints['right_eye'], keypoints['nose']

                #print(nose, left_eye, right_eye)
                # draw first circles
                ##image_numpy = np.float32(img)
                #for p in [left_eye, right_eye, nose]:
                #    image_numpy = cv2.circle(image_numpy, tuple(p), radius=2, color=(0, 0, 255), thickness=1) # color is (BGR) or (RGB)?
                ##image_numpy = np.uint8(image_numpy)
                # create image for rotating (pre-cropping)
                ##img = Image.fromarray(image_numpy) # cv2 -> PIL
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
                #   draw third circles
                #draw = ImageDraw.Draw(im_rot)
                #for p in [left_eye_, right_eye_, nose_]:
                #    draw.ellipse((p[0] - 4, p[1] - 4, p[0] + 4, p[1] + 4), outline=(255, 255, 0)) # RGB

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

                im_rots.append(im_rot)
                keypoints_160.append([left_eye_, right_eye_, nose_])
            except Exception:
                print('!!!!! WTF!!! ERROR IN rotate_pack !!!!!')

        return im_rots, keypoints_160

"""
            # for image for rotating
            rot_w_2 = int(width * (1 + 2 * padding) * 1.8 / 2) # 1.41 -> 1.8
            rot_h_2 = int(height * (1 + 2 * padding) * 1.8 / 2)
            im_rot = img.crop((fc_xc - rot_w_2, fc_yc - rot_h_2, fc_xc + rot_w_2, fc_yc + rot_h_2))

            # rotate image
            #print(nose, left_eye, right_eye)
            ### Rotate image
            #   calc keypoints after pre-cropping
            left_eye_, right_eye_, nose_ = [], [], []
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

            # calculating box (largest_face['box']) after rotate instead of new detection
            largest_face = {'box': [0, 0, 0, 0]}
            mid_x = int((left_eye_[0] + right_eye_[0]) / 2)
            mid_y = int((nose_[1] + (left_eye_[1] + right_eye_[1]) / 2) / 2)
            #print(mid_x, mid_y, width, height, mid_x - int(width / 2), mid_y - int(height / 2))
            #print('-------')
            largest_face['box'] = [mid_x - int(width / 2),
                                   mid_y - int(height / 2),
                                   width,
                                   height]

            # final cropping
            fc_x, fc_y, width, height = largest_face['box']
            fc_x2, fc_y2 = fc_x + width, fc_y + height
            fc_xc, fc_yc = int((fc_x + fc_x2) / 2), int((fc_y + fc_y2) / 2)
            wh_max = max(width, height)
            wh_2 = int(wh_max * (1 + 2 * padding) / 2)
            im_rot = im_rot.crop((fc_xc - wh_2, fc_yc - wh_2, fc_xc + wh_2, fc_yc + wh_2))
            ####w_, h_ = crop_img_in(im_rot, img_for_crop, fc_xc - wh_2, fc_yc - wh_2, fc_xc + wh_2, fc_yc + wh_2)
            ####im_rot = img_for_crop[:h_, :w_, :]
            #   calc keypoints after final cropping
            for p in left_eye_, right_eye_, nose_:
                p[0] -= (fc_xc - wh_2)
                p[1] -= (fc_yc - wh_2)


            (oldw, oldh) = im_rot.size
            im_rot = np.asarray(im_rot)

            h, w, _ = im_rot.shape
            im_rot = cv2.resize(im_rot, (size, size), interpolation=cv2.INTER_AREA) # pylint: disable=E1101
            #   calc final keypoints after resizing
            for p in left_eye_, right_eye_, nose_:
                p[0] = int(p[0] * size / oldw)
                p[1] = int(p[1] * size / oldw)
            #   draw fifth circles
            for p in [left_eye_, right_eye_, nose_]:
                im_rot = cv2.circle(im_rot, tuple(p), radius=4, color=(255, 255, 255), thickness=1) # color is (BGR) or (RGB)?
"""


def detect_Blaze(fcdet, imgs, Smin, size, padding):
    det_imgs = []
    det_xywhs = []
    for img in imgs:
        xywhs, faces, fo_detections, facepointss = fcdet.get_detected_faces_xywhs(img.copy())
        xywhs2, faces2, fo_detections2, facepointss2 = [], [], [], []
        for xywh, face, fo_detection, facepoints in zip(xywhs, faces, fo_detections, facepointss):
            #fc_x, fc_y, width, height = face['box']
            S = xywh[2] * xywh[3]#width * height
            if S >= Smin:
                xywhs2.append(xywh)
                faces2.append(face)
                fo_detections2.append(fo_detection)
                facepointss2.append(facepoints)
        #try:
        det_imgs += fcdet.rotate_pack(img, Smin, xywhs2, faces2, size, padding)
        for xywh, face, fo_detection, facepoints in zip(xywhs2, faces2, fo_detections2, facepointss2):
            #if xywh[2] * xywh[3] >= Smin:
            det_xywhs.append(xywh)
        #except:
        #    print('    Error detecting image')
    return det_imgs, det_xywhs

