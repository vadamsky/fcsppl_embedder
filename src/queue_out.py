#!/usr/bin/env python3

# apt install python3-protobuf
# ./protoc -I=. --python_out=. ./out.proto

import time
import sys
import json
import pickle
import base64
import math
#mport shared_memory
import numpy as np
import cv2

#rom named_atomic_lock import NamedAtomicLock
from rabbit_wrapper import RabbitWrapper
from zmq_wrapper import ZmqWrapper
from base_class import BaseClass

import out_pb2 as Out

from constants import QUEUE_OUT_ADDR
from constants import RABBIT_IN_ADDR, RABBIT_OUT_ADDR, RABBIT_IN_EXCH, RABBIT_OUT_EXCH
from constants import RABBIT_IN_QNAME, RABBIT_OUT_QNAME, RABBIT_IN_RKEY, RABBIT_OUT_RKEY


def get_base64_from_np_image(np_img, image_quality=90):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), image_quality]
    _, nparr = cv2.imencode('.jpg', np_img[..., ::-1], encode_param)
    return base64.b64encode(nparr.tobytes())


class QueueOut(BaseClass):
    def __init__(self):
        self.dt_embeds = {'insight': [], 'facenet': []}
        self.img_id_dts = {}
        self.img_ids_to_publish = []

        self.rabbit = RabbitWrapper(tp='out', rabt_addr=RABBIT_OUT_ADDR, exch_nm=RABBIT_OUT_EXCH,
                                    q_nm=RABBIT_OUT_QNAME, r_key=RABBIT_OUT_RKEY)
        self.zmq_out = ZmqWrapper('', addr=QUEUE_OUT_ADDR, tp='server')
        self.zmq_out.run_serv(self.zmq_handle_func)


    def zmq_handle_func(self, dt):
        (identifier, tm, msg_type, data) = dt
        #print('QueueOut:', data)
        self.handle_embed_imgs(msg_type, data)

    def handle_embed_imgs(self, msg_type, data):
        #print(self.dt_embeds)
        print('QueueOut  img_dct["img_id"]:', data['img_id'])#[0][0])
        #
        if msg_type in ['double_model']:
            # publish data now
            # data: {'img_id': img_id, 'detected':[('img_id, img_to_embed, face)], 'embedded_double':[embedding]}
            img_dct = data
            img_id = img_dct['img_id']
            print('0 publish_img_id_dts:', img_id)
            face_imgs = [d[1] for d in img_dct['detected']]
            faces = [d[2] for d in img_dct['detected']]
            emb_doubles = img_dct['embedded_double']
            #
            self.publish_img_dt(img_id, faces, [], [], emb_doubles, face_imgs)
        #
        if msg_type in ['double_changer', 'detector']:
            # publish data now
            # data: {'img_id': img_id, 'detected':[('img_id, img_to_embed, face)], 'embedded_insight':[embedding], 'embedded_facenet':[embedding]}
            img_dct = data
            img_id = img_dct['img_id']
            print('1 publish_img_id_dts:', img_id)
            face_imgs = [d[1] for d in img_dct['detected']]
            faces = [d[2] for d in img_dct['detected']]
            emb_insights = img_dct['embedded_insight']
            emb_facenets = img_dct['embedded_facenet']
            #
            self.publish_img_dt(img_id, faces, emb_insights, emb_facenets, [], face_imgs)
        #
        if msg_type in ['insight', 'facenet']:
            # collect data to publish
            # data: img_id, img_to_embed, face, embedding
            self.dt_embeds[msg_type] = self.dt_embeds[msg_type] + data # data:List=embed
            #print(self.dt_embeds)
            i = 0
            for insight_dt in self.dt_embeds['insight']:
                j = 0
                for facenet_dt in self.dt_embeds['facenet']:
                    print(i, j, insight_dt[0], facenet_dt[0]) #, insight_dt, facenet_dt)
                    if insight_dt[0] == facenet_dt[0]: # img_ids
                        self.handle_embed_imgs_pair(insight_dt, facenet_dt)
                        self.dt_embeds['insight'] = self.dt_embeds['insight'][:i] + [] if len(self.dt_embeds['insight'])<=i+1 else self.dt_embeds['insight'][i+1:]
                        self.dt_embeds['facenet'] = self.dt_embeds['facenet'][:j] + [] if len(self.dt_embeds['facenet'])<=j+1 else self.dt_embeds['facenet'][j+1:]
                        if not (insight_dt[0] in self.img_ids_to_publish):
                            self.img_ids_to_publish.append(insight_dt[0])
                        j += 1
                i += 1

            if len(self.img_ids_to_publish):
                for img_id in self.img_ids_to_publish:
                    self.publish_img_id_dts(img_id)
                self.img_ids_to_publish = []
        return None

    def handle_embed_imgs_pair(self, insight_dt, facenet_dt):
        # handle
        #
        if self.img_id_dts.get(insight_dt[0]) is None:
            self.img_id_dts[insight_dt[0]] = []
        # data: img_id, img_to_embed, face, embedding
        self.img_id_dts[insight_dt[0]].append( (insight_dt[0], insight_dt[1], insight_dt[2], insight_dt[3], facenet_dt[3]) )
        # img_id, img_to_embed, face, embIns, embFcn

    def publish_img_id_dts(self, img_id):
        # handle
        #
        print('publish_img_id_dts:', img_id)
        dt = self.img_id_dts[img_id]
        print('===', len(dt))
        ####self.rabbit.publish(b'')
        print('Publish:', dt[0], dt[1].shape, dt[2], len(dt[3]), len(dt[4]))
        self.img_id_dts.pop(img_id, None) # remove key from self.img_id_dts


    """
    def publish_img_dt(self, img_id, faces, emb_insights, emb_facenets, emb_doubles, face_imgs=[]):
        #jsn = {'img_id': img_id, 'faces': [], 'emb_insights': [], 'emb_facenets': []}
        #if face_imgs:
        jsn = {'img_id': img_id, 'faces': [], 'face_imgs': [], 'emb_insights': [], 'emb_facenets': []}
        #
        for fc in faces:
            face = {'x': int(fc['box'][0]), 'y': int(fc['box'][1]),
                    'w': int(fc['box'][2]), 'h': int(fc['box'][3])}
            jsn['faces'].append(face)
        #
        for face_img in face_imgs:
            face_img_base64_str = get_base64_from_np_image(face_img).decode('ascii')
            jsn['face_imgs'].append(face_img_base64_str)
        #
        for em in emb_insights:
            insight = [float(elem) for elem in em.tolist()]
            jsn['emb_insights'].append(insight)
        #
        for em in emb_facenets:
            facenet = [float(elem) for elem in em.tolist()]
            jsn['emb_facenets'].append(facenet)
        #
        for em in emb_doubles:
            facenet = [float(elem) for elem in em[:128].tolist()]
            jsn['emb_facenets'].append(facenet)
            insight = [float(elem) for elem in em[128:].tolist()]
            jsn['emb_insights'].append(insight)
        #
        json_str = json.dumps(jsn)  #,indent=4)
        self.rabbit.publish( json_str )
    """



    def publish_img_dt(self, img_id, faces, emb_insights, emb_facenets, emb_doubles, face_imgs=[]):
        #jsn = {'img_id': img_id, 'faces': [], 'emb_insights': [], 'emb_facenets': []}
        #if face_imgs:
        jsn = {'img_id': img_id, 'results': []}
        #
        if emb_doubles:
            for fc, face_img, e_d in zip(faces, face_imgs, emb_doubles):
                result = {}
                face = {'x': int(fc['box'][0]), 'y': int(fc['box'][1]),
                        'w': int(fc['box'][2]), 'h': int(fc['box'][3])}
                result['face'] = face
                face_img_base64_str = get_base64_from_np_image(face_img).decode('ascii')
                result['face_img'] = face_img_base64_str

                facenet = [float(elem) for elem in e_d[:128].tolist()]
                result['emb_facenet'] = facenet
                insight = [float(elem) for elem in e_d[128:].tolist()]
                result['emb_insight'] = insight

                jsn['results'].append(result)
        else:
            for fc, face_img, e_i, e_f in zip(faces, face_imgs, emb_insights, emb_facenets):
                result = {}
                face = {'x': int(fc['box'][0]), 'y': int(fc['box'][1]),
                        'w': int(fc['box'][2]), 'h': int(fc['box'][3])}
                result['face'] = face
                face_img_base64_str = get_base64_from_np_image(face_img).decode('ascii')
                result['face_img'] = face_img_base64_str

                insight = [float(elem) for elem in e_i.tolist()]
                result['emb_insight'] = insight
                facenet = [float(elem) for elem in e_f.tolist()]
                result['emb_facenet'] = facenet

                jsn['results'].append(result)
        #
        json_str = json.dumps(jsn)  #,indent=4)
        self.rabbit.publish( json_str )



    """
    # Old implementation
    def publish_img_dt(self, img_id, faces, emb_insights, emb_facenets, emb_doubles):
        out = Out.ImageResponse()
        out.img_id = img_id
        #
        for fc in faces:
            face = out.faces.add()
            face.x = fc[0]
            face.y = fc[1]
            face.w = fc[2]
            face.h = fc[3]
        #
        for em in emb_insights:
            insight = out.insights.add()
            insight.type = "insight"
            for i in range(512):
                insight.emb.append(em[i])
        #
        for em in emb_facenets:
            facenet = out.facenets.add()
            facenet.type = "facenet"
            for i in range(128):
                facenet.emb.append(em[i])
        #
        for em in emb_doubles:
            facenet = out.facenets.add()
            facenet.type = "facenet"
            for i in range(128):
                facenet.emb.append(em[i])
            insight = out.insights.add()
            insight.type = "insight"
            for i in range(128, 640):
                insight.emb.append(em[i])
        #
        self.rabbit.publish( out.SerializeToString() )
        """


def main():
    if len(sys.argv) != 1:
        print("""USAGE: queue_out.py""")
        exit(1)

    r = QueueOut()

################################################################################

if __name__ == "__main__":
    main()

################################################################################
