# https://github.com/leondgarse/Keras_insightface

import os

import tensorflow as tf
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
tf_version = int(tf.__version__.split(".")[0])
print('tf_version:', tf_version)

if tf_version == 1:
    from keras.models import Model
    from keras.layers import Resizing
    from keras.layers import Dense
    from keras.layers import Input
else:
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Resizing
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import Input

import keras


def loadModel(weights_fname='./weights/glint360k_cosface_r50_fp16_0.1.h5'):
    if tf_version == 1:
        tf.config.gpu.set_per_process_memory_growth(True)
        #config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
        #config.gpu_options.allow_growth=True # It works!
        #self.sess = tf.Session(config=config)
    if tf_version == 2:
        physical_devices = tf.config.list_physical_devices('GPU') 
        print('physical_devices:', physical_devices)
        for gpu_instance in physical_devices: 
            tf.config.experimental.set_memory_growth(gpu_instance, True)

    inputs = Input(shape=(160, 160, 3))
    #inputs = Input(shape=(112, 112, 3))
    x = (inputs - 127.5) / 128
    x = Resizing(112, 112, interpolation="bicubic", crop_to_aspect_ratio=False)(x)
    if_model = keras.models.load_model(weights_fname, compile=False)
    x = if_model(x)
    norm = tf.math.sqrt(tf.keras.backend.sum(tf.math.square(x), axis=1))
    norm = tf.transpose(tf.tensordot(tf.fill((512), 1.), norm, axes=0))
    x = x / norm

    model = Model(inputs=inputs, outputs=x)

    return model
