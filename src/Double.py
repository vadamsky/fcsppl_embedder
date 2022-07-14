import os
import numpy as np

import tensorflow as tf
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
tf_version = int(tf.__version__.split(".")[0])
print('tf_version:', tf_version)


#from keras.layers.core import Dense
#from keras.layers import Flatten
#from keras.layers import Input
#from keras.layers import concatenate
#from keras.models import Model

if tf_version == 1:
    from keras.models import Model
    from keras.layers import concatenate
    from keras.layers import Dense
    from keras.layers import Input
else:
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import concatenate
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import Input


import Facenet
import Insight


def loadModel(insight_weights_fname='./weights/glint360k_cosface_r100_fp16_0.1.h5'):
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

    df_model = Facenet.loadModel()
    if_model = Insight.loadModel(insight_weights_fname)

    # combinedOutput = tf.keras.layers.Concatenate(axis=1)([np.array([[0, 1]]), np.array([[2, 3, 4, 5]])])
    # combinedOutput = concatenate([np.array([[0, 1]]), np.array([[2, 3, 4, 5]])], axis=1)
    combinedOutput = concatenate([df_model.output, if_model.output], axis=1)
    #combinedOutput = if_model.output

    model = Model(inputs=[df_model.input, if_model.input], outputs=combinedOutput)
    #model = Model(inputs=df_model.input, outputs=combinedOutput)

    return model
