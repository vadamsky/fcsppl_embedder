import os
import sys
import time
import json


def load_config(json_name='config_main_server.txt'):
    DET_PROCS = 20
    EMB_PROCS = 1
    subservers = []

    json_file = open(json_name)
    config_data = json.load(json_file)

    DET_PROCS = config_data.get('DET_PROCS', DET_PROCS)
    EMB_PROCS = config_data.get('EMB_PROCS', EMB_PROCS)
    QUE_B_CNT_BY_DET = config_data.get('QUE_B_CNT_BY_DET', 32)
    subservers = config_data.get('SUBSERVERS', subservers)
    IMAGE_SIZE = config_data.get('IMAGE_SIZE', 160)
    S_MIN      = config_data.get('S_MIN', 900)
    PADDING    = config_data.get('PADDING', 0.12)
    NEED_EMBED = config_data.get('NEED_EMBED', True)
    EMBEDDER_BATCH = config_data.get('EMBEDDER_BATCH', 128)
    N_GPUS     = config_data.get('N_GPUS', 0)
    QUEUES_DIR = config_data.get('QUEUES_DIR', './')
    #
    TIMEOUT_TO_RESPAWN_IF_NO_PING            = config_data.get('TIMEOUT_TO_RESPAWN_IF_NO_PING', 120)
    TIMEOUT_TO_START_NEXT_PROCESS            = config_data.get('TIMEOUT_TO_START_NEXT_PROCESS', 10)
    TIMEOUT_TO_START_NEXT_PROCESS_IN_RESPAWN = config_data.get('TIMEOUT_TO_START_NEXT_PROCESS_IN_RESPAWN', 3)
    TIMEOUT_TO_REMOVE_FROM_QUEUE             = config_data.get('TIMEOUT_TO_REMOVE_FROM_QUEUE', 240)
    #
    PORT_IN = config_data.get('PORT_IN', 8002)
    PORT_MON_IN = config_data.get('PORT_MON_IN', 39999)
    PORT_MON_OUT = config_data.get('PORT_MON_OUT', 8088)
    PORT_DET_IN = config_data.get('PORT_DET_IN', 40000)
    PORT_EMB_IN = config_data.get('PORT_EMB_IN', 40100)
    PORT_SRV_IN = config_data.get('PORT_SRV_IN', 40110)
    EMBEDDER_HOST = config_data.get('EMBEDDER_HOST', '95.216.44.199:8002')


    dct_ret = {}
    dct_ret['DET_PROCS'] = DET_PROCS
    dct_ret['EMB_PROCS'] = EMB_PROCS
    dct_ret['QUE_B_CNT_BY_DET'] = QUE_B_CNT_BY_DET
    dct_ret['IMAGE_SIZE'] = IMAGE_SIZE
    dct_ret['S_MIN'] = S_MIN
    dct_ret['PADDING'] = PADDING
    dct_ret['NEED_EMBED'] = NEED_EMBED
    dct_ret['EMBEDDER_BATCH'] = EMBEDDER_BATCH
    dct_ret['N_GPUS'] = N_GPUS
    dct_ret['QUEUES_DIR'] = QUEUES_DIR
    #
    dct_ret['TIMEOUT_TO_RESPAWN_IF_NO_PING'] = TIMEOUT_TO_RESPAWN_IF_NO_PING
    dct_ret['TIMEOUT_TO_START_NEXT_PROCESS'] = TIMEOUT_TO_START_NEXT_PROCESS
    dct_ret['TIMEOUT_TO_START_NEXT_PROCESS_IN_RESPAWN'] = TIMEOUT_TO_START_NEXT_PROCESS_IN_RESPAWN
    dct_ret['TIMEOUT_TO_REMOVE_FROM_QUEUE'] = TIMEOUT_TO_REMOVE_FROM_QUEUE
    #
    dct_ret['PORT_IN'] = PORT_IN
    dct_ret['PORT_MON_IN'] = PORT_MON_IN
    dct_ret['PORT_MON_OUT'] = PORT_MON_OUT
    dct_ret['PORT_DET_IN'] = PORT_DET_IN
    dct_ret['PORT_EMB_IN'] = PORT_EMB_IN
    dct_ret['PORT_SRV_IN'] = PORT_SRV_IN
    dct_ret['EMBEDDER_HOST'] = EMBEDDER_HOST

    return dct_ret
