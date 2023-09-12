import os
import sys
import time
import json


def load_config(json_name='config_main_server.txt'):
    EMB_PROCS = 1

    json_file = open(json_name)
    config_data = json.load(json_file)

    EMB_PROCS = config_data.get('EMB_PROCS', EMB_PROCS)
    IMAGE_SIZE = config_data.get('IMAGE_SIZE', 160)
    NEED_EMBED = config_data.get('NEED_EMBED', True)
    EMBEDDER_BATCH = config_data.get('EMBEDDER_BATCH', 128)
    N_GPUS     = config_data.get('N_GPUS', 0)
    #
    TEST_QUEUE = config_data.get('TEST_QUEUE', '127.0.0.1')
    BASE_QUEUE = config_data.get('BASE_QUEUE', '127.0.0.1')
    #
    TIMEOUT_TO_RESPAWN_IF_NO_PING            = config_data.get('TIMEOUT_TO_RESPAWN_IF_NO_PING', 120)
    TIMEOUT_TO_START_NEXT_PROCESS            = config_data.get('TIMEOUT_TO_START_NEXT_PROCESS', 10)
    TIMEOUT_TO_START_NEXT_PROCESS_IN_RESPAWN = config_data.get('TIMEOUT_TO_START_NEXT_PROCESS_IN_RESPAWN', 3)
    TIMEOUT_TO_REMOVE_FROM_QUEUE             = config_data.get('TIMEOUT_TO_REMOVE_FROM_QUEUE', 240)
    #
    PORT_MON_IN = config_data.get('PORT_MON_IN', 39999)
    PORT_MON_OUT = config_data.get('PORT_MON_OUT', 8088)


    dct_ret = {}
    dct_ret['EMB_PROCS'] = EMB_PROCS
    dct_ret['IMAGE_SIZE'] = IMAGE_SIZE
    dct_ret['NEED_EMBED'] = NEED_EMBED
    dct_ret['EMBEDDER_BATCH'] = EMBEDDER_BATCH
    dct_ret['N_GPUS'] = N_GPUS
    #
    dct_ret['TEST_QUEUE'] = TEST_QUEUE
    dct_ret['BASE_QUEUE'] = BASE_QUEUE
    #
    dct_ret['TIMEOUT_TO_RESPAWN_IF_NO_PING'] = TIMEOUT_TO_RESPAWN_IF_NO_PING
    dct_ret['TIMEOUT_TO_START_NEXT_PROCESS'] = TIMEOUT_TO_START_NEXT_PROCESS
    dct_ret['TIMEOUT_TO_START_NEXT_PROCESS_IN_RESPAWN'] = TIMEOUT_TO_START_NEXT_PROCESS_IN_RESPAWN
    dct_ret['TIMEOUT_TO_REMOVE_FROM_QUEUE'] = TIMEOUT_TO_REMOVE_FROM_QUEUE
    #
    dct_ret['PORT_MON_IN'] = PORT_MON_IN
    dct_ret['PORT_MON_OUT'] = PORT_MON_OUT

    return dct_ret
