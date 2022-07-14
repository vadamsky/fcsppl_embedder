import sys
import numpy as np
import time
from datetime import datetime
from threading import Thread, Lock
from monitor_processes import MonitorProcesses
from monitor_config import TIME_TO_SHOW
from load_config import load_config
#from zmq_wrapper import ZmqWrapper, MON_HOST
from shm_wrapper_t import ShmWrapperT


ARR_SZ = 8

"""
read_time
parse_time
srv_2_detrunner_time
in_detrunner_waiting_time
detrunner_2_detector_time
detect_time
detector_2_embedder_time
embed_time
embedder_2_srv_time
formed_time

dict_stat:
  v3:
    read_time
    start_parse_tm
    parse_time
    saved_tm    ?
    ------
    pre_formed_return_tm
    formed_return_tm
  v3_batch:
    read_time
    start_parse_tm
    parse_time
    saved_tm    ?
    ------
    pre_formed_return_tm
    formed_return_tm
  runner_v3_detect Main:
    det_load_from_saved_tm
    saved_queue_len    100500 if func push_quarter_queue_for_all_detectors_test()
    ----
    qsize
    det_put_to_internal_queue_tm
  runner_v3_detect Thread:
    det_get_from_internal_queue_tm
  detector_base2:
    det_detector_loaded_tm
    det_detected_tm
    det_save_in_detected_tm    ?
  runner_v3_embed:
    emb_load_from_detected_tm
    detected_queue_len
    ----
    emb_begin_save_in_resulted_tm
    emb_save_in_resulted_tm    ?
"""

class ValuesStack():
    def __init__(self):
        #self.name = name
        self.tms = np.zeros(ARR_SZ * 20, dtype=np.float64)
        #self.typs = np.zeros(ARR_SZ, dtype=np.str_)
        self.vals = np.zeros(ARR_SZ * 20, dtype=np.float64)
        self.shft = 0

    def add_value(self, tm, msg_type, value):
        if self.shft >= (20 * ARR_SZ):
            self.tms[:19 * ARR_SZ] = self.tms[ARR_SZ:]
            #self.typs[:19 * ARR_SZ] = self.typs[ARR_SZ:]
            self.vals[:19 * ARR_SZ] = self.vals[ARR_SZ:]
            self.shft = 19 * ARR_SZ
        #if self.shft == len(self.tms):
        #    self.tms = self.tms.resize(self.shft + ARR_SZ)
        #    #self.typs = self.typs.resize(self.shft + ARR_SZ)
        #    self.vals = self.vals.resize(self.shft + ARR_SZ)
        self.tms[self.shft] = tm
        #self.typs[self.shft] = msg_type
        self.vals[self.shft] = value
        self.shft += 1

    def get_last_value_wo_rm(self):
        if self.shft == 0:
            return None
        return self.vals[self.shft - 1]

    def modify_last_value(self, value):
        if self.shft == 0:
            return False
        self.vals[self.shft - 1] = value
        return True

    def get_values(self, group):
        if self.shft == 0:
            now = time.time()
            now_dt = datetime.now()
            now_str = now_dt.strftime("%Y-%m-%d %H:%M:%S")
            return now - 5, now + 5, [(now_str, 0, group)]
        #
        ret = []
        max_tm = float(np.max(self.tms[:self.shft])) + 5
        for i in range(self.shft):
            if self.tms[i] < (max_tm - 5 - TIME_TO_SHOW):
                continue
            dt = datetime.fromtimestamp(float(self.tms[i]))
            ret.append((dt.strftime("%Y-%m-%d %H:%M:%S"), float(self.vals[i]), group))
        #min_tm = float(np.min(self.tms[:self.shft])) - 5
        min_tm = max_tm - 5 - 5 - TIME_TO_SHOW
        return min_tm, max_tm, ret

    def get_last_tm(self):
        if self.shft > 0:
            return self.tms[self.shft - 1]
        return 0


class Monitor_Zmq():
    def __init__(self, json_name, proc_strs_lst):
        self.stop = False
        # CONTROLLER
        self.v_in_ld,  self.v_in_batch_ld  = ValuesStack(), ValuesStack()  # in loadings
        self.v_out_ld, self.v_out_batch_ld = ValuesStack(), ValuesStack()  # out loadings
        self.v_q_svd, self.v_q_dtd, self.v_q_rsd = ValuesStack(), ValuesStack(), ValuesStack()  # queue lengths
        self.v_q_in_detrunner = ValuesStack()  # NOT Controller queue length (in runner_v3_detect)
        #self.v_shm_b_len = ValuesStack()  # SHM_B queue length
        #self.v_shm_c_len = ValuesStack()  # SHM_C queue length
        self.v_root, self.v_home = ValuesStack(), ValuesStack()  # root and home loading
        # HANDLERS
        self._in_ld_cnt,  self._in_batch_ld_cnt  = 0, 0
        self._out_ld_cnt, self._out_batch_ld_cnt = 0, 0
        # RUNNER DETECT
        self._det_dct = {}
        self.v_det_all_ld, self.v_det_all_tm = ValuesStack(), ValuesStack()  # mean detectors loading, mean detectors tm per image
        self.v_det_pip_tm, self.v_det_onl_tm = ValuesStack(), ValuesStack()  # 
        # RUNNER EMBED
        self._in_emb_cnt = 0
        self._emb_tm, self._emb_tm_start = 0, 0
        self._in_det_cnt,  self._out_det_cnt = 0, 0
        #self.v_emb_ld, self.v_emb_tm = ValuesStack(), ValuesStack()  # mean embedder loading, mean embedder tm per image
        # STATISTIC
        self.v_sock_rd, self.v_parse, self.srv2detr, self.detr_wait = ValuesStack(), ValuesStack(), ValuesStack(), ValuesStack()
        self.detr2det, self.det, self.det2emb, self.emb = ValuesStack(), ValuesStack(), ValuesStack(), ValuesStack()
        self.emb2srv, self.response = ValuesStack(), ValuesStack()
        self.stat_vs = [self.v_sock_rd, self.v_parse, self.srv2detr, self.detr_wait, self.detr2det, self.det, self.det2emb, self.emb, self.emb2srv, self.response]
        self.stat_keys = ['read_time', 'parse_time', '0', '1', '2', '3', '4', '5', '6', '7']
        self.stat_d_keys = [
            ('saved_tm', 'det_load_from_saved_tm'), ('det_load_from_saved_tm', 'det_get_from_internal_queue_tm'),
            ('det_get_from_internal_queue_tm', 'det_detector_loaded_tm'), ('det_detector_loaded_tm', 'det_detected_tm'),
            ('det_detected_tm', 'emb_load_from_detected_tm'), ('emb_load_from_detected_tm', 'emb_begin_save_in_resulted_tm'),
            ('emb_begin_save_in_resulted_tm', 'srv_received_from_emb_tm'), ('pre_formed_return_tm', 'formed_return_tm')
        ]
        self.stat_values_cnt = 0
        #
        self.one_sec_lock = Lock()
        self.one_sec_thread = Thread(target = self.one_second_timer_func)
        self.one_sec_thread.start()
        #
        self.mon_proc = MonitorProcesses(json_name, proc_strs_lst)
        CONF_DCT = load_config(json_name)
        #self.zmq_mon = ZmqWrapper(identifier='monitor', addr=MON_HOST+':%d'%CONF_DCT['PORT_MON_IN'], tp='server', resp_type='with_response')
        self.zmq_mons = []
        self.zmq_mons.append(ShmWrapperT('controller', 'monitor', 4096))
        self.zmq_mons.append(ShmWrapperT('server3_mt', 'monitor', 4096))
        self.zmq_mons.append(ShmWrapperT('handler_v1_mt', 'monitor', 4096))
        self.zmq_mons.append(ShmWrapperT('handler_v2_mt', 'monitor', 4096))
        self.zmq_mons.append(ShmWrapperT('handler_v3_mt', 'monitor', 4096))
        self.zmq_mons.append(ShmWrapperT('handler_v3_mt_batch', 'monitor', 4096))
        self.zmq_mons.append(ShmWrapperT('runner_v3_detect', 'monitor', 4096))
        self.zmq_mons.append(ShmWrapperT('runner_v3_embed', 'monitor', 4096))
        self.zmq_mons.append(ShmWrapperT('embedder', 'monitor', 4096))
        for detector_id in range(CONF_DCT['DET_PROCS']):
            self.zmq_mons.append(ShmWrapperT(f'detector_{detector_id}', 'monitor', 4096))
            self.zmq_mons.append(ShmWrapperT(f'detect_thread_{detector_id}', 'monitor', 4096))

    def run(self):
        #self.zmq_mon.run_serv(self.callback_func)
        for zmq_mon in self.zmq_mons:
            zmq_mon_thr = Thread(target=zmq_mon.run_serv, args=(self.callback_func, ))
            zmq_mon_thr.start()
        while True:
            time.sleep(1)


    def callback_func(self, dt):
        (identifier, tm, msg_type, data) = dt
        #print('----------', identifier, msg_type, data)
        #
        self.mon_proc.ping(data['start_string'])
        if identifier[:10] == 'test_proc_':
            num = int(identifier[10:])
            print('callback_func:', num)
            self.mon_proc.ping(data['start_string'])
        #
        if identifier == 'controller':
            if data['act'] == 'root_home_loading':
                self.v_root.add_value(tm, '', data['root_loading'])
                self.v_home.add_value(tm, '', data['home_loading'])
            if data['act'] == 'len_fpaths':
                self.v_q_svd.add_value(tm, '', data['len_runner_detect'])
                self.v_q_dtd.add_value(tm, '', data['len_runner_embed'])
                self.v_q_rsd.add_value(tm, '', data['len_server'])
        #
        if identifier == 'handler_v3_mt':
            self.one_sec_lock.acquire()
            if msg_type == 'get_return_sync' and data['act'] == 'end':
                self._out_ld_cnt += 1
            self.one_sec_lock.release()
        #
        if identifier == 'handler_v3_mt_batch':
            self.one_sec_lock.acquire()
            if msg_type == 'get_answer' and data['act'] == 'end':
                self._in_batch_ld_cnt += data['ids_cnt']
            if msg_type == 'get_answer_get_n' and data['act'] == 'end':
                self._out_batch_ld_cnt += data['len_result']
            self.one_sec_lock.release()
        #
        if identifier in ['handler_v3_mt', 'handler_v3_mt_batch']:
            dict_stat = data.get('dict_stat', None)
            if not (dict_stat is None):
                new_value_flag = (time.time() - self.v_sock_rd.get_last_tm() >= 5)
                for v, key in zip(self.stat_vs, self.stat_keys):
                    value = 0
                    if len(key) > 2:  # single key
                        value = dict_stat[key]
                    else:
                        key = int(key)
                        value = dict_stat[self.stat_d_keys[key][1]] - dict_stat[self.stat_d_keys[key][0]]
                    #
                    old_value = v.get_last_value_wo_rm()
                    if new_value_flag:
                        #if old_value and self.stat_values_cnt:
                        #if old_value:
                        #    v.modify_last_value(old_value / self.stat_values_cnt)
                        v.add_value(tm, '', value)
                    else:
                        if old_value:
                            v.modify_last_value((old_value * self.stat_values_cnt + value) / (self.stat_values_cnt + 1))
                #
                if new_value_flag:
                    self.stat_values_cnt = 1
                else:
                    self.stat_values_cnt += 1
                    
        #
        if identifier == 'runner_v3_embed':
            pass
            ##print('----------', identifier, data)
            #self.one_sec_lock.acquire()
            #if data['act'] == 'pre get_embs':
            #    self._in_emb_cnt += data['len(self.imgs)']
            #    self._emb_tm_start = tm
            #if data['act'] == 'post get_embs':
            #    self._emb_tm += (tm - self._emb_tm_start)
            #self.one_sec_lock.release()
        #
        if identifier == 'runner_v3_detect' or identifier[:13] == 'detect_thread':
            #print('----------', identifier, data)
            #
            # det_tm: start_tm, start_det_tm, cum_cnt, cum_tm_det_only, cum_tm_det+pipes, cum_tm_all
            if data['act'] in ['item received', 'img handled', 'img saved']:
                self.one_sec_lock.acquire()
                detector_id = data['det_id']
                det_dt = self._det_dct.get(detector_id, None)
                if det_dt is None:
                    det_dt = [tm, 0, 0, 0, 0, 0]
                if data['act'] == 'item received':
                    det_dt[0] = tm
                if data['act'] == 'img handled':
                    if det_dt[1] == 0: # None or "img handled" recv earlier
                        det_dt = [0, 0, 0, 0, 0, 0]
                    else:
                        det_dt[4] += (tm - det_dt[1])
                        det_dt[1] = 0
                if data['act'] == 'img saved':
                    if det_dt[0] == 0:
                        det_dt = [0, 0, 0, 0, 0, 0]
                    else:
                        det_dt[2] += 1
                        det_dt[5] += (tm - det_dt[0])
                        det_dt[0] = 0
                #
                self._det_dct[detector_id] = det_dt
                #print(self._det_dct)
                self.one_sec_lock.release()

            if data['act'] == 'self.q.qsize()' or data['act'] == 'fpaths cnt':
                if tm - self.v_q_in_detrunner.get_last_tm() > 0.8:
                    self.v_q_in_detrunner.add_value(tm, '', data['self.q.qsize()'])
                    #self.v_shm_b_len.add_value(tm, '', data['shm_b_filled'])
                    #self.v_shm_c_len.add_value(tm, '', data['shm_c_filled'])
        #
        if identifier[:8] == 'detector':
            #print('----------', identifier, data)
            if data['act'] in ['img from pipe received', 'img handled', 'img returned']:
                self.one_sec_lock.acquire()
                detector_id = data['det_id']
                det_dt = self._det_dct.get(detector_id, None)
                if det_dt is None:
                    det_dt = [0, 0, 0, 0, 0, 0]
                if data['act'] == 'img from pipe received':
                    det_dt[1] = tm
                if data['act'] == 'img handled':
                    if det_dt[1] == 0: # None or "img handled" recv earlier
                        det_dt = [0, 0, 0, 0, 0, 0]
                    else:
                        det_dt[3] += (tm - det_dt[1])
                if data['act'] == 'img returned':
                    pass
            #
            if data['act'] in ['img from pipe received', 'img handled', 'img returned']:
                self._det_dct[detector_id] = det_dt
                #print(self._det_dct)
                self.one_sec_lock.release()
        #
        #print(identifier, tm, msg_type, data)


    def one_second_timer_func(self):
        while not self.stop:
            tm = time.time()
            self.one_sec_lock.acquire()
            #
            # HANDLER_V3_MT
            #self.v_in_ld.add_value(tm, '', self._in_ld_cnt)
            self.v_out_ld.add_value(tm, '', self._out_ld_cnt)
            #
            # HANDLER_V3_MT_BATCH
            self.v_in_batch_ld.add_value(tm, '', self._in_batch_ld_cnt)
            self.v_out_batch_ld.add_value(tm, '', self._out_batch_ld_cnt)
            #
            # RUNNER_V3_DETECT
            sum_det_cnt, sum_det_onl_tm, sum_det_pip_tm, sum_det_all_tm = 0, 0, 0, 0
            for key in self._det_dct:
                det_dt = self._det_dct[key]
                sum_det_cnt += det_dt[2]
                sum_det_onl_tm  += det_dt[3]
                sum_det_pip_tm  += det_dt[4]
                sum_det_all_tm  += det_dt[5]
                det_dt[2], det_dt[3], det_dt[4], det_dt[5] = 0, 0, 0, 0
                self._det_dct[key] = det_dt
            self.v_det_all_ld.add_value(tm, '', sum_det_cnt)
            self.v_det_onl_tm.add_value(tm, '', 0. if sum_det_cnt == 0 else sum_det_onl_tm / sum_det_cnt)
            self.v_det_pip_tm.add_value(tm, '', 0. if sum_det_cnt == 0 else sum_det_pip_tm / sum_det_cnt)
            self.v_det_all_tm.add_value(tm, '', 0. if sum_det_cnt == 0 else sum_det_all_tm / sum_det_cnt)
            #
            # RUNNER_V3_EMBED
            #self.v_emb_ld.add_value(tm, '', self._in_emb_cnt)
            #self.v_emb_tm.add_value(tm, '', 0. if self._in_emb_cnt == 0 else self._emb_tm / self._in_emb_cnt)
            #
            # NULLIFICATOR
            self._in_ld_cnt,  self._in_batch_ld_cnt  = 0, 0
            self._out_ld_cnt, self._out_batch_ld_cnt = 0, 0
            self._in_emb_cnt = 0
            self._emb_tm = 0
            #
            self.one_sec_lock.release()
            #print('================\r\n', self._det_dct)
            time.sleep(1)

    def get_cur_labels_items(self):
        # grph_names = [['root', 'home'], ['In queue', 'Detected queue', 'Out queue']]
        min_tm, max_tm = time.time() + 100500, 0
        cur_labels = []
        cur_items = []
        
        # CONTROLLER
        min_tm_0, max_tm_0, ret_0 = self.v_root.get_values(0)
        min_tm_1, max_tm_1, ret_1 = self.v_home.get_values(1)
        #min_tm = min(min_tm, min_tm_0, min_tm_1)
        #max_tm = max(max_tm, max_tm_0, max_tm_1)
        #add_items = ret_0 + ret_1
        #cur_items.append(add_items)
        cur_labels.append([ret_0[-1][1], ret_1[-1][1]])
        #
        min_tm_0, max_tm_0, ret_0 = self.v_q_svd.get_values(0)
        min_tm_1, max_tm_1, ret_1 = self.v_q_dtd.get_values(1)
        min_tm_2, max_tm_2, ret_2 = self.v_q_rsd.get_values(2)
        min_tm_3, max_tm_3, ret_3 = self.v_q_in_detrunner.get_values(3)
        #min_tm_4, max_tm_4, ret_4 = self.v_shm_b_len.get_values(4) # v_bad_saved_fnames
        #min_tm_5, max_tm_5, ret_5 = self.v_shm_c_len.get_values(5) # v_bad_saved_fnames
        min_tm = min(min_tm, min_tm_0, min_tm_1, min_tm_2, min_tm_3)
        max_tm = max(max_tm, max_tm_0, max_tm_1, max_tm_2, max_tm_3)
        add_items = ret_0 + ret_1 + ret_2 + ret_3
        cur_items.append(add_items)
        #
        # HANDLER_V3_MT & HANDLER_V3_MT_BATCH
        min_tm_0, max_tm_0, ret_0 = self.v_out_ld.get_values(0)
        min_tm_1, max_tm_1, ret_1 = self.v_in_batch_ld.get_values(1)
        min_tm_2, max_tm_2, ret_2 = self.v_out_batch_ld.get_values(2)
        min_tm = min(min_tm, min_tm_0, min_tm_1, min_tm_2)
        max_tm = max(max_tm, max_tm_0, max_tm_1, max_tm_2)
        add_items = ret_0 + ret_1 + ret_2
        cur_items.append(add_items)
        #
        # RUNNERS
        min_tm_0, max_tm_0, ret_0 = self.v_det_all_ld.get_values(0)
        #min_tm_1, max_tm_1, ret_1 = self.v_emb_ld.get_values(1)
        min_tm = min(min_tm, min_tm_0)  # , min_tm_1)
        max_tm = max(max_tm, max_tm_0)  # , max_tm_1)
        add_items = ret_0  # + ret_1
        cur_items.append(add_items)
        #
        #min_tm_0, max_tm_0, ret_0 = self.v_det_onl_tm.get_values(0)
        #min_tm_1, max_tm_1, ret_1 = self.v_det_pip_tm.get_values(1)
        #min_tm_2, max_tm_2, ret_2 = self.v_det_all_tm.get_values(2)
        ##min_tm_1, max_tm_1, ret_1 = self.v_emb_tm.get_values(1)
        #min_tm = min(min_tm, min_tm_0, min_tm_1, min_tm_2)  # , min_tm_1)
        #max_tm = max(max_tm, max_tm_0, max_tm_1, max_tm_2)  # , max_tm_1)
        #add_items = ret_0 + ret_1 + ret_2  # + ret_1
        tms_rets = [self.stat_vs[i].get_values(i) for i in range(len(self.stat_vs))]
        min_tm = min([tm_ret[0] for tm_ret in tms_rets])
        max_tm = max([tm_ret[1] for tm_ret in tms_rets])
        add_items = tms_rets[0][2]
        for tm_ret in tms_rets[1:]:
            add_items += tm_ret[2]
        cur_items.append(add_items)
        #
        #
        #print('====', self.v_home.shft, self.v_home.vals)
        #self.one_sec_lock.acquire()
        #self.one_sec_lock.release()
        #min_tm_str = datetime.fromtimestamp(min_tm).strftime("%Y-%m-%d %H:%M:%S")
        min_tm_str = datetime.fromtimestamp(max(min_tm, max_tm - 120)).strftime("%Y-%m-%d %H:%M:%S")
        max_tm_str = datetime.fromtimestamp(max_tm + 5).strftime("%Y-%m-%d %H:%M:%S")
        return min_tm_str, max_tm_str, cur_labels, cur_items

"""
detector:
self.zmq_mon.send({'act': 'img from pipe received', 'req_id': req_id, 'image_np.shape': image_np.shape, 'det_id': self.detector_id}, msg_type='run')
self.zmq_mon.send({'act': 'img handled', 'req_id': req_id, 'det_id': self.detector_id}, msg_type='run')
self.zmq_mon.send({'act': 'img returned', 'req_id': req_id, 'det_id': self.detector_id}, msg_type='run')

handler_v3_mt:
self.zmq_mon.send({'act': 'start', 'req_id': req_id}, msg_type='get_answer_sync')
self.zmq_mon.send({'act': 'start'}, msg_type='parse_request')
self.zmq_mon.send({'act': 'end', 'req_id': req_id}, msg_type='get_return_sync')

handler_v3_mt_batch:
self.zmq_mon.send({'act': 'start'}, msg_type='parse_request')
self.zmq_mon.send({'act': 'end', 'req_id': req_id, 'ids_cnt': ids_cnt, 'id_strs': img_ids}, msg_type='get_answer')
self.zmq_mon.send({'act': 'start', 'req_id': req_id}, msg_type='get_answer_get')
self.zmq_mon.send({'act': 'start', 'cid': cid, 'n': n, 'max_wait_seconds': max_wait_seconds}, msg_type='get_answer_get_n')
# self.zmq_mon.send({'act': 'pre_cid_check', 'len_resulted': len(fpaths)}, msg_type='get_answer_get_n')
self.zmq_mon.send({'act': 'end', 'len_result': len(lst), 'id_strs': [dct["key"] for dct in lst]}, msg_type='get_answer_get_n')

runner_v3_detect: Thread
self.zmq_mon.send({'act': 'item received', 'req_id': req_id, 'img_id': img_id, 'det_id': self.detector_id}, msg_type='run')
self.zmq_mon.send({'act': 'img handled', 'req_id': req_id, 'img_id': img_id, 'det_id': self.detector_id}, msg_type='run')
runner_v3_detect: Main
self.zmq_mon.send({'act': 'self.q.qsize()', 'self.q.qsize()': self.q.qsize()}, msg_type='run')
self.zmq_mon.send({'act': 'fpaths cnt', 'len_saved': len(fpaths)}, msg_type='run')

runner_v3_embed:
self.zmq_mon.send({'act': 'len_fpaths', 'len_detected': len(fpaths)}, msg_type='run')
self.zmq_mon.send({'act': 'pre get_embs', 'len(self.imgs)': len(self.imgs)}, msg_type='run')
self.zmq_mon.send({'act': 'post get_embs', 'embs is None': (embs is None)}, msg_type='run')
self.zmq_mon.send({'act': 'continue'}, msg_type='run')

controller:
self.zmq_mon.send({'act': 'len_fpaths', 'len_saved': len(fpaths_svd), 'len_detected': len(fpaths_dtd), 'len_resulted': len(fpaths_rsd)}, msg_type='run')
self.zmq_mon.send({'act': 'root_home_loading', 'root_loading': root_loading, 'home_loading': home_loading}, msg_type='run')

"""


def main():
    if len(sys.argv) != 2:
        print("""USAGE: monitor_zmq.py cnf_file_name
EXAMPLE: monitor_zmq.py config_video003.txt""")
        exit(1)

    r = Monitor_Zmq(sys.argv[1])
    r.run()

################################################################################

if __name__ == "__main__":
    main()

################################################################################
