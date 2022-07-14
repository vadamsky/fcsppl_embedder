import os
import sys
import time
import signal
import subprocess
from subprocess import check_output
from multiprocessing import Lock #,shared_memory
from shared_memory import SharedMemory2
import mmap
import readchar
from threading import Thread, Lock, get_ident

from load_config import load_config
from named_atomic_lock import NamedAtomicLock
from logger import create_or_clear_file

from constants import SHM_PRF
from constants import RUNNED, NEED_STOP_PROC, NEED_STOP_THRD, STOPPED

################################################
# SERVICE FUNCTIONS
def get_pid(name):
    #rez = check_output(['pgrep', '-f', '"'+name+'"'] + opt)
    try:
        rez = check_output(['pgrep -f ' + '"' + name + '"'], shell=True)
    except Exception as e:
        return []
    else:
        rez = str(rez).replace("b'", '').replace("'", "").rstrip("\\n")
        rez = rez.split("\\n")
        return [int(r) for r in rez]
    return []

def check_pid(pid, name):
    try:
        rez = check_output(["ps", "-o", 'args', '-p', pid])
        print('name', type(name), name, pid)
        if rez.find(name) > -1:
            return True
        else:
            return False
    except:
        print('except: ', pid, name)
        return False


def kill_all(name):
    rez = get_pid(name)
    rez = str(rez).replace("b'", "").replace("'", "")
    rez = rez.split("\\n")
    for p in rez:
        time.sleep(1)	
        print(type(p), p)
        if len(p)>0:
            os.system('kill %s' % p)

def kill_all_proc(lpid):
    for p in lpid:
        time.sleep(1)
        print(type(p), p)
        if len(p)>0:
            #print(get_command_pid(p))
            os.system(r'kill %s' % p)

def release_shm(shms):
    for shm in shms:
        shm.close()
        shm.unlink()

def kill_proc(pid):
    os.system('kill -s 9 %d' % pid)

def new_py_proc(f, name_id):
    os.system(r'ps -ax | python3 %s %s &' % (f, name_id))


################################################
# SHM FUNCTIONS
def create_shms(cnf_name):
    json_name = cnf_name
    CONF_DCT = load_config(json_name)
    IMAGE_SIZE, S_MIN, PADDING, EMBEDDER_BATCH = CONF_DCT['IMAGE_SIZE'], CONF_DCT['S_MIN'], CONF_DCT['PADDING'], CONF_DCT['EMBEDDER_BATCH']
    DET_PROCS, QUE_B_CNT_BY_DET = CONF_DCT['DET_PROCS'], CONF_DCT['QUE_B_CNT_BY_DET']

    # Open SHM_stop_processes
    # # 64 int64 values:
    # # - stop all flag (if 1 then stop all processes)
    # # - reserve
    shm_stop = None
    try:
        shm_stop = SharedMemory2(name=SHM_PRF+'stop', create=True, size=8 * 128)
    except Exception:
        # if exist now
        shm_stop = SharedMemory2(name=SHM_PRF+'stop', create=False, size=8 * 128)
    print('shm_stop:', shm_stop)

    # Open SHM A
    # # 3 int64 values:
    # # - last handled id
    # # - last saved id
    # # - reserve
    # # - len of lst of runner_v3_detect
    # # - len of lst of runner_v3_embed
    # # - len of lst of server
    # # - DET_PROCS flags of detectors busyness
    shm_a = None
    shm_a_size = 8 * (6 + DET_PROCS * 2)
    try:
        shm_a = SharedMemory2(name=SHM_PRF+'a', create=True, size=shm_a_size)
    except Exception:
        # if exist now
        shm_a = SharedMemory2(name=SHM_PRF+'a', create=False, size=shm_a_size)
    print('shm_a:', shm_a)

    # Open SHM B
    shm_b = None
    # # 4 blocks of:
    # # - block filling flag (1 byte)
    # # - EMBEDDER_BATCH * req_id (8 bytes)
    # # - EMBEDDER_BATCH * image (IMAGE_SIZE * IMAGE_SIZE * 3 bytes)
    # # - if np.sum(image_160)==0 - no faces detected
    # # - EMBEDDER_BATCH * (rect4+keypoints6+agegender3) (13 * 8 bytes)
    shm_b_size = 4 * 8 + DET_PROCS * QUE_B_CNT_BY_DET * 8 # 4 * (1 + EMBEDDER_BATCH * SHM_B_ONE_SZ)
    print('======== SHM_B_SIZE=', shm_b_size)
    try:
        shm_b = SharedMemory2(name=SHM_PRF+'b', create=True, size=shm_b_size)
    except Exception:
        # if exist now
        shm_b = SharedMemory2(name=SHM_PRF+'b', create=False, size=shm_b_size)
    print('shm_b:', shm_b)

    # Open SHM C
    shm_c = None
    # # 4 blocks of:
    # # - block filling flag (1 byte)
    # # - EMBEDDER_BATCH * req_id (8 bytes)
    # # - EMBEDDER_BATCH * embedding (512 * 8 bytes)
    shm_c_size = 4 * 8 + 4 * EMBEDDER_BATCH * 8  # 4 * (1 + EMBEDDER_BATCH * SHM_C_ONE_SZ)
    try:
        shm_c = SharedMemory2(name=SHM_PRF+'c', create=True, size=shm_c_size)
    except Exception:
        # if exist now
        shm_c = SharedMemory2(name=SHM_PRF+'c', create=False, size=shm_c_size)
    print('shm_c:', shm_c)

    # === FILL SHM'S ===
    x_64 = 0
    # Fill SHM STOP
    for i in range(128):
        shm_stop.buf[i*8:i*8+8] = x_64.to_bytes(8, 'big')
    # Fill SHM A
    for i in range(int(shm_a_size / 8)):
        shm_a.buf[i*8:i*8+8]   = x_64.to_bytes(8, 'big')
    # Fill SHM B
    for i in range(int(shm_b_size / 8)):
        shm_b.buf[i*8:i*8+8]   = x_64.to_bytes(8, 'big') # filling flags
    # Fill SHM C
    for i in range(int(shm_c_size / 8)):
        shm_c.buf[i*8:i*8+8]   = x_64.to_bytes(8, 'big') # filling flags

    # === Create SHM locks ===
    shm_a_lock = NamedAtomicLock(SHM_PRF+'a_lock')
    if shm_a_lock.isHeld:
        print('shm_a_lock is held')
        shm_a_lock.release(forceRelease=True)
    #
    shm_b_lock = NamedAtomicLock(SHM_PRF+'b_lock')
    if shm_b_lock.isHeld:
        print('shm_b_lock is held')
        shm_b_lock.release(forceRelease=True)
    #
    shm_c_lock = NamedAtomicLock(SHM_PRF+'c_lock')
    if shm_c_lock.isHeld:
        print('shm_c_lock is held')
        shm_c_lock.release(forceRelease=True)

    # === Create files ===
    create_or_clear_file('server.csv')
    create_or_clear_file('handler.csv')
    create_or_clear_file('detector.csv')
    create_or_clear_file('server_in.csv')
    create_or_clear_file('runner_embed.csv')

    shms = (shm_stop, shm_a, shm_b, shm_c)
    return shms


    def wait_char_q_and_stop(shms):
        while True:  # making a loop
            #c = readchar.readchar()
            c = readchar.readkey()
            print('Key pressed:', c)
            if c == 'q':  # if key 'q' is pressed 
                print('You Pressed A Key "q"!')
                break  # finishing the loop
            else:
                #break  # if user pressed a key other than the given key the loop will break
                print('You Pressed other Key!')
        stop_flag = 1
        shm_stop.buf[:8] = stop_flag.to_bytes(8, 'big')
        time.sleep(5)
        kill_procs(procs)
        release_shm(shms)



class MonitorProcesses():
    def __init__(self, json_name, proc_strs_lst):
        self.cnf_name = json_name
        self.CONF_DCT = load_config(json_name)
        self.proc_strs_lst = proc_strs_lst
        self.stop = False
        #
        create_shms(json_name)
        #
        print('MonitorProcesses:', self.proc_strs_lst)
        self.last_ping_tms_dct = {}
        tm = time.time()
        for proc_str in self.proc_strs_lst:
            f = proc_str.find('src/')
            proc_st = proc_str[f+4:]
            if proc_st[:8] == 'embedder' or proc_st[:15] == 'runner_v3_embed':
                continue
            self.last_ping_tms_dct[proc_str] = tm

    def check_last_ping_tms(self):
        TIMEOUT_TO_RESPAWN_IF_NO_PING = self.CONF_DCT['TIMEOUT_TO_RESPAWN_IF_NO_PING']
        while not self.stop:
            #print('check_last_ping_tms, stop=', self.stop)
            time.sleep(1)
            if self.stop:
                break
            tm = time.time()
            keys = list(self.last_ping_tms_dct.keys())
            for k in keys:
                if tm - self.last_ping_tms_dct[k] >= TIMEOUT_TO_RESPAWN_IF_NO_PING:
                    print('RESPAWN BY KEY:', k)
                    self.respawn_proc(k)
                    break
        print('Err: MonitorProcesses.check_last_ping_tms stopped')

    def respawn_proc(self, start_string):
        #return
        if self.stop:
            return
        #
        TIMEOUT_TO_START_NEXT_PROCESS_IN_RESPAWN = self.CONF_DCT['TIMEOUT_TO_START_NEXT_PROCESS_IN_RESPAWN']
        self.last_ping_tms_dct[start_string] = time.time()
        pids = get_pid(start_string)
        #
        print('RESPAWN: ', start_string[12:])
        """
        if start_string[12:20] == 'detector':  # or start_string[22:28] == 'detect':
            pids = get_pid(start_string)
            for pid in pids:
                kill_proc(pid)
                print(pid, ' has killed')
            os.system( start_string + ' &' )
            self.ping(start_string)
            #keys = list(self.last_ping_tms_dct.keys())
            # kill all detectors and runner_v3_detect
            #for k in keys:
            #    if k[12:20] == 'detector' or k[22:28] == 'detect':
            #        pids = get_pid(k)
            #        for pid in pids:
            #            kill_proc(pid)
            #            print(pid, ' has killed')
            # respawn all detectors
            #for k in keys:
            #    if k[12:20] == 'detector':
            #        os.system( k + ' &' )
            #        time.sleep(TIMEOUT_TO_START_NEXT_PROCESS_IN_RESPAWN)
            #        for k_ in keys:
            #            self.ping(k_)
            # respawn runner_v3_detect
            #for k in keys:
            #    if k[22:28] == 'detect':
            #        os.system( k + ' &' )
        """
        ## kill and respawn process
        pids = get_pid(start_string)
        for pid in pids:
            kill_proc(pid)
            print(pid, ' has killed')
            self.ping(start_string)
            os.system( start_string + ' &' )

        self.last_ping_tms_dct[start_string] = time.time()
        print(start_string + ' has respawned')
        print(self.last_ping_tms_dct)


    def kill_all_procs(self):
        for proc_str in self.proc_strs_lst:
            pids = get_pid(proc_str)
            print(proc_str)
            print(pids, type(pids))
            #time.sleep(1)
            for pid in pids:
                kill_proc(pid)
            self.last_ping_tms_dct[proc_str] = time.time()
        time.sleep(10)
        for proc_str in self.proc_strs_lst:
            pids = get_pid(proc_str)
            print(pids, type(pids))
            for pid in pids:
                kill_proc(pid)
            self.last_ping_tms_dct[proc_str] = time.time()

    def start_all_procs(self):
        TIMEOUT_TO_START_NEXT_PROCESS = self.CONF_DCT['TIMEOUT_TO_START_NEXT_PROCESS']
        for proc_str in self.proc_strs_lst:
            if self.stop:
                return
            os.system( proc_str + ' &' )
            pids = get_pid(proc_str)
            print(pids, type(pids))
            #print('%s    %s' % (proc_str, pid))
            time.sleep(TIMEOUT_TO_START_NEXT_PROCESS)
            if self.stop:
                return
            for proc_str_ in self.proc_strs_lst:
                if True:  # proc_str_.find('embed') == -1:
                    self.last_ping_tms_dct[proc_str_] = time.time()

    def run(self):
        print('MonitorProcesses: run:', self.proc_strs_lst)
        print(self.CONF_DCT)
        TIMEOUT_TO_START_NEXT_PROCESS = self.CONF_DCT['TIMEOUT_TO_START_NEXT_PROCESS']
        #start_all(self.cnf_name)
        self.start_all_procs()
        #
        self.thread_check_last_ping_tms = Thread(target = self.check_last_ping_tms)
        self.thread_check_last_ping_tms.start()
        while not self.stop:
            #print('run, stop=', self.stop)
            time.sleep(1)
        print('MonitorProcesses.run pre-stopped')
        self.thread_check_last_ping_tms.join()
        print('MonitorProcesses.run stopped')

    def ping(self, start_string):
        self.last_ping_tms_dct[start_string] = time.time()
        #print('ping:', start_string, time.time())
