#!/usr/bin/env python3

import time
import sys
import os
import pickle
#from multiprocessing import shared_memory
#import shared_memory
from constants import SHM_PRF

from conversions import int_to_bytes, int_from_bytes
from constants import RUNNED, NEED_STOP_PROC, NEED_STOP_THRD, STOPPED


class PipesWorker():
    def __init__(self, pipein_name, pipeout_name, blocked=True, timeout=.003, proc_num=-1):
        self.proc_num = proc_num
        self.blocked = blocked
        self.timeout = timeout
        self.pipein_name = pipein_name
        self.pipeout_name = pipeout_name

        self.pipein = None
        if pipein_name != '' and not pipein_name is None:
            if not os.path.exists(pipein_name):
                os.mkfifo(pipein_name)  
                print('PipesWorker created %s' % pipein_name)
            pipein = os.open(pipein_name, os.O_RDWR)
            self.pipein = open(pipein, 'rb')
            if not blocked:
                os.set_blocking(self.pipein.fileno(), False)

        self.pipeout = None
        if pipeout_name != '' and not pipeout_name is None:
            if not os.path.exists(pipeout_name):
                os.mkfifo(pipeout_name)  
                print('PipesWorker created %s' % pipeout_name)
            if not blocked:
                self.pipeout = os.open(pipeout_name, os.O_RDWR | os.O_NONBLOCK)
            else:
                self.pipeout = os.open(pipeout_name, os.O_RDWR)

        self.readed_bts_buf = b''
        #
        # Open SHM_stop_processes
        # # 128 int64 values:
        # # - stop all flag (if 1 then stop all processes)
        # # - stop processes by num
        self.stop_proc_flag_b = False
        self.stop_proc_flag_a = False
        self.shm_stop = None
        ####try:
        ####    self.shm_stop = shared_memory.SharedMemory2(name=SHM_PRF+'stop')
        ####except Exception:
        ####    self.shm_stop = shared_memory.SharedMemory2(name=SHM_PRF+'stop', create=True, size=8 * 128)

    def __del__(self):
        for path in [self.pipein_name, self.pipeout_name]:
            try:
                os.remove(path)
            except Exception:
                print('Cant delete %s' % path)
                pass

    def need_stop_all_processes(self):
        if self.stop_proc_flag_b:
            return self.stop_proc_flag_b
        stop_proc_flag = False  # int.from_bytes(self.shm_stop.buf[0:8], 'big')
        if stop_proc_flag:
            print('PipesWorker need_stop_all_processes() == True')
            self.stop_proc_flag_b = bool(stop_proc_flag)
        return self.stop_proc_flag_b

    def need_stop_this_process(self, proc_num=-1):
        # This proc only
        stop_this_proc_flag = False
        if (self.proc_num) >= 0 or (proc_num >= 0):
            p_n = proc_num if proc_num >= 0 else self.proc_num
            #print('p_n', p_n)
            shft = 8 + 8 * p_n
            stop_this_proc_flag = int.from_bytes(self.shm_stop.buf[shft:shft+8], 'big')
            #print('p_n', p_n, '  stop_this_proc_flag', stop_this_proc_flag)
            if (stop_this_proc_flag == NEED_STOP_PROC):# or (stop_this_proc_flag == NEED_STOP_PROC + NEED_STOP_THRD):
                print('PipesWorker need_stop_this_process == True')
            #time.sleep(0.5)
        return (stop_this_proc_flag == NEED_STOP_PROC)# or (stop_this_proc_flag == NEED_STOP_PROC + NEED_STOP_THRD)

    def notrunned_this_process(self, proc_num=-1):
        # This proc only
        stop_this_proc_flag = 0
        if (self.proc_num) >= 0 or (proc_num >= 0):
            p_n = proc_num if proc_num >= 0 else self.proc_num
            shft = 8 + 8 * p_n
            stop_this_proc_flag = int.from_bytes(self.shm_stop.buf[shft:shft+8], 'big')
        return bool(stop_this_proc_flag != RUNNED)


    def _read_from_pipein(self, length):
        if self.need_stop_all_processes() or self.need_stop_this_process():
            return None
        #print('self.pipein:', self.pipein, '  length: ', length, '  self.readed_bts_buf: ', self.readed_bts_buf, ' len(self.readed_bts_buf):', len(self.readed_bts_buf))
        if self.blocked:
            readed = self.pipein.read(length)
            self.readed_bts_buf += readed
        else:
            while len(self.readed_bts_buf) < length:
                if self.need_stop_all_processes() or self.need_stop_this_process():
                    return None
                readed = b''
                readed = self.pipein.read()
                if not readed is None:
                    self.readed_bts_buf += readed
                if len(self.readed_bts_buf) < length:
                    time.sleep(self.timeout)
        ret = self.readed_bts_buf[:length]
        self.readed_bts_buf = self.readed_bts_buf[length:]
        return ret

    #def _read_from_pipein(self, length):
    #    readed = self.pipein.read()
    #    if not readed is None:
    #        self.readed_bts_buf += readed
    #
    #    if len(self.readed_bts_buf) >= length:
    #        ret = self.readed_bts_buf[:length]
    #        self.readed_bts_buf = self.readed_bts_buf[length:]
    #        return ret
    #    return None

    #def _read_from_pipein_wait(self, length):
    #    readed = self._read_from_pipein(length)
    #    while readed is None:
    #        time.sleep(1. / 50000)
    #        readed = self._read_from_pipein(length)
    #    return readed

    def _read_from_pipein_simple(self, length=0):
        if length == 0:
            return self.pipein.read()
        return self.pipein.read(length)

    def _write_in_pipeout(self, bts):
        written = os.write(self.pipeout, bts)
        #print('self.pipeout:', self.pipeout, '  written: ', written, '  bts: ', bts)
        #written = 0
        while written < len(bts):
            if self.need_stop_all_processes() or self.need_stop_this_process():
                return None
            #n = self.pipeout.write(bts[written:])
            n = None
            try:
                n = os.write(self.pipeout, bts[written:])
            except Exception:
                #print('_write_in_pipeout except, n=', n)
                #time.sleep(.5)
                pass
            if n is None:
                time.sleep(self.timeout)
            else:
                written += n
            #print('written: ', written, '  bts: ', bts)
        return written

    # STRUCT
    def write_in_pipeout_strct(self, strct):
        verse = pickle.dumps(strct)
        ret = self._write_in_pipeout(int_to_bytes(len(verse)))
        if ret is None:
            return None
        ret = self._write_in_pipeout(verse)
        if ret is None:
            return None
        return ret

    def read_from_pipein_strct(self):
        line = self._read_from_pipein(4)
        if line is None:
            return None
        length = int_from_bytes(line)
        #print('length:', length)
        if length == 0:
            return ''
        #print('Embedder length:', length)
        verse = self._read_from_pipein(length)
        if verse is None:
            return None
        strct = pickle.loads(verse)
        return strct

    # SIGNAL + STRUCT
    def write_in_pipeout_signal_strct(self, signal:int = 0, strct=''):
        verse = pickle.dumps(strct)
        ret = self._write_in_pipeout(int_to_bytes(signal) + int_to_bytes(len(verse)))
        if ret is None:
            return None
        ret = self._write_in_pipeout(verse)
        if ret is None:
            return None
        return ret

    def read_from_pipein_signal_strct(self):
        line = self._read_from_pipein(8)
        if line is None:
            return None
        signal = int_from_bytes(line[:4])
        length = int_from_bytes(line[4:])
        #print('length:', length)
        if length == 0:
            return (signal, '')
        #print('Embedder length:', length)
        verse = self._read_from_pipein(length)
        if verse is None:
            return None
        strct = pickle.loads(verse)
        return (signal, strct)
