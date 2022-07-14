# -*- coding: utf-8 -*-
"""Class for sending messages 2 monitor and recv they from monitor.
"""

import zmq
import time
import random
import pickle
from collections import defaultdict
from threading import Thread, Lock

#from constants import MONITOR_ADDR
#MON_ADDR = MONITOR_ADDR
#MON_ADDR = "tcp://127.0.0.1:40002"
MON_HOST = "tcp://127.0.0.1"
DET_HOST = "tcp://127.0.0.1"
EMB_HOST = "tcp://127.0.0.1"
SRV_HOST = "tcp://127.0.0.1"


class ZmqWrapper:
    def __init__(self, identifier, addr, tp='client', resp_type='without_response'):
        self.addr = addr
        self.serv_running = True
        self.identifier = identifier
        self.context = zmq.Context()
        self.resp_type = resp_type
        self.connected = False
        self.send_counter = 0
        self.send_threads = []  # {}
        #self.send_threads_last_id = 0
        self.send_threads_lock = Lock()
        if tp == 'client':
            self.connect_func(addr)
        if tp == 'server':
            self.socket = self.context.socket(zmq.ROUTER)
            self.bind_func(addr)

    def __del__(self):
        self.context.destroy()

    def bind_func(self, addr):
        try:
            addr_ = addr[:addr.rfind('/')+1] + '*' + addr[addr.rfind(':'):]
            self.socket = self.context.socket(zmq.ROUTER)
            self.socket.bind(addr)
            self.connected = True
            print('binded')
            self.poll = zmq.Poller()
            self.poll.register(self.socket, zmq.POLLIN)
        except Exception as e:
            print('Binding error:', e)

    def connect_func(self, addr):
        try:
            self.socket = self.context.socket(zmq.DEALER)
            self.socket.setsockopt_string(zmq.IDENTITY, self.identifier)
            self.socket.connect(addr)
            self.connected = True
            print('connected')
        except Exception as e:
            print('Connection error:', e)

    #def send_func(self, thr_id, data, msg_type=None, ):
    def send_func(self, data, msg_type=None, ):
        #try:
        self.socket.send(pickle.dumps((self.identifier, time.time(), msg_type, data)))#, zmq.NOBLOCK)
        #except Exception as e:
        #    print('Error ZmqWrapper.send: self.socket.send', e)
        """
        try:
            if self.resp_type != 'without_response':
                ####self.socket.RCVTIMEO = 1000 # in milliseconds
                pass  # ret = self.socket.recv()#zmq.NOBLOCK)
        except Exception as e:
            print('ZmqWrapper.send:  recv error', e)
        #
        #self.send_threads_can_join[thr_id] = True
        """
        return None


    def send(self, data, msg_type=None):
        self.send_func(data, msg_type)
        """
        #tm = time.time()
        self.send_threads_lock.acquire()
        try:
            #send_thread = Thread(target=self.send_func, args = (self.send_threads_last_id, data, msg_type, ))
            send_thread = Thread(target=self.send_func, args = (data, msg_type, ))
            send_thread.start()
            #self.send_threads[self.send_threads_last_id] = send_thread
            #self.send_threads_last_id += 1
            self.send_threads.append(send_thread)
        except Exception as e:
            print('ZmqWrapper send error:', e)
            self.send_threads_lock.release()
        else:
            self.send_threads_lock.release()
        #time_sending = time.time() - tm
        #tm = time.time()

        joined = 0
        while True:
            not_joined = True
            #keys = list(self.send_threads.keys())
            #for thread_id in keys:
            #for send_thread in self.send_threads:
            #for i, send_thread in enumerate(self.send_threads):
            for i in range(len(self.send_threads)):
                #send_thread = self.send_threads[thread_id]
                send_thread = self.send_threads[i]
                try:
                    if not send_thread.is_alive():
                        if True:  # self.send_threads_can_join[thread_id]:
                            not_joined = False
                            send_thread.join()
                            self.send_threads_lock.acquire()
                            self.send_threads = self.send_threads[:i] + self.send_threads[i+1:]
                            #self.send_threads.pop(thread_id, None)
                            #self.send_threads_can_join.pop(thread_id, None)
                            self.send_threads_lock.release()
                            del send_thread
                            joined += 1
                            break
                except Exception as e:
                    print('ZmqWrapper join error:', e)
                    self.send_threads_lock.release()
            if not_joined:
                break
        #time_joining = time.time() - tm
        """

        #print('------------ Send times:', time_sending, time_joining, joined)
        return None


    def run_serv(self, callback_func):
        while self.serv_running:
            identity, rcv = None, None
            sockets = dict(self.poll.poll(1000))
            if sockets:
                identity = self.socket.recv()
                rcv = self.socket.recv()
                #print('MSG:', rcv)

            if rcv and identity:
                identifier, tm, msg_type, data = pickle.loads(rcv)
                if self.resp_type != 'without_response':
                    #self.socket.send(identity, zmq.SNDMORE)
                    pass  # self.socket.send(b"ok")
                #print('ZmqWrapper run_serv:', data)
                callback_func((identifier, tm, msg_type, data))
            # #except Exception as e:
            # #    print(e)
        print('ZmqWrapper.run_serv pre-stopped')
        self.socket.close()
        self.context.term()
        print('ZmqWrapper.run_serv stopped')



"""
class ZmqWrapper:
    def __init__(self, identifier, addr, tp='client', resp_type='without_response'):
        #addr = 'tcp://127.0.0.1:%d' % port
        self.addr = addr
        self.serv_running = True
        self.identifier = identifier
        self.context = zmq.Context()
        self.resp_type = resp_type
        self.connected = False
        self.send_counter = 0
        self.send_threads = []
        self.send_threads_lock = Lock()
        if tp == 'client':
            self.socket = self.context.socket(zmq.REQ)
            self.connect_func(addr)
            #print('ZmqWrapper client __init__ ok')
            ####self.connect_thread = Thread(target=self.connect_func, args = (addr, ))
            ####self.connect_thread.start()
        if tp == 'server':
            self.socket = self.context.socket(zmq.REP)
            self.connect_thread = Thread(target=self.bind_func, args = (addr, ))
            self.connect_thread.start()
            #print('ZmqWrapper server __init__ ok')

    def __del__(self):
        self.context.destroy()

    def bind_func(self, addr):
        try:
            self.socket.bind(addr)
            self.connected = True
            print('binded')
        except Exception:
            pass

    def connect_func(self, addr):
        try:
            self.socket.connect(addr)
            self.connected = True
            print('connected')
        except Exception:
            pass

    def send_func(self, data, msg_type=None):
        try:
            self.socket.send(pickle.dumps((self.identifier, time.time(), msg_type, data)))#, zmq.NOBLOCK)
        except Exception as e:
            print('Error ZmqWrapper.send: self.socket.send', e)
            #return 'Error ZmqWrapper.send: self.socket.send'
        #ret = b''
        try:
            if self.resp_type != 'without_response':
                self.socket.RCVTIMEO = 1000 # in milliseconds
                ret = self.socket.recv()#zmq.NOBLOCK)
        except Exception as e:
            print('ZmqWrapper.send:  recv error', e)
            ####self.connected = False
            #return 'Error ZmqWrapper.send: self.socket.recv'
        return  # ret


    def send(self, data, msg_type=None):
        ####if self.connected:
        ####    if self.connect_thread:
        ####        self.connect_thread.join()
        ####        self.connect_thread = None
        ####    self.send_counter += 1
        ####else:
        ####    self.connect_thread = Thread(target=self.connect_func, args = (self.addr, ))
        ####    self.connect_thread.start()

        send_thread = Thread(target=self.send_func, args = (data, msg_type, ))
        self.send_threads_lock.acquire()
        self.send_threads.append(send_thread)
        self.send_threads_lock.release()
        send_thread.start()
        #print('ZmqWrapper send:', data)

        while True:
            not_joined = True
            for i, send_thread in enumerate(self.send_threads):
                if not send_thread.is_alive():
                    not_joined = False
                    send_thread.join()
                    self.send_threads_lock.acquire()
                    self.send_threads = self.send_threads[:i] + self.send_threads[i+1:]
                    self.send_threads_lock.release()
                    del send_thread
            if not_joined:
                break

        return None


    def run_serv(self, callback_func):
        while self.serv_running:
            #try:
            rcv = None
            #try:
            rcv = self.socket.recv()#zmq.NOBLOCK)
            #except Exception as e:
            #    print('.   ', e)
            if rcv:
                identifier, tm, msg_type, data = pickle.loads(rcv)
                if self.resp_type != 'without_response':
                    self.socket.send(b'ok')
                #print('ZmqWrapper run_serv:', data)
                callback_func((identifier, tm, msg_type, data))
            #except Exception as e:
            #    print(e)
        self.socket.close()
        self.context.term()
"""
