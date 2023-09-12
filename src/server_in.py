#!/usr/bin/env python3
# conda create -n py39 python=3.9
# conda deactivate
# conda activate py39
# pip install Pillow numpy opencv-python kombu requests netifaces readchar logger torch tqdm

import os
import base64
import pathlib
import time
import sys
import json
import pickle
import math
import numpy as np # type: ignore
import cv2 # type: ignore

from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread, Semaphore, get_ident, Lock
from io import BytesIO
from PIL import Image # type: ignore

import socket
from kombu import Connection, Exchange, Queue, Consumer

from logger import log_in_file
from base_class import BaseClass
from load_config import load_config
from runner_embed import RunnerEmbed
from constants import RABBIT_IN_ADDR, RABBIT_IN_QNAME, RABBIT_IN_EXCH, RABBIT_IN_RKEY


server_in = None


class ServerIn(BaseClass):
    def __init__(self, cnf_name):
        self.req_id = 1
        self.runner_embed = RunnerEmbed(cnf_name)
        CONF_DCT = load_config(cnf_name)
        self.TEST_QUEUE = CONF_DCT['TEST_QUEUE']

    def handle_rabbit_message(self, body, message=None):
        verse = body
        log_in_file('server_in.csv', f'handle_rabbit_message: start')

        (req_id, client_id, img_id, faces_imgs, frames, faces_keypoints, status, dict_stat) = pickle.loads(bytes(verse))
        log_in_file('server_in.csv', f'handle_rabbit_message: req_id={req_id}, client_id={client_id}')
        print('QueueIn:', img_id)
        #
        try:
            dict_stat['emb_load_from_saved_tm'] = time.time()
            data = (req_id, client_id, img_id, faces_imgs, frames, faces_keypoints, status, dict_stat)
            identifier, tm, msg_type = 'server', time.time(), 'send_detected'
            dt = (identifier, tm, msg_type, data)
            self.runner_embed.zmq_callback_func(dt)
        except Exception as e:
            log_in_file('server_in.csv', f'handle_rabbit_message: Error: req_id={req_id}, client_id={client_id}  e={e}')
        #
        if not (message is None):
            message.ack()

    def run(self):
        self.runner_embed.run()
        pass


def get_rabbit_messages(server_in):
    rabbit_url = f"amqp://guest2:guest2@{server_in.TEST_QUEUE}:5672/"
    conn = Connection(rabbit_url)
    exchange = Exchange("test_exchange", type="direct")
    queue = Queue(name="test_queue", exchange=exchange, routing_key="BOB")
    #consumer = Consumer(conn, queues=queue, callbacks=[server_in.handle_rabbit_message], accept=["text/plain"])
    #consumer.drain_events(timeout=5)
    #consumer.consume()

    def establish_connection(conn):
        revived_connection = conn.clone()
        revived_connection.ensure_connection(max_retries=3)
        channel = revived_connection.channel()
        consumer.revive(channel)
        consumer.consume()
        return revived_connection

    def consume():
        new_conn = establish_connection(conn)
        while True:
            try:
                new_conn.drain_events(timeout=2)
            except socket.timeout:
                new_conn.heartbeat_check()

    while True:
        #try:
        #    consume()
        #except conn.connection_errors:
        #    print("connection revived")
        try:
            with Consumer(conn, queues=queue, callbacks=[server_in.handle_rabbit_message], accept=["text/plain"]):
                while True:
                    try:
                        conn.drain_events(timeout=5)
                    except socket.timeout:
                        pass
        except Exception as e:
            log_in_file('server_in.csv', f'consumers: Error  e={e}')
            #del consumer


'''
class ThreadedHTTPServer(HTTPServer):
    def process_request(self, request, client_address):
        thread = Thread(target=self.__new_request, args=(self.RequestHandlerClass, request, client_address, self))
        thread.start()
    def __new_request(self, handlerClass, request, address, server):
        handlerClass(request, address, server)
        self.shutdown_request(request)

class HandleRequests(BaseHTTPRequestHandler):
    """
    Child class of BaseHTTPRequestHandler that handles requests.
    """
    def __init__(self, *args):
        BaseHTTPRequestHandler.__init__(self, *args)

    def _set_headers(self):
        """
        Send headers to answer.
        Args:
            None.
        Returns:
            None.
        """
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()

    def _write_response(self, status: int, response_str: bytes):
        """
        Send headers to answer.
        Args:
            None.
        Returns:
            None.
        """
        i = 0
        for _ in range(10):
            #answer_lock.acquire()
            try:
                self.send_response(status)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(response_str)
            except Exception as e:
                #answer_lock.release()
                #logger.error('_write_response exception:', Exception)
                print(f'Error in _write_response: {e}')
                i += 1
                time.sleep(0.2)
                continue
            #answer_lock.release()
            break
        if i == 10:
            logger.error('TID: %d  ERROR WRITING SOCKET!' % get_ident())
            logger.error('        len(response_str): %d' % len(response_str))
            #logger.error('        ' + response_str[:100])

    def do_POST(self): # pylint: disable=C0103
        """
        Handles post query.
        Args:
            None.
        Returns:
            None.
        """
        global server_in
        #
        if self.path != '/for_emb':
            return
        content_len = int(self.headers.get('Content-Length'))
        try:
            post_body = self.rfile.read(content_len)
            print(f'do_POST: content_len={content_len}')
            server_in.handle_rabbit_message(post_body)
        except Exception as e:
            err_status = 500
            print(f'Error in do_POST: {e}')
            ret = self.get_bad_answer(err_status)
            self._write_response(err_status, ret)
            return
        #
        ret = b'{"ret": "OK"}'
        self._write_response(200, ret)

    def get_bad_answer(self, http_error):
        """
        Create answer to http error situation.
        Args:
            http_error: Number of HTTP status.
        Returns:
            Bytes array: answer.
        """
        return b'Http error %d was generated.' % http_error


HOST: str = '95.216.44.199'
PORT: int = 8002
server_address = (HOST, PORT)

httpd = ThreadedHTTPServer(server_address, HandleRequests)

sa = httpd.socket.getsockname()
print("Serving HTTP on", sa[0], "port", sa[1], "...")
server_thr = Thread(target=httpd.serve_forever, args=() )
server_thr.start()
print('server running on port {}'.format(httpd.server_port))
'''


def main():
    global server_in

    if len(sys.argv) != 2:
        print("""USAGE: server_in.py cnf_file_name
EXAMPLE: server_in.py config_all.txt""")
        exit(1)

    server_in = ServerIn(sys.argv[1])
    rabbit_thr = Thread(target=get_rabbit_messages, args=(server_in, ))
    rabbit_thr.start()
    server_in.run()


################################################################################

if __name__ == "__main__":
    main()

################################################################################
