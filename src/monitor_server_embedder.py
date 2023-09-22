# -*- coding: utf-8 -*-
"""Server that receives images with faces and get their to handler;
    then it checks that handler handles this images, and get theirto client.
"""

import os
import sys
import imp
import time
import json
from datetime import datetime
from datetime import timedelta
from datetime import date
import requests

from netifaces import AF_INET, AF_INET6, AF_LINK, AF_PACKET, AF_BRIDGE
import netifaces as ni

from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from urllib.parse import urlparse, parse_qs

from threading import Thread, Lock, get_ident
import threading
#import shared_memory
#from named_atomic_lock import NamedAtomicLock

#from constants import SHM_PRF
#from load_config import load_config
#new_dir, old_dir = __file__[:__file__.rfind('/')], os.getcwd()
new_dir, old_dir = './' if __file__.rfind('/')==-1 else __file__[:__file__.rfind('/')], os.getcwd()
os.chdir(new_dir)
load_config    = imp.load_source(new_dir, 'load_config.py').load_config
SHM_PRF        = imp.load_source(new_dir, 'constants.py').SHM_PRF
mc             = imp.load_source(new_dir, 'monitor_config.py')
os.chdir(old_dir)
lab_titles, grph_titles, grph_names, grph_heights, grph_groups, grph_names, grph_axess, labels_zero, items_zero = \
    mc.lab_titles, mc.grph_titles, mc.grph_names, mc.grph_heights, mc.grph_groups, mc.grph_names, mc.grph_axess, mc.labels_zero, mc.items_zero
#from monitor_zmq import Monitor_Zmq


answer_lock = Lock()
threads_lock = Lock()

#saved_lock = NamedAtomicLock(SHM_PRF+'saved_lock')

now = datetime.now() # current date and time
first_time_st = now.strftime("%Y-%m-%d %H:%M:%S")

mon_zmq = None
httpd = None
thread_m_procs = None


def get_cur_labels_items_from_mon_zmq():
    global mon_zmq
    if mon_zmq is None:
        now = datetime.now() # current date and time
        first_time_st = now.strftime("%Y-%m-%d %H:%M:%S")
        return first_time_st, first_time_st, labels_zero, items_zero
    return mon_zmq.get_cur_labels_items()


def modify_contents(contents):
    insert_code_1, insert_code_2 = '', ''
    #
    i = 0
    for title in lab_titles:
        insert_code_1 += '<br />\r\n<b>%s:</b> ' % title
        for j in range(len(labels_zero[i])):
            insert_code_1 += '<span id="label_%d_%d">%.2f</span> &nbsp ' % (i, j, labels_zero[i][j])
        i += 1
    insert_code_1 += '<br />'
    #
    i = 0
    for title in grph_titles:
        insert_code_1 += '<br />\r\n<b>%s</b>' % title
        insert_code_1 += '<br />\r\n<div id="visualization_%d" width="90%%"></div>\r\n' % i
        i += 1
    #
    insert_code_2 += '    var items = ['
    for i in range(len(grph_titles)):
        insert_code_2 += '[{x: "%s", y: 0, group: "0"}], ' % (items_zero[i][0][0])
    insert_code_2 = insert_code_2[:-2] + '];\r\n'
    insert_code_2 += '    var graph2d_s = ['
    for i in range(len(grph_titles)):
        insert_code_2 += 'null, '
    insert_code_2 = insert_code_2[:-2] + '];\r\n'
    #
    for i in range(len(grph_titles)):
        insert_code_2 += '    var container_%d = document.getElementById("visualization_%d");\r\n' % (i, i)
        insert_code_2 += '    var groups_%d = new vis.DataSet();\r\n' % (i)
        for j in range(grph_groups[i]):
            insert_code_2 += '    groups_%d.add({id: "%d", content: "%s"});\r\n' % (i, j, grph_names[i][j])
        insert_code_2 += '    var options_%d = {height: "%dpx", ' % (i, grph_heights[i])
        insert_code_2 += 'dataAxis: {showMinorLabels: true, left: {title: {text: "%s"}}}, ' % (grph_axess[i])
        #
        d = timedelta(seconds=10)
        start = (datetime.strptime(items_zero[i][0][0], "%Y-%m-%d %H:%M:%S") - d).strftime("%Y-%m-%d %H:%M:%S")
        end   = (datetime.strptime(items_zero[i][0][0], "%Y-%m-%d %H:%M:%S") + d).strftime("%Y-%m-%d %H:%M:%S")
        #
        insert_code_2 += 'legend: {left:{position:"bottom-left"}}, start: "%s", end: "%s" };\r\n' % (start, end)
        insert_code_2 += '    var items_%d = [' % (i)
        for j in range(grph_groups[i]):
            insert_code_2 += '{x: "%s", y: 0, group: "%d"}, ' % (items_zero[i][j][0], j)
        insert_code_2 = insert_code_2[:-2] + '];\r\n'
        insert_code_2 += '    items[%d] = items_%d;\r\n' % (i, i)
        insert_code_2 += '    graph2d_s[%d] = new vis.Graph2d(container_%d, items_%d, groups_%d, options_%d);\r\n\r\n' % (i,i,i,i,i)
    #
    contents = contents.replace('<!-- TO INSERT CODE 1 -->', insert_code_1)
    contents = contents.replace('// TO INSERT CODE 2', insert_code_2)
    return contents


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
            answer_lock.acquire()
            try:
                self.send_response(status)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(response_str)
            except Exception:
                answer_lock.release()
                print('_write_response exception')
                i += 1
                time.sleep(0.2)
                continue
            answer_lock.release()
            break
        if i == 10:
            print('TID: %d  ERROR WRITING SOCKET!' % get_ident())
            try:
                logger.error('TID: %d  ERROR WRITING SOCKET!' % get_ident())
                logger.error('        len(response_str): %d' % len(response_str))
                #logger.error('        ' + response_str[:100])
            except Exception:
                pass

    def do_POST(self): # pylint: disable=C0103
        """
        Handles post query.
        Args:
            None.
        Returns:
            None.
        """
        pass

    def do_PUT(self): # pylint: disable=C0103
        """
        Handles put query.
        Args:
            None.
        Returns:
            None.
        """
        self.do_POST()

    def do_GET(self):
        global items_zero
        global httpd
        global thread_m_procs

        if self.path in ['/restart']:
            print('RESTART PROGRAMM')
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(''.encode())

            self.restart_working_processes()

        #
        if self.path in ['/', '/config', '/json']:
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            if self.path == '/':
                f = open('./src/graph3.html', 'rt')
                contents = f.read()
                contents = modify_contents(contents)
                self.wfile.write(contents.encode())
            if self.path == '/config':
                jsn = {"type": "config"}
                jsn["titles"] = titles

                jsn_str = json.dumps(jsn)
                self.wfile.write(jsn_str.encode())
            if self.path == '/json':
                now = datetime.now()
                ####min_tm_str, max_tm_str, cur_labels, cur_items = get_cur_labels_items_from_mon_zmq()
                #
                jsn = {"type": "full"}
                jsn["labels"] = cur_labels
                jsn["items"] = cur_items
                #d = timedelta(seconds=5)
                jsn["start"] = min_tm_str  #(datetime.strptime(cur_items[0][0][0], "%Y-%m-%d %H:%M:%S") - d).strftime("%Y-%m-%d %H:%M:%S")
                jsn["end"]   = max_tm_str  #(datetime.strptime(cur_items[0][-1][0], "%Y-%m-%d %H:%M:%S") + d).strftime("%Y-%m-%d %H:%M:%S")

                jsn_str = json.dumps(jsn)
                self.wfile.write(jsn_str.encode())

    def get_bad_answer(self, http_error):
        """
        Create answer to http error situation.
        Args:
            http_error: Number of HTTP status.
        Returns:
            Bytes array: answer.
        """
        return b'Http error %d was generated.' % http_error

    def stop_program(self):
        httpd.server_close()
        httpd.stopped = True
        ####mon_zmq.stop = True
        ####mon_zmq.mon_proc.stop = True
        ####mon_zmq.zmq_mon.serv_running = False
        print('mon_zmq.mon_proc.stop = True')
        #sys.exit()
        
    def restart_working_processes(self):
        ####mon_zmq.mon_proc.kill_all_procs()
        ####mon_zmq.mon_proc.start_all_procs()
        pass
        


###################################################################


class MonitorServer():
    def __init__(self, json_name, proc_strs_lst=[]):
        ####global mon_zmq
        CONF_DCT = load_config(json_name)
        self.CONF_DCT = CONF_DCT
        self.PORT = CONF_DCT['PORT_MON_OUT']

        print(ni.interfaces())

        ifcs = ['enp0s31f6', 'tap0']
        HOST: str = ''
        for ifc in ifcs:
            try:
                HOST: str = ni.ifaddresses(ifc)[AF_INET][0]['addr']
            except:
                continue
            break
        if HOST == '':
            print('No valid interfaces detected')
            sys.exit(1)

        self.server_address = (HOST, self.PORT)
        ####mon_zmq = Monitor_Zmq(json_name, proc_strs_lst)
        self.proc_strs_lst = proc_strs_lst
        self.start_all_procs()


    def start_all_procs(self):
        TIMEOUT_TO_START_NEXT_PROCESS = self.CONF_DCT['TIMEOUT_TO_START_NEXT_PROCESS']
        for proc_str in self.proc_strs_lst:
            os.system( proc_str + ' &' )
            #pids = get_pid(proc_str)
            #print(pids, type(pids))
            #print('%s    %s' % (proc_str, pid))
            time.sleep(TIMEOUT_TO_START_NEXT_PROCESS)
            #for proc_str_ in self.proc_strs_lst:
            #    if True:  # proc_str_.find('embed') == -1:
            #        self.last_ping_tms_dct[proc_str_] = time.time()


    def run(self):
        global httpd
        global mon_zmq
        global thread_m_procs
        ####thread_m_zmq = Thread(target = mon_zmq.run)
        ####thread_m_zmq.start()
        #self.zmq_mon.run_serv(self.callback_func)

        httpd = ThreadedHTTPServer(self.server_address, HandleRequests)
        sa = httpd.socket.getsockname()
        print("Serving HTTP on", sa[0], "port", sa[1], "...")
        thread = Thread(target = httpd.serve_forever)
        thread.start()
        print('server running on port {}'.format(httpd.server_port))

        ####thread_m_procs = Thread(target = mon_zmq.mon_proc.run)
        ####print('thread_m_procs.pre-start()')
        ####thread_m_procs.start()
        ####print('thread_m_procs.start()')
        ####thread_m_procs.join()
        ####print('thread_m_procs.join()')
        ####thread_m_zmq.join()
        ####print('thread_m_zmq.join()')
        ####mon_zmq.one_sec_thread.join()
        ####print('mon_zmq.one_sec_thread.join()')
        ####httpd.shutdown()
        ####thread.join()
        ####print('thread.join()')
        
        while True:
            time.sleep(1)

        return

################################################################################

def get_proc_strs_lst(cnf_file):
    CONF_DCT = load_config(cnf_file)

    #proc_lst = ['python3 src/controller.py %s']
    proc_lst = ['python3 src/embedder.py %s']
    #proc_lst.append('python3 src/client_out.py %s')
    proc_lst.append('python3 src/server_in.py %s')


    proc_strs_lst = [s % cnf_file for s in proc_lst]
    return proc_strs_lst


################################################################################

def main():
    if len(sys.argv) != 2:
        print("""USAGE: monitor_server_embedder.py json_name
EXAMPLE: monitor_server_embedder.py config_all.txt""")
        exit(1)

    cnf_file = sys.argv[1]
    proc_strs_lst = get_proc_strs_lst(cnf_file)
    print(proc_strs_lst)
    #time.sleep(10)
    
    r = MonitorServer(cnf_file, proc_strs_lst)
    r.run()

################################################################################

if __name__ == "__main__":
    main()
    quit()

################################################################################
