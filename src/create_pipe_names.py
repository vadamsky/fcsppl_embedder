#!/usr/bin/env python3

import sys
import os
import os.path
import pathlib


def main(detectors_num, embedders_num, nofaces_num, agegender_num):
    lst = [(detectors_num, 'detector'),
           (embedders_num, 'embedder'),
           (nofaces_num, 'nofaces'),
           (agegender_num, 'agegender'),
          ][1:2]

    for (num, pref) in lst:
        pipe_out_name_pref = './pipes/pipe_main_%s_' % pref
        pipe_out_name_pref2 = './pipes/pipe_%s_main_' % pref

        pipe_out_names = [pipe_out_name_pref + '%d' % i for i in range(num)]
        pipe_out_names += [pipe_out_name_pref2 + '%d' % i for i in range(num)]
        for pipe_out_name in pipe_out_names:
            if not os.path.exists(pipe_out_name):
                os.mkfifo(pipe_out_name)  

    for (num, pref) in lst:
        pipe_out_name_pref = '/cache/pipes/pipe_main_%s_' % pref
        pipe_out_name_pref2 = '/cache/pipes/pipe_%s_main_' % pref

        pipe_out_names = [pipe_out_name_pref + '%d' % i for i in range(num)]
        pipe_out_names += [pipe_out_name_pref2 + '%d' % i for i in range(num)]
        for pipe_out_name in pipe_out_names:
            if not os.path.exists(pipe_out_name):
                os.mkfifo(pipe_out_name)  

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("""USAGE: create_pipe_names.py detectors_num\nEXAMPLE 1: create_pipe_names.py 8""")
        exit(1)
    embedders_num = 4
    nofaces_num = 1
    agegender_num = 1
    main(int(sys.argv[1]), embedders_num, nofaces_num, agegender_num)
