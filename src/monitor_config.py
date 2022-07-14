import os
import sys
import json
import time
from datetime import datetime
from datetime import timedelta
from datetime import date


TIME_TO_SHOW = 120

lab_titles = ['Root and home loading']
#grph_titles = ['Graph of root and home loading', 'Graph of queues lengths',
grph_titles = ['Graph of queues lengths',
               'Graph of in/out loading', 'Graph of all-detectors loading',  # 'Graph of all-detectors and embedder loading',
               'Graph of statistical times']  # 'Graph of all-detectors and embedder time per img/face']
#grph_axess = ['Disk loading, percents', 'Queue length',
grph_axess = ['Queue length',
              'Loading, images', 'Loading, images/faces',
              'Time, seconds']
#grph_heights = [250, 300, 300, 250, 250]
grph_heights = [300, 300, 250, 500]
#grph_groups = [2, 5, 3, 1, 1]
grph_groups = [4, 3, 1, 10]
#grph_names = [['root', 'home'], ['In queue', 'Detected queue', 'Out queue', 'In-DetRunner queue', 'Bad-fnames len'],
grph_names = [['In queue', 'Detected queue', 'Out queue', 'In-DetRunner queue'],
              ['Sync', 'Batch in', 'Batch out'], ['All detectors'],  # ['All detectors', 'Embedder'],
              ['Read socket', 'Parse', 'Srv 2 DetRunner', 'In-DetRunner waiting', 'DetRunner 2 Detector',
               'Detect', 'Detector 2 EmbRunner', 'Embed+', 'EmbRunner 2 Srv', 'Response forming']]  # ['All detectors', 'Embedder']]

now = datetime.now() # current date and time
fts = now.strftime("%Y-%m-%d %H:%M:%S")
labels_zero = [[0, 0]]
items_zero = []
for grph_count in grph_groups: 
    items_add = []
    for j in range(grph_count):
        items_add.append((fts, 0, j))  # time, value, group
    items_zero.append(items_add)
