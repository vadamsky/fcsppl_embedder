# Bad method:
# for i in {1..48}; do pkill python3; python3 start.py ; sleep 1800; done

import os
import sys
import time


################################################################################

def main():
    if len(sys.argv) != 2:
        print("""USAGE: start_embedder.py config_file
EXAMPLE: start_embedder.py config_all.txt""")
        exit(1)

    os.system(r'python3 src/monitor_server_embedder.py %s' % (sys.argv[1]))
    quit()

################################################################################

if __name__ == "__main__":
    main()

################################################################################
