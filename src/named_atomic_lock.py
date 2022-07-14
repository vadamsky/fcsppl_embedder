import os
import sys
import time
try:
    from shared_memory import SharedMemory2
except Exception:
    from src.shared_memory import SharedMemory2


class NamedAtomicLock():
    def __init__(self, name):
        self.name = name
        self.isHeld = False
        self.shm = None
        try:
            self.shm = SharedMemory2(name='namedatomiclock_'+name, create=True, size=8)
        except Exception:
            # if exist now
            self.shm = SharedMemory2(name='namedatomiclock_'+name, create=False, size=8)
        # Fill SHM
        x_64 = 0
        self.shm.buf[:8] = x_64.to_bytes(8, 'big')


    def acquire(self, block=True, timeout=None):
        isHeld = int.from_bytes(self.shm.buf[0:8], 'big')
        if not block:
            return not isHeld
        start = time.time()
        if timeout is None:
            while bool(isHeld):
                time.sleep(0.02)
                isHeld = int.from_bytes(self.shm.buf[0:8], 'big')
        else:
            while (time.time() - start < timeout) and bool(isHeld):
                time.sleep(0.02)
                isHeld = int.from_bytes(self.shm.buf[0:8], 'big')
        x_64 = 1
        self.shm.buf[:8] = x_64.to_bytes(8, 'big')
        self.isHeld = True
        return isHeld == 0

    def release(self):
        isHeld = int.from_bytes(self.shm.buf[0:8], 'big')
        x_64 = 0
        self.shm.buf[:8] = x_64.to_bytes(8, 'big')
        self.isHeld = False
        return isHeld==1

    def locked(self):
        isHeld = int.from_bytes(self.shm.buf[0:8], 'big')
        return isHeld
