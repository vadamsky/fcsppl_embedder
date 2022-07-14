# -*- coding: utf-8 -*-
"""Functions to work with images byte arrays.
"""

import io
import numpy as np # type: ignore

def int_to_bytes(x_32: int) -> bytes:
    """
    Pack int32 in bytes[4].
    Args:
        x_32: int32.
    Returns:
        Bytes[4].
    """
    return x_32.to_bytes(4, 'big')

def int_from_bytes(xbytes: bytes) -> int:
    """
    Unpack int32 from bytes[4].
    Args:
        xbytes: bytes[4].
    Returns:
        int32.
    """
    return int.from_bytes(xbytes, 'big')

def serialize_numpy_array(np_arr):
    """
    Serialize numpy array to bytes.
    Args:
        np_arr: Numpy array.
    Returns:
        Serialized array in bytes.
    """
    memfile = io.BytesIO()
    np.save(memfile, np_arr)
    memfile.seek(0)
    serialized = memfile.read()
    return serialized

def deserialize_numpy_array(serialized):
    """
    Deserialize numpy array from bytes.
    Args:
        serialized: bytes.
    Returns:
        Numpy array.
    """
    memfile = io.BytesIO()
    memfile.write(serialized)
    memfile.seek(0)
    np_arr = np.load(memfile)
    return np_arr
