### util.py
#
# Author: Allen Wang
###
import errno
import os
import shutil
import numpy as np

def mkdirP(path):
    """
    Create a directory and don't error if the path already exists.

    If it exists, do nothing.
    """
    assert path is not None

    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def rmP(path):
    """
    Remove a file and don't error if the file doesn't exist
    """
    assert path is not None

    try:
        os.remove(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.ENOENT:
            pass
        else:
            raise

def rmdirP(path):
    """
    Remove a folder, ignore error
    """
    shutil.rmtree(path, True)

def mvP(src, dest):
    """
    Move a folder recursively
    """
    os.rename(src, dest)

def iterImgDirectory(directory):
    """
    Create a list of all the image directories in a folder
    """
    assert directory is not None

    paths = []

    for subdir, dirs, files in os.walk(directory):
        for f in files:
            paths.append(os.path.join(directory, f))

    return paths

def chmodR(path):
    """
    chmod -R for a path
    """
    os.chmod(path, 0o777)
    for root, dirs, files in os.walk(path):
        for d in dirs:
            os.chmod(os.path.join(root, d), 0o777)

def chmodP(path):
    """
    chmod for a single file
    """
    os.chmod(path, 0o777)

def divideArray(in_arr, num_processes):
    return np.array_split(in_arr, num_processes)
