from fnmatch import fnmatch
import os
import shutil
import re
import sys
import subprocess
#import pdb 
from numpy.random import rand
from functools import partial

"""
Contains a collection of function that clean, decode and move files around.
"""

def get_paths(base_path, file_type="*", relative=False, get_iter=False):
    """
    Crawls subdirectories and returns an iterator over paths to files that
    match the file_type.

    Parameters
    ----------
    base_path : String
        Path to the directory that will be crawled
    file_type : String
        String to filter files with.  E.g. '*.txt'.  Note that the filenames
        will be converted to lowercase before this comparison.
    relative : Boolean
        If True, get paths relative to base_path
        If False, get absolute paths
    get_iter : Boolean
        If True, return an iterator over paths rather than a list.
    """
    path_iter = _get_paths_iter(
        base_path, file_type=file_type, relative=relative)

    if get_iter:
        return path_iter
    else:
        return [path for path in path_iter]


def _get_paths_iter(base_path, file_type="*", relative=False):
    path_list = []
    for path, subdirs, files in os.walk(base_path, followlinks=True):
        for name in files:
            if fnmatch(name.lower(), file_type):
                if relative:
                    path = path.replace(base_path, "")
                    if path.startswith('/'):
                        path = path[1:]
                yield os.path.join(path, name)


def path_to_name(path):
    """
    Takes one path and returns the filename, excluding the extension.
    """
    head, tail = os.path.split(path)
    filename, ext = os.path.splitext(tail)

    return filename


def path_to_newname(path, name_level=1):
    """
    Takes one path and returns a new name, combining the directory structure
    with the filename.

    Parameters
    ----------
    path : String
    name_level : Integer
        Form the name using items this far back in the path.  E.g. if
        path = mydata/1234/3.txt and name_level == 2, then name = 1234_3

    Returns
    -------
    name : String
    """
    name_plus_ext = path.split('/')[-name_level:]
    name, ext = os.path.splitext('_'.join(name_plus_ext))

    return name


def paths_to_files_iter(paths, mode='r'):
    """
    Returns an iterator that opens files in path_list.

    Parameters
    ----------
    paths : Iterable over paths
        Each path is a string
    mode : String
        mode to open the files in

    Returns
    -------
    file_iter : Iterable
        file_iter.next() gives the next open file.
    """
    for path in paths:
        yield open(path.strip(), mode=mode)


