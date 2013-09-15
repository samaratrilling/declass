from fnmatch import fnmatch
import os
import shutil
import re
import sys
import subprocess
import pdb 
from numpy.random import rand
from functools import partial

"""
Contains a collection of function that clean, decode and move files around.
"""


def get_paths(dir_name, file_type="*", relative=False):
    """
    Crawls subdirectories and returns list of paths to files that match the file_type.

    Parameters
    ----------
    dir_name : String
        Path to the directory that will be crawled
    file_type : String
        String to filter files with.  E.g. '*.txt'.  Note that the filenames
        will be converted to lowercase before this comparison.
    relative : Boolean
        If True, get paths relative to dir_name
        If False, get absolute paths
    """
    path_list = []
    for path, subdirs, files in os.walk(dir_name, followlinks=True):
        for name in files:
            if fnmatch(name.lower(), file_type):
                if relative:
                    path = path.replace(dir_name, "")
                    if path.startswith('/'):
                        path = path[1:]
                path_list.append(os.path.join(path,name))    

    return path_list


def path_to_name(path, name_level=1):
    """
    Takes one path and returns a name.

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


def get_paths_iter(base_path, file_type="*", relative=False):
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
    """
    path_list = []
    for path, subdirs, files in os.walk(base_path, followlinks=True):
        for name in files:
            if fnmatch(name.lower(), file_type):
                if relative:
                    path = path.replace(base_path, "")
                    if path.startswith('/'):
                        path = path[1:]
                yield os.path.join(path, name)


def paths_to_files(paths, mode='r'):
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
