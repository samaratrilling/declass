from fnmatch import fnmatch
import os
import shutil
import re
import sys
import subprocess
import pdb 
from numpy.random import rand
from functools import partial

from jrl_utils.src.parallel_easy import imap_easy, map_easy

from akin.src import stripper

"""
Contains a collection of function that clean, decode and move files around.
"""


def get_paths(dirName, fileType="*", relative=False):
    """
    Crawls subdirectories and returns list of paths to files that match the fileType.

    Parameters
    ----------
    dirName : String
        Path to the directory that will be crawled
    fileType : String
        String to filter files with.  E.g. '*.txt'.  Note that the filenames
        will be converted to lowercase before this comparison.
    relative : Boolean
        If True, get paths relative to dirName
        If False, get absolute paths
    """
    path_list = []
    for path, subdirs, files in os.walk(dirName, followlinks=True):
        for name in files:
            if fnmatch(name.lower(), fileType):
                if relative:
                    path = path.replace(dirName, "")
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


def paths_to_files(path_list, mode='r'):
    """
    Returns an iterator that opens files in path_list.

    Parameters
    ----------
    path_list : List of Strings
    mode : String
        mode to open the files in

    Returns
    -------
    file_iter : Iterable
        file_iter.next() gives the next open file.
    """
    for path in path_list:
        yield open(path.strip(), mode=mode)
