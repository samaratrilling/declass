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


def get_one_file_name(path):
    """
    Takes one path and returns the filename, excluding the extension.

    Parameters
    ----------
    path : String

    Returns
    -------
    filename : String
    """
    head, tail = os.path.split(path)
    filename, ext = os.path.splitext(tail)

    return filename




