from fnmatch import fnmatch
import os
import shutil
import re
import sys
import subprocess
import pdb 
from numpy.random import rand
from functools import partial
import pymysql

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


class DBCONNECT(object):
    """
    Connects to a MySQL DB and executes basic queiries.

    Example
    -------
    dbCon = DBCONNECT(host_name='mysql.csail.mit.edu', db_name='declassification', user_name='declass', pwd='declass')
    table_name = 'declassification'
    doc_id = 242518
    fields = 'body, title'
    doc = dbCon.get_row_by_id(row_id=doc_id, table_name=table_name, fields=fields)
    doc_list = dbCon.get_rows_by_idlist(id_list=[242518, 317509], table_name=table_name, fields=fields)

    Notes
    -----
    TODO : figure out why pymysql cursor does not execute 'where in list' type statements

    """
    def __init__(self, host_name, db_name, user_name, pwd):
        """
        Initializes the class and pymysql cursor object.

        Parameters
        ----------
        host_name : string
        db_name : string
        user_name : string
        pwd : string
        """
        self.conn = pymysql.connect(host=host_name, user=user_name, passwd=pwd, db=db_name)
        self.cursor = self.conn.cursor()
        self.conn.autocommit(1)

    def get_row_by_id(self, row_id, table_name, fields='*'):
        """
        Parameters
        ----------
        row_id : string or int
        table_name : string
        fields : string
            format = 'field1, field2, ...'; default is all fields
        
        Notes
        -----
        assumes table has an 'id' entry
        """
        sql = 'select %s from Document where id = %s'%(fields, row_id)
        self.cursor.execute(sql)
        output = self.cursor.fetchall()
        output = list(output[0])
        return output

    def get_rows_by_idlist(self, id_list, table_name, fields='*'):
        """
        Parameters
        ----------
        id_list : list of strings or ints
        table_name : string
        fields : string
            format = 'field1, field2, ...'; default is all fields
        
        Notes
        -----
        assumes table has an 'id' entry
        TODO: remove after sort out pymysql 'where in ' bug

        """
        output_list = []
        [output_list.append(self.get_row_by_id(row_id=row_id, table_name=table_name,
            fields=fields)) for row_id in id_list]
        return output_list
        
    def close(self):
        """
        Closes the mysql connection.

        Notes
        -----
        Not strictly necessary, but good practice to close session after use.
        """
        self.conn.close()
              
    

if __name__ == '__main__':
    
    dbCon = DBCONNECT(host_name='mysql.csail.mit.edu', db_name='declassification', user_name='declass', pwd='declass')
    table_name = 'declassification'
    doc_id = 242518
    fields = 'body, title'
    doc = dbCon.get_row_by_id(row_id=doc_id, table_name=table_name, fields=fields)
    print doc
    doc_list = dbCon.get_rows_by_idlist(id_list=[242518, 317509], table_name=table_name, fields=fields)
    print doc_list
    dbCon.close()


