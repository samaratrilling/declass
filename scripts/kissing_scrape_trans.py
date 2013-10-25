from BeautifulSoup import BeautifulSoup
import urllib2
import re
import simplejson as json
import os
import wget
import yaml
import pymysql
import MySQLdb

from functools import partial
from urllib import ContentTooShortError
from pymysql import ProgrammingError
from time import localtime, strftime

from parallel_easy.base import imap_easy, map_easy
from declass.utils.filefilter import get_paths


####yaml file containing login data
login_file = os.getenv("HOME") + '/.declass_db'
col_to_json_dict = {'subject': 'subject', 'date': 'docDate', 'year': 'docDate', 'doc_id': 'pdfLink'}

def get_data(master_url, master_load_path=None, master_save_path=None,
        MYDATA_dir=None, n_jobs=7, chunksize=100):
    """
    Fetches Kissinger data.

    Parameters
    ----------
    master_url : str
        url for kissinger data json
    master_load_path : str
        if provided will load the json from disk
    master_save_path : str
        if provided will save json to disk
    MYDATA_dir : str
        base dir for kissnger data (should have 'raw' and 'processed' as sub dirs)
    n_jobs : int
        number of jobs for multiprocessing 
    chunksize : int
        chunksize for multiplrocessing
    """
    dirs = _dirs(MYDATA_dir)
    master_json = get_master_json(master_url, master_load_path, master_save_path)
    paths_iter = parse_file_urls(master_json['Results'])
    #change dir since wget doesn's seem to have a outdir option
    os.chdir(dirs['PROCESSED'])
    download_files(paths_iter, n_jobs, chunksize)
    pdftotext_all()


def push_data(master_url, master_load_path=None, master_save_path=None,
        MYDATA_dir=None, limit=None):
    """
    Pushes data to Kisisnger table in db. 
    """
    dirs = _dirs(MYDATA_dir)
    master_json = get_master_json(master_url, master_load_path, master_save_path)
    results_json = master_json['Results']
    results_json = results_json[:limit]
    cursor = connect_db()
    update_table(results_json, cursor)


def update_column(master_url, col_name, master_load_path=None, master_save_path=None,
        table_name='Kissinger', MYDATA_dir=None, 
        fails_file='/tmp/Kissinger_fails.txt', limit=None):
    master_json = get_master_json(master_url, master_load_path, master_save_path)
    results_json = master_json['Results']
    results_json = results_json[:limit]
    cursor = connect_db()
    for i, result in enumerate(results_json):
        _update(col_name, result, cursor, table_name, MYDATA_dir,
                fails_file)
        if i%100==0: 
            print 'done with %s documents'%i


def _update(col_name, doc_data, db_cursor, table_name, MYDATA_dir,
        fails_file):
    id_field = col_to_json_dict['doc_id']
    id_data = doc_data[id_field]
    doc_id = _parse_doc_id(id_data)
    value = parse_data(doc_data, col_name) 
    sql = 'UPDATE %s SET %s = "%s" where doc_id = "%s";'%(
            table_name, col_name, value, doc_id)
    ff = open(fails_file, 'w')
    try:
        db_cursor.execute(sql)
    except ProgrammingError:
        ff.writeline(doc_id)
        print doc_id
    ff.close()

def parse_data(doc_data, col_name):
    field_name = col_to_json_dict[col_name]
    data = doc_data[field_name]
    parse_func = globals()['_parse_%s'%col_name]
    return parse_func(data)  

def _parse_date(data):
    s = re.search(r'\((\d*)\)', data)
    try:
        sec_string = s.group(1)
    except AttributeError:
        return None
    sec_epoch = int(sec_string[:-3])
    local = localtime(sec_epoch)
    return strftime('%Y-%m-%d', local)

def _parse_doc_id(data):
    doc = data.split('\\')[2]
    doc_id = doc.split('.')[0]
    return doc_id

def _parse_time(subject):
    return _parse_names_time(subject)['time']

def _parse_names(subject):
    return _parse_names_time(subject)['names']

def _parse_names_time(subject):
    s = re.search(
            r'(WITH|with) {,1}(.*) (AT|at) ([\d:]* [APap]\.[Mm]\.|)', 
            subject)
    try:
        names = s.group(2)
        time = s.group(4)
    except AttributeError:
        s = re.search(r'(WITH|with) ([A-Z\s\.].*)$', subject)
        try:
            names = s.group(2)
            time = None
        except AttributeError:
            names = time = None
            print subject
    return {'names': names, 'time': time}

def _parse_year(data):
    s = re.search(r'\((\d*)\)', data)
    try:
        sec_string = s.group(1)
    except AttributeError:
        return None
    sec_epoch = int(sec_string[:-3])
    return localtime(sec_epoch).tm_year

def _parse_subject(data):
    return data



def get_master_json(master_url, master_load_path=None, master_save_path=None):
    """
    Grab the master json which contains all the file info either from the site or
    from disk.
    """
    if master_load_path:
        with open(master_load_path) as f:
            master_json = json.loads(f.read())
    else:        
        page = urllib2.urlopen(master_url)
        page_string = page.read()
        #convert to workable json object after cleaning
        page_string = _clean_string(page_string)
        master_json = _convert_to_json(page_string)
    if master_save_path:
        with open(master_save_path, 'w') as f:
            json.dump(master_json, f)
    return master_json


def _convert_to_json(string):
    json_obj = json.loads(string)
    return json_obj


def _clean_string(string):
    """
    When coming from the site search API the returned data is "almost" json;
    some cleaning is needed.
    """
    string = re.sub(r'(new {,1}Date\(-{,1}\d*\))', r'"\g<0>"', string)
    return string


def parse_file_urls(results_json, master_url=('http://foia.state.gov/searchapp/'
            'DOCUMENTS/kissinger/')):
    """
    Parses the results_json object returned from the master_url and returns the all
    file urls. 
    """
    for doc_data in results_json:
        url_string = doc_data['pdfLink']
        doc = url_string.split('\\')[2]
        doc_id = doc.split('.')[0]
        doc_path = os.path.join(master_url, doc)
        yield doc_path


def download_files(paths, n_jobs=7, chunksize=100):
    """
    Wgets tjhe files, but uses the parallel_easy imap function to make things 
    go a lot faster. See documentation therein.
    """
    imap_easy(_downld_func, paths, n_jobs, chunksize)


def _downld_func(path):
    try:
        wget.download(path, bar=None)
    except IOError, ContentTooShortError:
        sleep(2)
        try:
            wget.download(path, bar=None)
        except IOError, ContentTooShortError:
            pass


def _dirs(MYDATA_dir=None):
    if MYDATA_dir is None:
        DATA = os.environ['DATA'] 
        ME = os.environ['ME']  
        MYDATA = os.path.join(DATA, ME, 'declass', 'kissinger')    
        RAW = os.path.join(MYDATA, 'raw')      
        PROCESSED = os.path.join(MYDATA, 'processed')
    return {'RAW': RAW, 'PROCESSED': PROCESSED}


def pdftotext_all(MYDATA_dir=None):
    """
    Runs the unix pdftotext utility on pdf files in the 'processed' directory 
    and moves these pdfs to 'raw'. 
    """
    dirs = _dirs(MYDATA_dir)
    RAW = dirs['RAW']
    PROCESSED = dirs['PROCESSED']
    paths = get_paths(PROCESSED)
    for p in paths:
        try:
            _pdftotext(p)
        except SyntaxError:
            continue
        file_name = os.path.split(p)[1]
        new_path = os.path.join(RAW, file_name)
        command = 'mv %s %s'%(p, new_path)
        os.system(command)  
    

def _pdftotext(path):
    command = 'pdftotext -layout %s'%path
    os.system(command)


def connect_db(login_file=login_file):
    """
    Connects to the DB. You need to have the DB info saved in a yaml file 
    under the login_file path. 
    """
    login_info = yaml.load(open(login_file))
    host_name = login_info['host_name']
    db_name = login_info['db_name']
    user_name = login_info['user_name']
    pwd = login_info['pwd']
    conn = pymysql.connect(host=host_name, user=user_name, passwd=pwd, 
            db=db_name)
    conn.autocommit(1)
    cursor = conn.cursor()    
    return cursor


def update_table(results_json, db_cursor, table_name='Kissinger', 
        MYDATA_dir=None, fails_file='/tmp/Kissinger_fails.txt'):
    """
    Updates the Kissenger table in the db. Table created with:
    
    CREATE TABLE Kissinger (
        id INT NOT NULL AUTO_INCREMENT PRIMARY KEY, 
        doc_id varchar(32), 
        body longtext, 
        names varchar(64), 
        time varchar(32),
        year int,
        subject varchar(64));
    """
    for i, result in enumerate(results_json):
        _update_table_func(result, db_cursor, table_name, MYDATA_dir,
                fails_file)
        if i%100==0: 
            print 'done with %s documents'%i


def _update_table_func(doc_data, db_cursor, table_name, MYDATA_dir,
        fails_file):
    doc_id = parse_doc_id(doc_data)
    PROCESSED = _dirs(MYDATA_dir)['PROCESSED']
    doc_path = os.path.join(PROCESSED, doc_id + '.txt')
    try:
        with open(doc_path) as f:
            body = f.read()
            body = _clean_body(body)
    except IOError:
        pass
    subj = parse_subject(doc_data)
    year = parse_year(doc_data)
    sql = ('insert ignore into %s (doc_id, body, names, time, year) ' 
            'Values("%s", "%s", "%s", "%s", %s)')%(table_name, doc_id, 
                    body, subj['names'], subj['time'], year)
    ff = open(fails_file, 'w')
    try:
        db_cursor.execute(sql)
    except ProgrammingError:
        ff.writeline(doc_id)
        print doc_id
    ff.close()


def _clean_body(text):
    """
    Replace all " with ' so as not to clash with sql in _update_table_func
    """
    text = re.sub(r'"', "'", text)
    return text
    
    


if __name__ == "__main__":
    import os
    DATA = os.environ['DATA'] 
    ME = os.environ['ME']  
    MYDATA = os.path.join(DATA, ME, 'declass', 'kissinger')    
    RAW = os.path.join(MYDATA, 'raw')      
    PROCESSED = os.path.join(MYDATA, 'processed')
    
    master_url = ("http://foia.state.gov/searchapp/Search/SubmitSimpleQuery"
            "?_dc=1382218348286&searchText=*&beginDate=false&endDate="
            "false&collectionMatch=KISSINGER&postedBeginDate=false&"
            "postedEndDate=false&caseNumber=false&page=1&start=0&limit=5000#")

    master_path = os.path.join(MYDATA, 'master_doc_data.json')
    #get_data(master_url, master_load_path=master_path)
    #push_data(master_url, master_load_path=master_path)
    update_column(master_url, 'date',  master_load_path=master_path, limit=None)


    
