from BeautifulSoup import BeautifulSoup
import urllib2
import re
import simplejson as json
import os
import wget
import yaml
import pymysql

from functools import partial
from urllib import ContentTooShortError

from parallel_easy.base import imap_easy, map_easy
from declass.utils.filefilter import get_paths


####yaml file containing login data
login_file = os.getenv("HOME") + '/.declass_db'

def main(master_url, master_load_path=None, master_save_path=None,
        MYDATA_dir=None, limit=None, n_jobs=7, chunksize=100):
    import pdb; pdb.set_trace()
    dirs = _dirs(MYDATA_dir)
    master_json = get_master_json(master_url, master_load_path, master_save_path)
    paths_iter = parse_file_urls(master_json['Results'])
    #cursor = connect_db()
    #doc_data = master_json['Results'][0]
    #update_db(cursor, doc_data)
    #change dir since wget doesn's seem to have a outdir option
    os.chdir(dirs['PROCESSED'])
    download_files(paths_iter, n_jobs, chunksize)
    pdftotext_all()



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

def parse_subject(doc_data):
    subject = doc_data['subject']
    s = re.search(r'WITH {,1}(.*) AT ([\d:]* [AP]\.M\.|)', subject)
    try:
        names = s.group(1)
        time = s.group(2)
    except AttributeError:
        s = re.search(r'WITH ([A-Z\s\.].*)$', subject)
        try:
            names = s.group(1)
            time = None
        except AttributeError:
            names = time = None
            print subject
    return {'names': names, 'time': time}

def parse_doc_id(doc_data):
    url_string = doc_data['pdfLink']
    doc = url_string.split('\\')[2]
    doc_id = doc.split('.')[0]
    return doc_id



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
    login_info = yaml.load(open(login_file))
    host_name = login_info['host_name']
    db_name = login_info['db_name']
    user_name = login_info['user_name']
    pwd = login_info['pwd']
    conn = pymysql.connect(host=host_name, user=user_name, passwd=pwd, db=db_name)
    conn.autocommit(1)
    cursor = conn.cursor(pymysql.cursors.DictCursor)    
    return cursor

def update_db(db_cursor, doc_data, table_name='Kissinger', MYDATA_dir=None):
    doc_id = parse_doc_id(doc_data)
    PROCESSED = _dirs(MYDATA_dir)['PROCESSED']
    doc_path = os.path.join(PROCESSED, doc_id + '.txt')
    try:
        with open(doc_path) as f:
            body = f.read()
    except IOError:
        pass
    subj = parse_subject(doc_data)
    sql = ("insert ignore into %s (doc_id, body, names, time) " 
            "Values('%s', '%s', '%s', '%s')")%(table_name, doc_id, 
                    body, subj['names'], subj['time'])
    db_cursor.execute(sql)


    
    



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
    main(master_url, master_load_path=master_path)
    
