"""
Use the database module to extract kissinger files to text
"""
import os
import yaml

import pandas as pd

from declass.utils.database import DBCONNECT


# Set paths
# You should have your own login_file somewhere.  
# DO NOT commit this to the (public) repository!
login_file = os.path.join('/home/langmore/lib/declass/', 'conf', 'db_login.yml')
RAW = os.path.join(os.getenv('DATA'), os.getenv('ME'), 'kiss-01/raw')
bodyfiles_basepath = os.path.join(RAW, 'bodyfiles')
metafile_path = os.path.join(RAW, 'meta', 'meta.csv')

# Get login info
login_info = yaml.load(open(login_file))
host_name = login_info['host_name']
db_name = login_info['db_name']
user_name = login_info['user_name']
pwd = login_info['pwd']

# Set up DB and retrieve all records as a dict
# First run with "limit 10", then erase once things work
dbCon = DBCONNECT(host_name, db_name, user_name, pwd)
#records = dbCon.run_query('select * from Kissinger limit 10;')
records = dbCon.run_query('select * from Kissinger;')


# Write records to disk
meta = {
    'doc_id': [], 'time': [], 'names': [], 'year': [], 'subject': [],
    'date': []}

for rec in records:
    doc_id = rec['doc_id']
    # Write the body text
    filepath = os.path.join(bodyfiles_basepath, doc_id + '.txt')
    with open(filepath, 'w') as f:
        f.write(rec['body'])

    meta['doc_id'].append(doc_id)
    for field in ['time', 'names', 'year', 'subject', 'date']:
        meta[field].append(rec[field])

meta = pd.DataFrame(meta)
meta.to_csv(metafile_path, sep='|', index=False)


