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
RAW = os.path.join(os.getenv('DATA'), os.getenv('ME'), 'cables-01/raw')
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
fields = ['doc_nbr', 'concepts', 'date', 'msgfrom', 'office', 'origclass', 'orighandling', 'PREVCLASS', 'PREVHANDLING', 'REVIEW_AUTHORITY', 'REVIEW_DATE', 'SUBJECT', 'TAGS', 'MSGTO', 'TYPE', 'CHANNEL', 'PREVCHANNEL', 'DISP_COMMENT', 'DISP_AUTH', 'DISP_DATE', 'REVIEW_AUTH', 'REVIEW_FLAGS', 'REVIEWHISTORY', 'MSGTEXT', 'cleanFrom', 'cleanTo']
fields = [f.lower() for f in fields]
meta_fields = [f for f in fields if f != 'msgtext']

records = dbCon.run_query(
    'select %s '
    'from statedeptcables '
     'limit 5000000;' % ', '.join(fields))

# Write records to disk
meta = {f: [] for f in meta_fields}
meta['doc_id'] = []


for rec in records:
    doc_id = rec['doc_nbr'].replace(' ', '_')
    # Write the body text
    filepath = os.path.join(bodyfiles_basepath, doc_id + '.txt')
    with open(filepath, 'w') as f:
        f.write(rec['msgtext'])

    meta['doc_id'].append(doc_id)
    for field in meta_fields:
        meta[field].append(rec[field])

meta = pd.DataFrame(meta)
meta.to_csv(metafile_path, sep='|', index=False)



