import os

import declass.utils.cable_helpers as cb

from declass.utils.database import DBCONNECT


def main(write_dir, limit=None):
    """
    Parameters
    ----------
    write_dir : string
        dir you want the cables to go to
    limit : int

    Notes
    -----
    If you don't want all 1mil or so cables to go to the same dir; you can modify the dir write as desired. 
    Also, cb.get_cables is set to skip cables with empty text by default.
    """
    
    db = DBCONNECT(host_name='mysql.csail.mit.edu', db_name='declassification', user_name='declass', pwd='declass')
    cables = cb.get_cables(db=db, limit=limit)
    
    for c in cables:
        doc_nbr = c['DOC_NBR']
        text = c['MSGTEXT']
        with open(os.path.join(write_dir, doc_nbr), 'w') as f:
            f.write(text)

if __name__ == '__main__':
    DATA = os.getenv('DATA')
    DECLASS = os.path.join(DATA, 'prod', 'declass')
    CABLES = os.path.join(DECLASS, 'cables')
    from time import time
    t0 = time()
    main(write_dir=CABLES, limit=100000)
    t1 = time()
    print 'poll took %s seconds'%(t1-t0)

