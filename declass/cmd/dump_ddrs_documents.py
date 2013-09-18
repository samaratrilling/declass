"""
Dump the DDRS Documents to flat files.
"""
import argparse
import declass.declass.ddrs as ddrs

def _cli():
    parser = argparse.ArgumentParser(
        description=globals()['__doc__'],
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument(
        '-o', '--outdir', required=True,
        help='Directory to write formatted files.')

    parser.add_argument(
        '-f', '--format', required=True,
        help='Format to write out the files.')
    args = parser.parse_args()

    dbCon = ddrs.make_db_connect()
    rows = dbCon.run_query("SELECT id, body FROM Document")
    documents = (ddrs.Document(row["id"], row["body"]) 
                 for row in rows)
    ddrs.Document.write_to_files(args.outdir, documents, args.format)

if __name__ == '__main__':
    _cli()


