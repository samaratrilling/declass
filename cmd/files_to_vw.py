"""
Converts files into newline separated lists of tokens.
Tokens are represented in Vowpal Wabbit format.
"""
import argparse
from functools import partial
import sys
from collections import Counter

from declass.declass import filefilter, text_processors, nlp

from jrl_utils.src import parallel_easy


def _cli():
    # Text to display after help
    epilog = """
    EXAMPLES

    Convert file1 and file2 to vw format, redirect to my_vw_file
    $ python files_to_vw.py file1 file2 > my_vw_file

    Convert all files in mydir/ to vw format
    $ python files_to_vw.py  --base_path=mydir

    Convert the first 10 files in mydir/ to vw format 
    $ find mydir/ -type f | head | python files_to_vw.py

    Vowpal Wabbit format is:

    [label] [weight] ['tag] | feature1:score1 feature2:score2 ...

    label is e.g. 0 or 1 for classification, 
    weight is an importance weight, 
    tag identifies the record and must be preceeded by a single-quote '
    scores default to 1,
    """
    parser = argparse.ArgumentParser(
        description=globals()['__doc__'], epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        '--base_path', dest='base_path', 
        help='Walk this directory for documents.')
    parser.add_argument(
        'path_list', nargs='*', type=argparse.FileType('r'), default=sys.stdin,
        help='Convert files in this space separated list.  If not specified,'
        ' use base_path or read paths from stdin.')
    parser.add_argument(
        '-o', '--outfile', dest='outfile', default=sys.stdout,
        type=argparse.FileType('w'),
        help='Write to OUT_FILE rather than sys.stdout.')
    parser.add_argument(
        '-t', '--tokenizer_type', default='basic',
        help='Use TOKENIZER_TYPE to tokenize the raw text')
    parser.add_argument(
        '--n_jobs', help="Use n_jobs to tokenize files.",
        type=int, default=1)
    parser.add_argument(
        '--chunksize', help="Have workers process CHUNKSIZE files at a time.",
        type=int, default=1000)

    # Parse and check args
    args = parser.parse_args()
    if args.base_path and (args.path_list == sys.stdin):
        args.path_list = None

    # Call the module interface
    tokenize(
        args.outfile, path_list=args.path_list, base_path=args.base_path,
        tokenizer_type=args.tokenizer_type, n_jobs=args.n_jobs,
        chunksize=args.chunksize)


def tokenize(
    outfile, path_list=None, base_path=None, tokenizer_type='basic', n_jobs=1,
    chunksize=50):
    """
    Write later if module interface is needed. See _cli for the documentation.
    """
    assert not (path_list and base_path)

    if base_path:
        path_list = filefilter.get_paths(base_path, fileType='*')

    tokenizer_dict = {'basic': text_processors.TokenizerBasic}
    tokenizer = tokenizer_dict[tokenizer_type]()

    formatter = text_processors.VWFormatter()

    func = partial(_tokenize_one, tokenizer, formatter)

    results_iterator = parallel_easy.imap_easy(
        func, path_list, n_jobs, chunksize)

    for result in results_iterator:
        outfile.write(result + '\n')


def _tokenize_one(tokenizer, formatter, path):
    """
    Tokenize file contained in path.  Return results in a sparse format.
    """
    # If path comes from find (and a pipe to stdin), there will be newlines.
    path = path.strip()
    text = open(path, 'rb').read()

    tokens = tokenizer.text_to_counter(text)

    # Format
    rec_name = filefilter.get_one_file_name(path)
    tok_str = formatter.write_str(tokens, rec_name=rec_name)

    return tok_str



if __name__ == '__main__':
    _cli()
