"""
Converts files into newline separated lists of tokens.
Tokens are represented in Vowpal Wabbit format.
"""
import argparse
from functools import partial
import sys
from collections import Counter

from declass.declass import filefilter, text_processors, nlp

from parallel_easy.parallel_easy.base import imap_easy


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

    Vowpal Wabbit format is 
    https://github.com/JohnLangford/vowpal_wabbit/wiki/Input-format
    [target] [Importance [Tag]]| feature1[:value1] feature2[:value2] ...

    target is e.g. 0 or 1 for classification, 
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
        'paths', nargs='*',
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
        '--name_level', default=1, type=int,
        help='Form the record name using items this far back in the path'
        ' e.g. if name_level == 2, and path = mydata/1234/3.txt, then we will'
        ' have name = 1234_3')
    parser.add_argument(
        '--n_jobs', help="Use n_jobs to tokenize files.",
        type=int, default=1)
    parser.add_argument(
        '--chunksize', help="Have workers process CHUNKSIZE files at a time.",
        type=int, default=1000)

    # Parse and check args
    args = parser.parse_args()

    if args.base_path:
        assert args.paths == []
    elif args.paths == []:
        args.paths = sys.stdin

    # Call the module interface
    tokenize(
        args.outfile, paths=args.paths, base_path=args.base_path,
        tokenizer_type=args.tokenizer_type, name_level=args.name_level, 
        n_jobs=args.n_jobs, chunksize=args.chunksize)


def tokenize(
    outfile, paths=[], base_path=None, tokenizer_type='basic',
    name_level=1, n_jobs=1, chunksize=1000):
    """
    Write later if module interface is needed. See _cli for the documentation.
    """
    assert (paths == []) or (base_path is None)

    if base_path:
        paths = filefilter.get_paths(base_path, file_type='*')

    tokenizer_dict = {'basic': text_processors.TokenizerBasic}
    tokenizer = tokenizer_dict[tokenizer_type]()

    formatter = text_processors.VWFormatter()

    func = partial(_tokenize_one, tokenizer, formatter, name_level)

    results_iterator = imap_easy(func, paths, n_jobs, chunksize)

    for result in results_iterator:
        outfile.write(result + '\n')


def _tokenize_one(tokenizer, formatter, name_level, path):
    """
    Tokenize file contained in path.  Return results in a sparse format.
    """
    # If path comes from find (and a pipe to stdin), there will be newlines.
    path = path.strip()
    with open(path, 'r') as f:
        text = f.read()

    tokens = tokenizer.text_to_counter(text)

    # Format
    tag = filefilter.path_to_name(path, name_level=name_level)
    tok_str = formatter.get_str(tokens, importance=1, tag=tag)

    return tok_str



if __name__ == '__main__':
    _cli()
