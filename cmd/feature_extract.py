"""
Converts files into newline separated lists of tokens.
Tokens are represented in Vowpal Wabbit format.
"""
import argparse
from functools import partial
import sys
from collections import Counter

from jrl_utils.src import common, parallel_easy, nlp
from trec_2010.src import filefilter
from trec_2010.src import text_processors


def _cli():
    # Text to display after help
    epilog = """
    EXAMPLES

    Pass the first 10 entries in mydir/ as the path_list
    $ find mydir/ -type f | head | python tokenizer.py

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
        type=int, default=50)

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

    func = partial(_tokenize_one, tokenizer_type)

    results_iterator = parallel_easy.imap_easy(
        func, path_list, n_jobs, chunksize)

    for result in results_iterator:
        outfile.write(result + '\n')


def _tokenize_one(tokenizer_type, path):
    """
    Tokenize file contained in path.  Return results in a sparse format.
    """
    # If path comes from find (and a pipe to stdin), there will be newlines.
    path = path.strip()
    text = open(path, 'rb').read()

    # Extract raw tokens as an iterable over pairs
    # (word1, value1),...
    # E.g. the dict {'word1': value1, 'word2': value2}
    # 
    # To use a cutom tokenizer, write a function and add to the dict
    # tokenizer_funcs
    tokenizer_funcs = {'basic': _get_tokens_basic}
    tokens = tokenizer_funcs[tokenizer_type](text)

    # Format
    tok_str = _format_vw(tokens, tag=filefilter.get_one_file_name(path))

    return tok_str


def _format_vw(tokens, tag=''):
    """
    Return a string reprsenting one record in VW format:
    [label] [weight] ['tag] | feature1:score1 feature2:score2 ...

    Notes
    -----
    Pipes, spaces, and colons are special characters.  
    A single-quote is special before the pipe.  
    These should NOT be in your tokens.
    """
    vw_str = "'%s |" % (tag)

    for word, count in tokens.iteritems():
        vw_str += " %s:%s" % (word, count)

    return vw_str


def _get_tokens_basic(text):
    """
    Basic tokenizer.  

    Parameters
    ----------
    text : String

    Returns
    -------
    tokens : Counter
        keys = the tokens
        values = counts of the tokens in text

    Notes
    -----
    Keeps only non-stopwords, converts to lowercase, keeps words of length >=2.
    """
    tokens = nlp.word_tokenize(text, L=2, numeric=False)
    tokens = Counter(
        word.lower() for word in tokens if word not in nlp.stopwords_eng)
    
    return tokens

    

if __name__ == '__main__':
    _cli()
