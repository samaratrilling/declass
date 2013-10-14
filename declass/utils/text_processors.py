from collections import Counter, defaultdict
from functools import partial
from hashlib import sha224
import random
import copy

import nltk
import numpy as np

from . import filefilter, nlp, common
from common import lazyprop


class BaseTokenizer(object):
    """
    Base class, don't use directly.
    """
    def text_to_counter(self, text):
        """
        Return a counter associated to tokens in text.  
        Filter/transform words according to the scheme this Tokenizer uses.

        Parameters
        ----------
        text : String

        Returns
        -------
        tokens : Counter
            keys = the tokens
            values = counts of the tokens in text
        """
        return Counter(self.text_to_token_list(text))

    def path_to_token_list(self, path):
        """
        Return a list of tokens.  
        Filter/transform words according to the scheme this Tokenizer uses.

        Parameters
        ----------
        path : String
            Path to a file to read

        Returns
        -------
        tokens : List
            Tokenized text, e.g. ['hello', 'my', 'name', 'is', 'ian']
        """
        with open(path, 'r') as f:
            text = f.read()

        return self.text_to_token_list(text)


class TokenizerBasic(BaseTokenizer):
    """
    A simple tokenizer.  Extracts word counts from text.

    Keeps only non-stopwords, converts to lowercase,
    keeps words of length >=2.
    """
    def text_to_token_list(self, text):
        """
        Return a list of tokens.  
        Filter/transform words according to the scheme this Tokenizer uses.

        Parameters
        ----------
        text : String

        Returns
        -------
        tokens : List
            Tokenized text, e.g. ['hello', 'my', 'name', 'is', 'ian']
        """
        tokens = nlp.word_tokenize(text, L=2, numeric=False)

        return [word.lower() for word in tokens if not nlp.is_stopword(word)]


class TokenizerPOSFilter(BaseTokenizer):
    """
    Tokenizes, does POS tagging, then keeps words that match particular POS.
    """
    def __init__(
        self, pos_types=[], sent_tokenizer=nltk.sent_tokenize,
        word_tokenizer=nltk.word_tokenize, pos_tagger=nltk.pos_tag):
        """
        Parameters
        ----------
        pos_types : List of Strings
            Parts of Speech to keep
        sent_tokenizer : Sentence tokenizer function
            Splits text into a list of sentences (each sentence is a string)
        word_tokenizer : Word tokenizer function
            Splits strings into a list of words (each word is a string)
        pos_tagger : POS tagging function
            Given a list of words, returns a list of tuples (word, POS)
        """
        self.pos_types = set(pos_types)
        self.sent_tokenizer = sent_tokenizer
        self.word_tokenizer = word_tokenizer
        self.pos_tagger = pos_tagger

    def text_to_token_list(self, text):
        """
        Tokenize a list of text that (possibly) includes multiple sentences.
        """
        # sentences = [['I am Ian.'], ['Who are you?']]
        sentences = self.sent_tokenizer(text)
        # tokenized_sentences = [['I', 'am', 'Ian.'], ['Who', 'are', 'you?']]
        tokenized_sentences = [self.word_tokenizer(sent) for sent in sentences]
        # tagged_sentences = [[('I', 'PRP'), ('am', 'VBP'), ...]]
        tagged_sentences = [
            self.pos_tagger(sent) for sent in tokenized_sentences]

        # Returning a list of words that meet the filter criteria
        token_list = sum(
            [self._sent_filter(sent) for sent in tagged_sentences], [])

        return token_list

    def _sent_filter(self, tokenized_sent):
       return [word for (word, pos) in tokenized_sent if pos in self.pos_types] 


class SparseFormatter(object):
    """
    Base class for sparse formatting, e.g. VW or svmlight.  
    Not meant to be directly used.
    """
    def _parse_sstr(self, feature_str):
        """
        Parses a sparse feature string and returns 
        feature_values = {feature1: value1, feature2: value2,...}
        """
        # We currently don't support namespaces, so feature_str must start
        # with a space then feature1[:value1] feature2[:value2] ...
        assert feature_str[0] == ' '
        feature_str = feature_str[1:]

        feature_values = {}
        feature_values_list = feature_str.split()
        for fv in feature_values_list:
            feature, value = fv.split(':')
            value = float(value) if value else 1.0
            feature_values.update({feature: value})

        return feature_values

    def sstr_to_dict(self, sstr):
        """
        Returns a dict representation of sparse record string.
        
        Parameters
        ----------
        sstr : String
            String representation of one record.

        Returns
        -------
        record_dict : Dict
            possible keys = 'target', 'importance', 'doc_id', 'feature_values'

        Notes
        -----
        rstrips newline characters from sstr before parsing.
        """
        sstr = sstr.rstrip('\n').rstrip('\r')

        idx = sstr.index(self.preamble_char)
        preamble, feature_str = sstr[:idx], sstr[idx + 1:]

        record_dict = self._parse_preamble(preamble)

        record_dict['feature_values'] = self._parse_sstr(feature_str)

        return record_dict

    def sstr_to_info(self, sstr):
        """
        Returns the full info dictionary corresponding to a sparse record
        string.  This holds "everything."

        Parameters
        ----------
        sstr : String
            String representation of one record.

        Returns
        -------
        info : Dict
            possible keys = 'tokens', 'target', 'importance', 'doc_id',
                'feature_values', etc...
        """
        info = self.sstr_to_dict(sstr)
        info['tokens'] = self._dict_to_tokens(info)

        return info

    def _dict_to_tokens(self, record_dict):
        token_list = []
        if 'feature_values' in record_dict:
            for feature, value in record_dict['feature_values'].iteritems():
                int_value = int(value)
                assert int_value == value
                token_list += [feature] * int_value

        return token_list

    def sstr_to_token_list(self, sstr):
        """
        Convertes a sparse record string to a list of tokens (with repeats)
        corresponding to sstr.

        E.g. if sstr represented the dict {'hi': 2, 'bye': 1}, then
        token_list = ['hi', 'hi', 'bye']  (up to permutation).

        Parameters
        ----------
        sstr : String
            Formatted according to self.format_name
            Note that the values in sstr must be integers.

        Returns
        -------
        token_list : List of Strings
        """
        record_dict = self.sstr_to_dict(sstr)
        return self._dict_to_tokens(record_dict)

    def sfile_to_token_iter(self, filepath_or_buffer, limit=None):
        """
        Return an iterator over filepath_or_buffer that returns, line-by-line,
        a token_list.

        Parameters
        ----------
        filepath_or_buffer : string or file handle / StringIO.
            File should be formatted according to self.format.

        Returns
        -------
        token_iter : Iterator
            E.g. token_iter.next() gets the next line as a list of tokens.
        """
        open_file, was_path = common.openfile_wrap(filepath_or_buffer, 'r')

        for index, line in enumerate(open_file):
            if index == limit:
                raise StopIteration
            yield self.sstr_to_token_list(line)

        if was_path:
            open_file.close()


class VWFormatter(SparseFormatter):
    """
    Converts in and out of VW format (namespaces currently not supported).
    https://github.com/JohnLangford/vowpal_wabbit/wiki/Input-format

    [target] [Importance [Tag]]| feature1[:value1] feature2[:value2] ...

    Every single whitespace, pipe, colon, and newline is significant.
    """
    def __init__(self):
        self.format_name = 'vw'
        self.preamble_char = '|'

    def get_sstr(
        self, feature_values=None, target=None, importance=None, doc_id=None):
        """
        Return a string reprsenting one record in sparse VW format:

        Parameters
        ----------
        feature_values : Dict-like 
            {feature1: value1,...}
        target : Real number
            The value we are trying to predict.
        importance : Real number
            The importance weight to associate to this example.
        doc_id : Number or string
            A name for this example.

        Returns
        -------
        formatted : String
            Formatted in VW format
        """
        if doc_id:
            # If doc_id, then we must have importance.
            # The doc_id sits right against the pipe.
            assert importance is not None
            formatted = " %s|" % doc_id
            # If no doc_id, insert a space to the left of the pipe.
        else:
            formatted = " |"

        if importance:
            # Insert a space to the left of importance.
            formatted = " " + str(importance) + formatted

        if target:
            # target gets stuck on the end
            formatted = str(target) + formatted

        # The feature part must start with a space unless there is a namespace.
        formatted += ' '
        for word, count in feature_values.iteritems():
            formatted += "%s:%s " % (word, count)

        return formatted

    def _parse_preamble(self, preamble):
        """
        Parse the VW preamble: [target] [Importance [Tag]]
        and return a dict with keys 'doc_id', 'target', 'importance' iff
        the corresponding values were found in the preamble.
        """
        # If preamble was butted directly against a pipe, then the right-most
        # part is a doc_id....extract it and continue.
        if preamble[-1] != ' ':
            doc_id_left = preamble.rfind(' ')
            doc_id = preamble[doc_id_left + 1:]
            preamble = preamble[: doc_id_left]
        else:
            doc_id = None

        # Step from left to right through preamble.  
        # We are in the target until we encounter the first space...if there
        # is no target, then the first character will be a space.
        in_target = True
        target = ''
        importance = ''
        for char in preamble:
            if char == ' ':
                in_target = False
            elif in_target:
                target += char
            else:
                importance += char

        parsed = {}
        items = (
            ('doc_id', doc_id), ('target', target), ('importance', importance))
        for key, value in items:
            if value:
                parsed[key] = value
        
        return parsed


class SVMLightFormatter(SparseFormatter):
    """
    For formatting in/out of SVM-Light format (info not currently supported)
    http://svmlight.joachims.org/

    <line> .=. <target> <feature>:<value> <feature>:<value> ...
    <target> .=. +1 | -1 | 0 | <float> 
    <feature> .=. <integer> | "qid"
    <value> .=. <float>
    <info> .=. <string>
    """
    def __init__(self):
        """
        """
        self.format_name = 'svmlight'
        self.preamble_char = ' '

    def get_sstr(
        self, feature_values=None, target=1, importance=None, doc_id=None):
        """
        Return a string reprsenting one record in SVM-Light sparse format
        <line> .=. <target> <feature>:<value> <feature>:<value>

        Parameters
        ----------
        feature_values : Dict-like
            {hash1: value1,...}
        target : Real number
            The value we are trying to predict.

        Returns
        -------
        formatted : String
            Formatted in SVM-Light
        """
        # For now, just use 0 for <target>
        formatted = str(target) + ' '

        for word, count in feature_values.iteritems():
            formatted += " %s:%s" % (word, count)

        return formatted

    def _parse_preamble(self, preamble):
        return {'target': float(preamble)}


class SFileFilter(object):
    """
    Filters results stored in sfiles (sparsely formattted bag-of-words files).
    """
    def __init__(self, formatter, bit_precision=18):
        """
        Parameters
        ----------
        formatter : Subclass of SparseFormatter
        bit_precision : Integer
            Hashes are taken modulo 2**bit_precision.  Currently must by < 32.
        """
        assert isinstance(bit_precision, int) and bit_precision <= 32

        self.formatter = formatter
        self.bit_precision = bit_precision

        self.precision = 2**bit_precision
        self.sfile_loaded = False

    def hash_fun(self, token):
        # TODO This is a slow high precision function...no need for that much
        # precision!  zlib.adler32 has too many collisions...
        # VW uses 32 bit murmerhash
        hexhash = sha224(token).hexdigest()

        return int(hexhash, 16) % self.precision

    def load_sfile(self, sfile):
        """
        Load an sfile, building self.token2hash and self.hash2token.

        Parameters
        ----------
        sfile : String or open file
            The sparse formatted file we will load.
        """
        # TODO Allow loading of more than one sfile
        assert not self.sfile_loaded

        # Build token2hash
        token2hash, token_score, num_docs_in = self._load_sfile_fwd(sfile)

        # Build hash2token
        hash2token = self._load_sfile_rev(token2hash)

        self.token2hash = token2hash
        self.token_score = token_score
        self.num_docs_in = num_docs_in
        self.hash2token = hash2token

        self.sfile_loaded = True

    def _load_sfile_fwd(self, sfile):
        """
        Builds the "forward" objects involved in loading an sfile.
        """
        token2hash = {}
        token_score = defaultdict(float)
        num_docs_in = defaultdict(int)

        open_file, was_path = common.openfile_wrap(sfile, 'r')

        # Each line represents one document
        for line in open_file:
            tokens_in_doc = set()
            record_dict = self.formatter.sstr_to_dict(line)
            for token, value in record_dict['feature_values'].iteritems():
                token2hash[token] = self.hash_fun(token)
                tokens_in_doc.add(token)
                token_score[token] += value

            for token in tokens_in_doc:
                num_docs_in[token] += 1
        
        if was_path:
            open_file.close()

        return token2hash, token_score, num_docs_in

    def _load_sfile_rev(self, token2hash, seed=None):
        """
        Builds the "reverse" objects involved in loading an sfile.

        Returns
        -------
        hash2token : Dict
        """
        hash2token = {}

        all_tokens = token2hash.keys()
        all_hashes = token2hash.values()
        hash_counts = Counter(all_hashes)
        
        # Make sure we don't have too many collisions
        vocab_size = len(all_tokens)
        num_collisions = vocab_size - len(hash_counts)
        if num_collisions > vocab_size / 20.:
            msg = (
                "num_collisions = %d.  vocab_size = %d.  Try using the "
                "function collision_probability to estimate needed precision" 
                % ( num_collisions, vocab_size))
            raise CollisionError(msg)

        collisions = set()
        for token, hash_value in zip(all_tokens, all_hashes):
            if hash_counts[hash_value] == 1:
                hash2token[hash_value] = token
            else:
                collisions.add(token)

        self._resolve_collisions(
            collisions, hash_counts, token2hash, hash2token, seed=seed)

        return hash2token

    def _resolve_collisions(
        self, collisions, hash_counts, token2hash, hash2token, seed=None):
        """
        Function used to resolve collisions.  Finds a hash value not already
        used using a "random probe" method.

        Parameters
        ----------
        collisions : Set of tokens
        hash_counts : Dict
            keys = hash_values, values = number of times each hash_value
            appears in token2hash
        token2hash : Dict
        hash2token : Dict
        """
        # Seed for testing
        random.seed(seed)

        for token in collisions:
            old_hash = token2hash[token]
            new_hash = old_hash
            # If hash_counts[old_hash] > 1, then the collision still must be
            # resolved.  In that case, change new_hash and update hash_counts
            if hash_counts[old_hash] > 1:
                # hash_counts is the only dict (at this time) holding every
                # hash you have ever seen
                while new_hash in hash_counts:
                    new_hash = random.randint(0, self.precision - 1)
                    new_hash = new_hash % self.precision
                hash_counts[old_hash] -= 1
                hash_counts[new_hash] = 1
            # Update dictionaries
            hash2token[new_hash] = token
            token2hash[token] = new_hash

    def filter_sfile(self, infile, outfile):
        pass

    def remove_extreme_tokens(self, num_below=5, frac_above=0.5):
        pass

    def remove_tokens(self, token_list):
        pass

    def add_doc_id_filter(self, doc_id, enforce_exact=True):
        pass


def collision_probability(self, vocab_size, bit_precision):
    """
    Approximate probability of collision (assuming perfect hashing)

    Parameters
    ----------
    vocab_size : Integer
        Number of unique words in vocabulary
    bit_precision : Integer
        Number of bits in space we are hashing to
    """
    return vocab_size**2 / 2.**(bit_precision + 1)


class CollisionError(Exception):
    pass

# TODO : Make this have a "dict" attribute that has many dict-like methods
# TODO : Should this object even hash....why not just add in sequence?  No need to compute a hash (other than implicityly while checking if item in self.dict).  The most common use-case is bulk adding of files...
