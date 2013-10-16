from collections import Counter, defaultdict
from functools import partial
import hashlib
import zlib
import random
import copy
import cPickle

import nltk
import numpy as np
import pandas as pd

from . import filefilter, nlp, common
from common import lazyprop, smart_open, SaveLoad


class BaseTokenizer(SaveLoad):
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


class MakeTokenizer(BaseTokenizer):
    """
    Makes a subclass of BaseTokenizer out of a function.
    """
    def __init__(self, tokenizer_func):
        """
        Parameters
        ----------
        tokenizer_func : Function
            Takes in strings, spits out lists of strings.
        """
        self.text_to_token_list = tokenizer_func


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
            # If the feature is an int, then store an int.  If float...
            # If no value, default to 1
            if value:
                value_to_use = self._string_to_number(value)
            else:
                value_to_use = 1

            feature_values[feature] =  value_to_use

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
        with smart_open(filepath_or_buffer) as open_file:
            for index, line in enumerate(open_file):
                if index == limit:
                    raise StopIteration
                yield self.sstr_to_token_list(line)


class VWFormatter(SparseFormatter):
    """
    Converts in and out of VW format (namespaces currently not supported).
    Many valid VW inputs are possible, we ONLY support

    [target] [Importance [Tag]]| feature1[:value1] feature2[:value2] ...

    Every single whitespace, pipe, colon, and newline is significant.

    See:
    https://github.com/JohnLangford/vowpal_wabbit/wiki/Input-format
    http://hunch.net/~vw/validate.html
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

        # Remove the trailing space...not required but it's screwy to have a
        # space-delimited file with a trailing space but nothing after it!
        if len(feature_values) > 0:
            formatted = formatted.rstrip()

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
                if key in ['target', 'importance']:
                    parsed[key] = self._string_to_number(value)
                else:
                    parsed[key] = value
        
        return parsed

    def _string_to_number(self, string):
        try:
            number = int(string)
        except ValueError:
            number = float(string)

        return number


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


class SFileFilter(SaveLoad):
    """
    Filters results stored in sfiles (sparsely formattted bag-of-words files).
    """
    def __init__(self, formatter, bit_precision=16, verbose=False):
        """
        Parameters
        ----------
        formatter : Subclass of SparseFormatter
        bit_precision : Integer
            Hashes are taken modulo 2**bit_precision.  Currently must by < 32.
        """
        assert isinstance(bit_precision, int)

        self.formatter = formatter
        self.bit_precision = bit_precision
        self.verbose = verbose

        self.precision = 2**bit_precision
        self.sfile_loaded = False

    def _get_hash_fun(self):
        """
        The fastest is the built in function hash.  Quick experimentation
        shows that this function maps similar words to similar values (not
        cryptographic) and therefore increases collisions...no big deal.

        hashlib.sha224 is up to 224 bit.
        """
        if self.bit_precision <= 64:
            hash_fun = lambda w: hash(w) % self.precision
        elif self.bit_precision <= 224:
            hash_fun = lambda w: (
                int(hashlib.sha224(w).hexdigest(), 16) % self.precision)
        else:
            raise ValueError("Precision above 224 bit not supported")
        
        return hash_fun

    def load_sfile(self, sfile):
        """
        Load an sfile, building self.token2id and self.id2token.

        Parameters
        ----------
        sfile : String or open file
            The sparse formatted file we will load.
        """
        # TODO Allow loading of more than one sfile
        assert not self.sfile_loaded

        # Build token2id
        token2id, token_score, doc_freq, num_docs = (
            self._load_sfile_fwd(sfile))

        self.token2id = token2id
        self.token_score = token_score
        self.doc_freq = doc_freq
        self.num_docs = num_docs

        self.sfile_loaded = True
        self.collisions_resolved = False

    def _load_sfile_fwd(self, sfile):
        """
        Builds the "forward" objects involved in loading an sfile.
        """
        token2id = {}
        token_score = defaultdict(float)
        doc_freq = defaultdict(int)
        num_docs = 0

        hash_fun = self._get_hash_fun()

        with smart_open(sfile) as open_file:
            # Each line represents one document
            for line in open_file:
                num_docs += 1
                record_dict = self.formatter.sstr_to_dict(line)
                for token, value in record_dict['feature_values'].iteritems():
                    hash_value = hash_fun(token)
                    token2id[token] = hash_value
                    token_score[token] += value
                    doc_freq[token] += 1

        return token2id, token_score, doc_freq, num_docs

    def resolve_collisions(self, seed=None):
        """
        Resolves collisions in self.token2id and sets self.id2token.
        Works by finding a new id value for every collision, making the map
        token2id injective.
        """
        # TODO Add support for resolving of collisions in the case that we
        # cannot make token2id injective.
        # TODO Make these three methods logical in their layout again
        self.id2token = self._load_sfile_rev(self.token2id, seed=seed)
        self.collisions_resolved = True

    def _load_sfile_rev(self, token2id, seed=None):
        """
        Builds the "reverse" objects involved in loading an sfile.

        Returns
        -------
        id2token : Dict
        """
        id2token = {}

        all_tokens = token2id.keys()
        all_ides = token2id.values()
        id_counts = Counter(all_ides)
        
        # Make sure we don't have too many collisions
        vocab_size = len(token2id)
        num_collisions = vocab_size - len(id_counts)
        self._print(
            "collisions, vocab_size = %d, %d" % (num_collisions, vocab_size))
        if num_collisions > vocab_size / 2.:
            msg = (
                "Too many collisions to be efficient: "
                "num_collisions = %d.  vocab_size = %d.  Try using the "
                "function collision_probability to estimate needed precision" 
                % ( num_collisions, vocab_size))
            raise CollisionError(msg)

        collisions = set()
        for token, id_value in zip(all_tokens, all_ides):
            if id_counts[id_value] == 1:
                id2token[id_value] = token
            else:
                collisions.add(token)

        self._resolve_collisions_core(
            collisions, id_counts, token2id, id2token, seed=seed)

        return id2token

    def _resolve_collisions_core(
        self, collisions, id_counts, token2id, id2token, seed=None):
        """
        Function used to resolve collisions.  Finds a id value not already
        used using a "random probe" method.

        Parameters
        ----------
        collisions : Set of tokens
        id_counts : Dict
            keys = id_values, values = number of times each id_value
            appears in token2id
        token2id : Dict
        id2token : Dict
        """
        # Seed for testing
        random.seed(seed)

        for token in collisions:
            old_id = token2id[token]
            new_id = old_id
            # If id_counts[old_id] > 1, then the collision still must be
            # resolved.  In that case, change new_id and update id_counts
            if id_counts[old_id] > 1:
                # id_counts is the only dict (at this time) holding every
                # id you have ever seen
                while new_id in id_counts:
                    new_id = random.randint(0, self.precision - 1)
                    new_id = new_id % self.precision
                id_counts[old_id] -= 1
                id_counts[new_id] = 1
            # Update dictionaries
            id2token[new_id] = token
            token2id[token] = new_id

    def filter_sfile(
        self, infile, outfile, doc_id_list=None, enforce_all_doc_id=True):
        """
        Change tokens to id values (using self.token2id) and remove
        tokens not in self.token2id.

        Parameters
        ----------
        infile : file path or buffer
        outfile : file path or buffer
        doc_id_list : Iterable over strings
            Remove rows with doc_id not in this list
        enforce_all_doc_id : Boolean
            If True (and doc_id is not None), raise exception unless all doc_id
            in doc_id_list are seen.
        """
        assert self.sfile_loaded, "Must load an sfile before you can filter"
        assert self.collisions_resolved, (
            "Must resolve collisions before you can filter")

        extra_filter = self._get_extra_filter(doc_id_list)

        with smart_open(infile) as f, smart_open(outfile, 'w') as g:
            # Each line represents one document
            for line in f:
                record_dict = self.formatter.sstr_to_dict(line)
                if extra_filter(record_dict):
                    record_dict['feature_values'] = {
                        self.token2id[token]: value 
                        for token, value
                        in record_dict['feature_values'].iteritems() 
                        if token in self.token2id}
                    new_sstr = self.formatter.get_sstr(**record_dict)
                    g.write(new_sstr + '\n')

        self._done_check(enforce_all_doc_id)

    def _get_extra_filter(self, doc_id_list):
        self._doc_id_seen = set()

        # Possible filters to use
        if doc_id_list is not None:
            self._doc_id_set = set(doc_id_list)
            def doc_id_filter(record_dict):
                doc_id = record_dict['doc_id']
                self._doc_id_seen.add(doc_id)

                return doc_id in self._doc_id_set
        else:
            self._doc_id_set = set()
            doc_id_filter = lambda record_dict: True

        # Add together all the filters into one function
        return lambda record_dict: doc_id_filter(record_dict)

    def _done_check(self, enforce_all_doc_id):
        """
        QA check to perform once we're done filtering an sfile.
        """
        # Make sure we saw all the doc_id we're supposed to
        if enforce_all_doc_id:
            assert self._doc_id_set.issubset(self._doc_id_seen), (
                "Did not see every doc_id in the passed doc_id_list")

    def remove_extreme_tokens(
        self, doc_freq_min=0, doc_freq_max=np.inf, doc_fraction_min=0,
        doc_fraction_max=1):
        """
        Remove extreme tokens from self (calling self.remove_tokens).

        Parameters
        ----------
        doc_freq_min : Integer
            Remove tokens that in less than this number of documents
        doc_freq_max : Integer
        doc_fraction_min : Float in [0, 1]
            Remove tokens that are in less than this fraction of documents
        doc_fraction_max : Float in [0, 1]
        """
        frame = self.to_frame()
        to_remove_mask = (
                  (frame.doc_freq < doc_freq_min)
                | (frame.doc_freq > doc_freq_max)
                | (frame.doc_freq < (doc_fraction_min * self.num_docs))
                | (frame.doc_freq > (doc_fraction_max * self.num_docs))
                )
        
        self._print(
            "Removed %d/%d tokens" % (to_remove_mask.sum(), len(frame)))
        self.remove_tokens(frame[to_remove_mask].index)

    def remove_tokens(self, tokens):
        """
        Remove tokens from appropriate attributes.  The removed tokens will
        removed when calling self.filter_sfile.

        Parameters
        ----------
        tokens : String or iterable over strings
            E.g. a single token or list of tokens
        """
        if isinstance(tokens, str):
            tokens = [tokens]

        for tok in tokens:
            id_value = self.token2id[tok]
            self.token2id.pop(tok)
            self.token_score.pop(tok)
            self.doc_freq.pop(tok)
            if hasattr(self, 'id2token'):
                self.id2token.pop(id_value)

    def _print(self, msg):
        if self.verbose:
            print(msg)

    def to_frame(self):
        """
        Return a dataframe representation of self.
        """
        token2id = self.token2id
        token_score = self.token_score
        doc_freq = self.doc_freq

        assert token2id.keys() == token_score.keys() == doc_freq.keys()
        frame = pd.DataFrame(
            {'id': token2id.values(),
             'token_score': token_score.values(),
             'doc_freq': doc_freq.values()},
            index=token2id.keys())
        frame.index.name = 'token'

        return frame

    @property
    def vocab_size(self):
        return len(self.token2id)


def collision_probability(vocab_size, bit_precision):
    """
    Approximate probability of collision (assuming perfect hashing).  See
    the Wikipedia article on "The birthday problem" for details.

    Parameters
    ----------
    vocab_size : Integer
        Number of unique words in vocabulary
    bit_precision : Integer
        Number of bits in space we are hashing to
    """
    exponent = - vocab_size * (vocab_size - 1) / 2.**bit_precision
    
    return 1 - np.exp(exponent)


class CollisionError(Exception):
    pass
