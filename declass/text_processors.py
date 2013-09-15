from collections import Counter
from functools import partial

from . import filefilter, nlp, common


class TokenizerBasic(object):
    """
    A simple tokenizer.  Extracts word counts from text.

    Keeps only non-stopwords, converts to lowercase,
    keeps words of length >=2.
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

        return [word for word in tokens if not nlp.is_stopword(word)]

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


class SparseFormatter(object):
    """
    Base class for sparse formatting, e.g. VW or svmlight.  
    Not meant to be directly used.
    """
    def _parse_feature_str(self, feature_str):
        """
        Returns feature_values = {feature1: value1, feature2: value2,...}
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

    def get_dict(self, record_str):
        """
        Returns a dict representation of record_str.
        
        Parameters
        ----------
        record_str : String
            String representation of one record.

        Returns
        -------
        record_dict : Dict
            possible keys = 'target', 'importance', 'tag', 'feature_values'
        """
        idx = record_str.index(self.preamble_char)
        preamble, feature_str = record_str[:idx], record_str[idx + 1:]

        record_dict = self._parse_preamble(preamble)

        record_dict['feature_values'] = self._parse_feature_str(feature_str)

        return record_dict

    def get_token_list(self, record_str):
        """
        Returns a list of tokens (with repeats) corresponding to record_str.

        If record_str represented the dict {'hi': 2, 'bye': 1}, then
        token_list = ['hi', 'hi', 'bye']  (up to permutation).

        Parameters
        ----------
        record_str : String
            Formatted according to self.format_name
            Note that the values in record_str must be integers.

        Returns
        -------
        token_list : List of Strings
        """
        record_dict = self.get_dict(record_str)
        token_list = []
        if 'feature_values' in record_dict:
            for feature, value in record_dict['feature_values'].iteritems():
                int_value = int(value)
                assert int_value == value
                token_list += [feature] * int_value

        return token_list

    def file_to_token_iter(self, filepath_or_buffer):
        """
        Return an iterator over filepath_or_buffer that returns, line-by-line,
        a token_list.

        Parameters
        ----------
        filepath_or_buffer : string or file handle / StringIO.

        Returns
        -------
        token_iter : Iterator
            E.g. token_iter.next() gets the next line as a list of tokens.
        """
        open_file, was_path = common.openfile_wrap(filepath_or_buffer, 'r')

        for line in open_file:
            line = line.rstrip('\n').rstrip('\r')
            yield self.get_token_list(line)

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

    def get_str(
        self, feature_values=None, target=None, importance=None, tag=None):
        """
        Return a string reprsenting one record in VW format:

        Parameters
        ----------
        feature_values : Dict-like 
            {feature1: value1,...}
        target : Real number
            The value we are trying to predict.
        importance : Real number
            The importance weight to associate to this example.
        tag : Number or string
            A name for this example.

        Returns
        -------
        formatted : String
            Formatted in VW format
        """
        if tag:
            # If tag, then we must have importance.
            # The tag sits right against the pipe.
            assert importance is not None
            formatted = " %s|" % tag
            # If no tag, insert a space to the left of the pipe.
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
        and return a dict with keys 'tag', 'target', 'importance' iff
        the corresponding values were found in the preamble.
        """
        # If preamble was butted directly against a pipe, then the right-most
        # part is a tag....extract it and continue.
        if preamble[-1] != ' ':
            tag_left = preamble.rfind(' ')
            tag = preamble[tag_left + 1:]
            preamble = preamble[: tag_left]
        else:
            tag = None

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
        items = (('tag', tag), ('target', target), ('importance', importance))
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

    def get_str(
        self, feature_values=None, target=1, importance=None, tag=None):
        """
        Return a string reprsenting one record in SVM-Light format
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


def path_to_token_list(tokenizer, path):
    return tokenizer.path_to_token_list(path)


class TokenStreamer(object):
    """
    Streams tokens from a source of text files.
    """
    def __init__(
        self, tokenizer, base_path=None, file_type='*', paths=None,
        limit=None):
        """
        Parameters
        ----------
        tokenizer : text_processors.Tokenizer object
            Used to create streams of tokens
        base_path : String
            The base directory.  Get all files of file_type within this.
        file_type : String
            Glob filter on the file, e.g. '*.txt'
        paths : Iterable
            E.g. a list of paths.
        limit : Integer
            Raise StopIteration after returning limit token lists.
        """
        self.tokenizer = tokenizer
        self.base_path = base_path
        self.file_type = file_type
        self.paths = paths
        self.limit = limit

    def __iter__(self):
        """
        Stream token lists from pre-defined path lists.
        """
        tokenizer = self.tokenizer
        base_path = self.base_path
        file_type = self.file_type
        paths = self.paths
        limit = self.limit

        assert (paths is None) or (base_path is None)

        if base_path:
            paths = filefilter.get_paths_iter(base_path, file_type=file_type)

        for i, onepath in enumerate(paths):
            if self.limit:
                if i == limit:
                    raise StopIteration

            with open(onepath, 'r') as f:
                text = f.read()
                yield tokenizer.text_to_token_list(text)


