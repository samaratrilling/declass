from collections import Counter

from . import filefilter, nlp


class TokenizerBasic(object):
    def __init__(self):
        pass

    def text_to_counter(self, text):
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
        Keeps only non-stopwords, converts to lowercase,
        keeps words of length >=2.
        """
        tokens = nlp.word_tokenize(text, L=2, numeric=False)
        tokens = Counter(
            word.lower() for word in tokens if not nlp.is_stopword(word))
        
        return tokens


class SparseFormatter(object):
    """
    Base class for sparse formatting, e.g. VW or svmlight.
    """
    def _parse_feature_str(self, feature_str):
        """
        Returns a feature_dict = {feature1: value1, feature2: value2,...}
        """
        # We currently don't support namespaces, so feature_str must start
        # with a space then feature1[:value1] feature2[:value2] ...
        assert feature_str[0] == ' '
        feature_str = feature_str[1:]

        feature_dict = {}
        feature_values = feature_str.split()
        for fs in feature_values:
            feature, value = fs.split(':')
            feature = feature
            value = float(value) if value else 1.0
            feature_dict.update({feature: value})

        return feature_dict

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
            possible keys = 'target', 'importance', 'tag', 'feature_dict'
        """
        idx = record_str.index(self.preamble_char)
        preamble, feature_str = record_str[:idx], record_str[idx + 1:]

        record_dict = self._parse_preamble(preamble)

        record_dict['feature_dict'] = self._parse_feature_str(feature_str)

        return record_dict


class VWFormatter(SparseFormatter):
    """
    Formats in and out of VW format (namespaces currently not supported).
    https://github.com/JohnLangford/vowpal_wabbit/wiki/Input-format

    [target] [Importance [Tag]]| feature1[:value1] feature2[:value2] ...

    Every single whitespace, pipe, colon, and newline is significant.
    """
    def __init__(self):
        self.name = 'vw'
        self.preamble_char = '|'

    def get_str(self, features, target=None, importance=None, tag=None):
        """
        Return a string reprsenting one record in VW format:

        Parameters
        ----------
        features : Dict-like 
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
        for word, count in features.iteritems():
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
    def __init__(self):
        """
        For formatting in SVM-Light format (http://svmlight.joachims.org/)

        <line> .=. <target> <feature>:<value> <feature>:<value> ... # info
        <target> .=. +1 | -1 | 0 | <float> 
        <feature> .=. <integer> | "qid"
        <value> .=. <float>
        <info> .=. <string>
        """
        self.name = 'svmlight'
        self.preamble_char = ' '

    def get_str(self, features, target=1):
        """
        Return a string reprsenting one record in SVM-Light format
        <line> .=. <target> <feature>:<value> <feature>:<value>

        Parameters
        ----------
        features : Dict-like
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

        for word, count in features.iteritems():
            formatted += " %s:%s" % (word, count)

        return formatted

    def _parse_preamble(self, preamble):
        return {'target': float(preamble)}

