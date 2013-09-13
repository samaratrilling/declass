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


class VWFormatter(object):
    def __init__(self):
        self.name = 'vw'

    def write_str(self, tokens, rec_name=''):
        """
        Return a string reprsenting one record in VW format:
        [label] [weight] ['tag] | feature1:score1 feature2:score2 ...

        Parameters
        ----------
        tokens : Dict or Counter
            {'word1': value1,...}

        Returns
        -------
        formatted : String
            Formatted in VW format

        Notes
        -----
        Pipes, spaces, and colons are special characters.  
        A single-quote is special before the pipe.  
        These should NOT be in your tokens.
        """
        formatted = "'%s |" % (rec_name)

        for word, count in tokens.iteritems():
            formatted += " %s:%s" % (word, count)

        return formatted

    def write_dict(self, record_str):
        """
        Returns a dict representation of record_str.
        
        Parameters
        ----------
        record_str : String
            String representation of one record in VW format.
        """
        _, feature_scores = record_str.split('|')
        feature_scores = feature_scores.split()

        record_dict = {}
        for fs in feature_scores:
            feature, score = fs.split(':')
            feature = feature
            score = float(score)
            record_dict.update({feature: score})

        return record_dict


class SVMLightFormatter(object):
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

    def write_str(self, tokens):
        """
        Return a string reprsenting one record in SVM-Light format
        <line> .=. <target> <feature>:<value> <feature>:<value>

        Parameters
        ----------
        tokens : Dict or Counter
            {'word1': value1,...}

        Returns
        -------
        formatted : String
            Formatted in SVM-Light
        """
        formatted = "0 |"

        for word, count in tokens.iteritems():
            formatted += " %s:%s" % (word, count)

        return formatted

    def write_dict(self, record_str):
        """
        Returns a dict representation of record_str.
        
        Parameters
        ----------
        record_str : String
            String representation of one record in VW format.
        """
        _, feature_scores = record_str.split('|')
        feature_scores = feature_scores.split()

        record_dict = {}
        for fs in feature_scores:
            feature, score = fs.split(':')
            feature = int(feature)
            score = float(score)
            record_dict.update({feature: score})

        return record_dict

