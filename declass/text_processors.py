from jrl_utils.src import common, parallel_easy, nlp



class Tokenizer(object):
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
            word.lower() for word in tokens if word not in nlp.stopwords_eng)
        
        return tokens


class VWFormatter(object):
    def __init__(self):
        self.name = 'vw'

    def format(tokens, rec_name=''):
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
        formatted = "'%s |" % (tag)

        for word, count in tokens.iteritems():
            formatted += " %s:%s" % (word, count)

        return formatted


