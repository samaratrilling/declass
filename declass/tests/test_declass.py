import unittest
from StringIO import StringIO
import sys
from numpy.testing import assert_allclose
from datetime import datetime
import copy
from collections import Counter, OrderedDict
import pandas as pd

from declass.utils import text_processors, streamers, topic_seek


class TestTokenizerBasic(unittest.TestCase):
    """
    """
    def setUp(self):
        self.Tokenizer = text_processors.TokenizerBasic

    def test_text_to_counter(self):
        text = "Hi there's:alot,of | food hi"
        result = self.Tokenizer().text_to_counter(text)
        benchmark = Counter(["hi", "there's", "alot", "food", "hi"])
        self.assertEqual(result, benchmark)


@unittest.skip('skipping VW - test decrecated')
class TestVWFormatter(unittest.TestCase):
    """
    """
    def setUp(self):
        self.Formatter = text_processors.VWFormatter

    def test_for1(self):
        rec_name = 'myname'
        tokens = OrderedDict([('hello', 1), ('dude', 3)])
        result = self.Formatter().write_str(tokens, rec_name=rec_name)
        benchmark = "'%s | hello:1 dude:3" % rec_name
        self.assertEqual(result, benchmark)

    def test_write_dict_01(self):
        record_str = "1 | hello:1 bye:2"
        result = self.Formatter().write_dict(record_str)
        benchmark = {'hello': 1.0, 'bye': 2.0}
        self.assertEqual(result, benchmark)


class TestTopic(unittest.TestCase):
    def setUp(self):
        self.Topics = topic_seek.Topics
        self.streamer = streamers.TextFileStreamer()
        def token_stream(tokens, doc_id=None):
            return tokens
        self.streamer.token_stream = token_stream
        

    def test_dictionary(self):
        tokens1 = ['Hi', 'this', 'is', 'is', 'not', 'is', 'this']
        tokens2 = ['one', 'two', 'one', 'three']
        T = self.Topics()
        T.streamer = ListStream([tokens1, tokens2])
        T.set_dictionary(no_below=0, no_above=1)
        result = T.dictionary.items()
        benchmark = [(0, 'this'), (1, 'is'), (2, 'three'), (3, 'two'),
                (4, 'Hi'), (5, 'not'), (6, 'one')]        
        self.assertEqual(result, benchmark)

    def test_get_words_docfreq(self):
        tokens1 = ['Hi', 'this', 'is', 'is', 'not', 'is', 'this']
        tokens2 = ['one', 'two', 'one', 'three']
        T = self.Topics()
        T.streamer = ListStream([tokens1, tokens2])
        T.set_dictionary(no_below=0, no_above=1)
        result = T.get_words_docfreq()
        benchmark = pd.DataFrame({'tokenid': [3,2,0,6,5,1,4], 'docfreq': [1]*7}, 
                index=['two', 'three', 'this', 'one', 'not', 'is', 'Hi'])
        benchmark = benchmark[['tokenid', 'docfreq']]
        self.assertEqual(result, benchmark)


class ListStream(object):
    def __init__(self, token_lists):
        self.token_lists = token_lists

    def token_stream(self, doc_id=None):
        """
        Uses 'dummy doc_id' as this is called in streamer applicatioins.
        """
        for t in self.token_lists:
            yield t

        

