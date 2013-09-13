import unittest
from StringIO import StringIO
import sys
from numpy.testing import assert_allclose
from datetime import datetime
import copy
from collections import Counter, OrderedDict

from declass.declass import text_processors


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


class TestVWFormatter(unittest.TestCase):
    """
    """
    def setUp(self):
        self.Formatter = text_processors.VWFormatter

    def test_format_vw_01(self):
        rec_name = 'myname'
        tokens = OrderedDict([('hello', 1), ('dude', 3)])
        result = self.Formatter().format(tokens, rec_name=rec_name)
        benchmark = "'%s | hello:1 dude:3" % rec_name
        self.assertEqual(result, benchmark)
