import unittest
from StringIO import StringIO
import sys
from datetime import datetime
import copy
from collections import Counter, OrderedDict
import random

import pandas as pd
from numpy.testing import assert_allclose
from pandas.util.testing import assert_frame_equal

from declass.utils import text_processors, vw_helpers


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

    def test_format_01(self):
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


class TestVWHelpers(unittest.TestCase):
    def setUp(self):
        self.varinfo_path = 'files/varinfo'
        self.topics_file_1 = StringIO(
            "Version 7.3\nlabel: 11\n"
            "0 1.1 2.2\n"
            "1 1.11 2.22")
        self.num_topics_1 = 2
        self.predictions_file_1 = StringIO(
            "0.0 0.0 doc1\n"
            "0.0 0.0 doc2\n"
            "1.1 2.2 doc1\n"
            "1.11 2.22 doc2")
        self.start_line_1 = 2

    def test_parse_varinfo_01(self):
        result = vw_helpers.parse_varinfo(self.varinfo_path)
        benchmark = pd.DataFrame(
            {
                'feature_name': ['bcc', 'illiquids'], 
                'hash_val': [77964, 83330], 
                'max_val': [1., 2.], 
                'min_val': [0., 5.], 
                'rel_score': [1., 0.6405],
                'weight': [0.2789, -0.1786]}).set_index('hash_val')
        assert_frame_equal(result, benchmark)

    def test_parse_lda_topics_01(self):
        result = vw_helpers.parse_lda_topics(
            self.topics_file_1, self.num_topics_1, normalize=False)
        benchmark = pd.DataFrame(
            {'hash_val': [0, 1], 'topic_0': [1.1, 1.11], 'topic_1': [2.2, 2.22]}
            ).set_index('hash_val')
        assert_frame_equal(result, benchmark)

    def test_parse_lda_predictions_01(self):
        result = vw_helpers.parse_lda_predictions(
            self.predictions_file_1, self.num_topics_1, self.start_line_1,
            normalize=False)
        benchmark = pd.DataFrame(
            {'doc_id': ['doc1', 'doc2'], 'topic_0': [1.1, 1.11],
                'topic_1': [2.2, 2.22]}).set_index('doc_id')
        assert_frame_equal(result, benchmark)

    def test_find_start_line_lda_predictions(self):
        result = vw_helpers.find_start_line_lda_predictions(
            self.predictions_file_1, self.num_topics_1)
        self.assertEqual(result, 2)


class TestSFileFilter(unittest.TestCase):
    def setUp(self):
        self.outfile = StringIO()
        formatter = text_processors.VWFormatter()
        self.sfile_filter = text_processors.SFileFilter(formatter)
        self.sfile_1 = StringIO(
            " 1 doc1| word1:1 word2:2\n"
            " 1 doc2| word1:1.1 word3:2")

    def test_load_sfile_fwd_1(self):
        token2hash, token_score, num_docs_in = (
            self.sfile_filter._load_sfile_fwd(self.sfile_1))
        self.assertEqual(len(token2hash), 3)
        self.assertEqual(token_score, {'word1': 2.1, 'word2': 2, 'word3': 2})
        self.assertEqual(num_docs_in, {'word1': 2, 'word2': 1, 'word3': 1})

    def test_load_sfile_rev_1(self):
        # No collisions
        token2hash = {'one': 1, 'two': 2}
        hash2token = self.sfile_filter._load_sfile_rev(token2hash)
        benchmark = {1: 'one', 2: 'two'}
        self.assertEqual(hash2token, benchmark)

    def test_load_sfile_rev_2(self):
        # One collision, both '0' and '100' map to 0
        token2hash = {str(i): i for i in range(50)}
        token2hash['100'] = 0
        hash2token = self.sfile_filter._load_sfile_rev(token2hash, seed=1976)
        benchmark = {i: str(i) for i in range(50)}
        benchmark[223414] = '0'
        benchmark[0] = '100'
        self.assertEqual(hash2token, benchmark)

    def test_resolve_collisions(self):
        token2hash = {str(i): random.randint(0, 5) for i in range(10)}
        hash_counts = Counter(token2hash.values())
        hash2token = {
            v: k for k, v in token2hash.iteritems() if hash_counts[v] == 1}
        collisions = set(
            k for k, v in token2hash.iteritems() if hash_counts[v] > 1)
        self.sfile_filter._resolve_collisions(
            collisions, hash_counts, token2hash, hash2token)
        # Check that the dicts are inverses of each other
        token2hash_rev = {v: k for k, v in token2hash.iteritems()}
        self.assertEqual(token2hash_rev, hash2token)

    def tearDown(self):
        self.outfile.close()
