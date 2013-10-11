"""
Wrappers to help with Vowpal Wabbit (VW).
"""
import csv
from collections import defaultdict

import pandas as pd

from declass.utils import common


def parse_varinfo(varinfo_file):
    """
    Uses the output of the vw-varinfo utility to get a DataFrame with variable
    info.

    Parameters
    ----------
    varinfo_file : Path or buffer
        The output of vw-varinfo
    """
    open_file, was_path = common.openfile_wrap(varinfo_file, 'r')

    # For some reason, pandas is confused...so just split the lines
    # Create a dict {item1: [...], item2: [...],...} for each item in the
    # header
    header = open_file.next().split()
    rows = {col_name: [] for col_name in header}
    for line in open_file:
        for i, item in enumerate(line.split()):
            rows[header[i]].append(item)

    if was_path:
        open_file.close()
    
    # Create a data frame
    varinfo = pd.DataFrame(rows)
    # Format columns correctly
    varinfo.FeatureName = varinfo.FeatureName.str.replace('^', '')
    varinfo.HashVal = varinfo.HashVal.astype(int)
    varinfo.MaxVal = varinfo.MaxVal.astype(float)
    varinfo.MinVal = varinfo.MinVal.astype(float)
    varinfo.RelScore = (
        varinfo.RelScore.str.replace('%', '').astype(float) / 100)
    varinfo.Weight = varinfo.Weight.astype(float)

    return varinfo


def parse_lda_topics(topics_file, num_topics):
    """
    Returns a DataFrame representation of the topics output of an lda VW run.

    Parameters
    ----------
    topics_file : filepath or buffer
        The --readable_model output of a VW lda run
    num_topics : Integer
        The number of topics in every valid row

    Notes
    -----
    The trick is dealing with lack of a marker for the information printed
    on top, and the inconsistant delimiter choice...hopefully this (horrible)
    format becomes consistent someday...
    """
    topics = {'topic_%d' % i: [] for i in range(num_topics)}
    topics['HashVal'] = []
    # The topics file contains a bunch of informational printout stuff at
    # the top.  Figure out what line this ends on
    open_file, was_path = common.openfile_wrap(topics_file, 'r')

    # Once we detect that we're in the valid rows, there better not be
    # any exceptions!
    in_valid_rows = False
    for i, line in enumerate(open_file):
        try:
            # If this row raises an exception, then it isn't a valid row
            # Sometimes trailing space...that's the reason for String.split()
            # rather than csv.reader or a direct pandas read.
            split_line = line.split()
            HashVal = int(split_line[0])
            topics['HashVal'].append(HashVal)
            topic_weights = [float(item) for item in split_line[1:]]
            assert len(topic_weights) == num_topics
            for i, weight in enumerate(topic_weights):
                topics['topic_%d' % i].append(weight)
            in_valid_rows = True
        except (ValueError, IndexError, AssertionError):
            if in_valid_rows:
                raise

    if was_path:
        open_file.close()

    topics = pd.DataFrame(topics)

    return topics


def parse_lda_predictions(predictions_file, num_topics):
    """
    Return a DataFrame representation of a VW prediction file.

    Parameters
    ----------
    predictions_file : filepath or buffer
        The -p output of a VW lda run
    num_topics : Integer
        The number of topics you should see
    """
    prediction = {'topic_%d' % i: [] for i in range(num_topics)}
    prediction['doc_id'] = []

    open_file, was_path = common.openfile_wrap(predictions_file, 'r')

    for line in open_file:
        split_line = line.split()
        # Currently only deal with topics + a doc_id
        assert len(split_line) == num_topics + 1
        for i, item in enumerate(split_line):
            if i < num_topics:
                prediction['topic_%d' % i].append(float(item))
            else:
                prediction['doc_id'].append(item)
            
    if was_path:
        open_file.close()

    return pd.DataFrame(prediction)
