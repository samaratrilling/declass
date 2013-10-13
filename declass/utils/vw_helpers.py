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

    # Rename columns to decent Python names
    varinfo = varinfo.rename(
        columns={'FeatureName': 'feature_name', 'HashVal': 'hash_val',
            'MaxVal': 'max_val', 'MinVal': 'min_val', 'RelScore': 'rel_score',
            'Weight': 'weight'}).set_index('hash_val')

    return varinfo


def parse_lda_topics(topics_file, num_topics, normalize=True):
    """
    Returns a DataFrame representation of the topics output of an lda VW run.

    Parameters
    ----------
    topics_file : filepath or buffer
        The --readable_model output of a VW lda run
    num_topics : Integer
        The number of topics in every valid row
    normalize : Boolean
        Normalize the rows so that they represent probabilities of topic
        given hash_val

    Notes
    -----
    The trick is dealing with lack of a marker for the information printed
    on top, and the inconsistant delimiter choice...hopefully this (horrible)
    format becomes consistent someday...
    """
    topics = {'topic_%d' % i: [] for i in range(num_topics)}
    topics['hash_val'] = []
    # The topics file contains a bunch of informational printout stuff at
    # the top.  Figure out what line this ends on
    open_file, was_path = common.openfile_wrap(topics_file, 'r')

    # Once we detect that we're in the valid rows, there better not be
    # any exceptions!
    in_valid_rows = False
    for line in open_file:
        try:
            # If this row raises an exception, then it isn't a valid row
            # Sometimes trailing space...that's the reason for String.split()
            # rather than csv.reader or a direct pandas read.
            split_line = line.split()
            hash_val = int(split_line[0])
            topic_weights = [float(item) for item in split_line[1:]]
            assert len(topic_weights) == num_topics
            for i, weight in enumerate(topic_weights):
                topics['topic_%d' % i].append(weight)
            topics['hash_val'].append(hash_val)
            in_valid_rows = True
        except (ValueError, IndexError, AssertionError):
            if in_valid_rows:
                raise

    if was_path:
        open_file.close()

    topics = pd.DataFrame(topics).set_index('hash_val')
    if normalize:
        topics = topics.div(topics.sum(axis=1), axis=0)

    return topics


def find_start_line_lda_predictions(predictions_file, num_topics):
    """
    Return the line number (zero indexed) of the start of the last set of
    predictions in predictions_file.

    Parameters
    ----------
    predictions_file : filepath or buffer
        The -p output of a VW lda run
    num_topics : Integer
        The number of topics you should see

    Notes
    -----
    The predictions_file contains repeated predictions...one for every pass.
    We parse out and include only the last predictions by looking for repeats
    of the first lines doc_id field.  We thus, at this time, require the VW 
    formatted file to have, in the last column, a unique doc_id associated
    with the doc.
    """
    open_file, was_path = common.openfile_wrap(predictions_file, 'r')

    for line_num, line in enumerate(open_file):
        split_line = line.split()
        # Currently only deal with topics + a doc_id
        assert len(split_line) == num_topics + 1
        doc_id = split_line[-1]
        if line_num == 0:
            first_doc_id = doc_id
        if doc_id == first_doc_id:
            start_line = line_num

    if was_path:
        open_file.close()

    return start_line


def parse_lda_predictions(
    predictions_file, num_topics, start_line, normalize=True):
    """
    Return a DataFrame representation of a VW prediction file.

    Parameters
    ----------
    predictions_file : filepath or buffer
        The -p output of a VW lda run
    num_topics : Integer
        The number of topics you should see
    start_line : Integer
        Start reading the predictions file here.
        The predictions file contains repeated predictions, one for every pass.
        You generally do not want every prediction.
    normalize : Boolean
        Normalize the rows so that they represent probabilities of topic
        given doc_id.
    """
    predictions = {'topic_%d' % i: [] for i in range(num_topics)}
    predictions['doc_id'] = []

    open_file, was_path = common.openfile_wrap(predictions_file, 'r')

    for line_num, line in enumerate(open_file):
        if line_num < start_line:
            continue
        split_line = line.split()
        for item_num, item in enumerate(split_line):
            if item_num < num_topics:
                predictions['topic_%d' % item_num].append(float(item))
            else:
                predictions['doc_id'].append(item)

    if was_path:
        open_file.close()

    predictions = pd.DataFrame(predictions).set_index('doc_id')
    if normalize:
        predictions = predictions.div(predictions.sum(axis=1), axis=0)

    return predictions


class LDAResults(object):
    """
    Facilitates working with results of VW lda runs.

    See http://hunch.net/~vw/  as a starting place for VW information.  

    See https://github.com/JohnLangford/vowpal_wabbit/wiki/lda.pdf
    for a brief tutorial of lda in VW.

    The tutorials above and documentation is far from all-inclusive.  
    More detail can be found by searching through the yahoo group mailing list:  
    http://tech.groups.yahoo.com/group/vowpal_wabbit/
    """
    def __init__(
        self, varinfo_file, topics_file, predictions_file, num_topics):
        """
        Parameters
        ----------
        varinfo_file : Path or buffer
            The output of vw-varinfo
        topics_file : filepath or buffer
            The --readable_model output of a VW lda run
        predictions_file : filepath or buffer
            The -p output of a VW lda run
        num_topics : Integer
            The number of topics in every valid row
        """
        self.num_topics = num_topics

        self.varinfo = parse_varinfo(varinfo_file)

        self.topics = parse_lda_topics(topics_file, num_topics)
        self.topics = self.topics.reindex(index=self.varinfo.index)

        start_line = find_start_line_lda_predictions(
            predictions_file, num_topics)
        self.predictions = parse_lda_predictions(
            predictions_file, num_topics, start_line)