"""
Helper objects/functions specifically for use with Gensim.
"""
import pandas as pd
from gensim import corpora, models

from . import common


class StreamerCorpus(object):
    """
    Streams (filtered) bag-of-words from token streams and dictionaries.

    Depending on your method for streaming tokens, this could be slow...
    Before modeling, it's usually better to serialize this corpus using: 

    self.to_corpus_plus(fname)
    or
    gensim.corpora.SvmLightCorpus.serialize(path, self) 
    """
    def __init__(self, dictionary, streamer, **streamer_kwargs):
        """
        Stream token lists from pre-defined path lists.

        Parameters
        ----------
        dictionary : gensim.corpora.Dictionary object
        streamer : Streamer compatible object.
            Method streamer.token_stream() returns a stream of lists of words.
        streamer_kwargs : Additional keyword args
            Passed to streamer.token_stream(), e.g.
                limit (int), cache_list (list of strings), 
                doc_id (list of strings)
        """
        self.streamer = streamer
        self.dictionary = dictionary
        self.streamer_kwargs = streamer_kwargs

    def __iter__(self):
        """
        Returns an iterator of "corpus type" over text files.
        """
        for token_list in self.streamer.token_stream(**self.streamer_kwargs):
            yield self.dictionary.doc2bow(token_list)

    def serialize(self, fname):
        """
        Save to svmlight (plus) format (with additional .doc_id file).
        """
        # See if you are limiting doc_id to some list
        if 'doc_id' in self.streamer_kwargs:
            doc_id = self.streamer_kwargs['doc_id']
        else:
            doc_id = self.streamer.doc_id

        # Make the corpus and .index file
        corpora.SvmLightCorpus.serialize(fname, self)

        # Make the .doc_id file
        with open(fname + '.doc_id', 'w') as f:
            f.write('\n'.join(doc_id))


class ToEraseCorpus(object):
    """
    Streams (filtered) bag-of-words from the triplet
    .svmlight, .svmlight.index, and .svmlight.doc_id files.
    """
    def __init__(self, fname, doc_id=None, limit=None):
        """
        Parameters
        ----------
        fname : Path
            Contains the .svmlight bag-of-words text file
        doc_id : Iterable over strings
            Limit all streaming results to docs with id in this list
        limit : Integer
            Limit all streaming results to this many
        """
        self.fname = fname
        self.limit = limit

        self.corpus = corpora.SvmLightCorpus(fname)
        
        # All possible doc_id in the corpus
        self.doc_id_all = common.get_list_from_filerows(
            fname + '.doc_id')

        # Limit all streaming results to docs in self.doc_id
        if doc_id is not None:
            self.doc_id = set(doc_id)
        else:
            self.doc_id = set(self.doc_id_all)

    def __iter__(self):
        """
        Returns a gensim-compatible corpus.

        Parameters
        ----------
        doc_id : Iterable over Strings
            Return info dicts iff doc_id in doc_id
        """
        for i, row in enumerate(self.corpus):
            if i == self.limit:
                raise StopIteration

            if self.corpus_doc_id[i] in self.doc_id:
                yield row


class SvmLightPlusCorpus(corpora.SvmLightCorpus):
    """
    Streams (filtered) bag-of-words from the triplet
    .svmlight, .svmlight.index, and .svmlight.doc_id files.
    """
    def __init__(self, fname, doc_id=None, limit=None):
        """
        Parameters
        ----------
        fname : Path
            Contains the .svmlight bag-of-words text file
        doc_id : Iterable over strings
            Limit all streaming results to docs with these doc_ids
        limit : Integer
            Limit all streaming results to this many
        """
        super(TestCorpus, self).__init__(fname)

        self.limit = limit
        
        # All possible doc_id in the corpus
        self.corpus_doc_id = common.get_list_from_filerows(
            fname + '.doc_id')

        # Limit all streaming results to docs in self.doc_id
        if doc_id is not None:
            self.doc_id = set(doc_id)
        else:
            self.doc_id = set(self.corpus_doc_id)

    def __iter__(self):
        """
        Returns a gensim-compatible corpus.

        Parameters
        ----------
        doc_id : Iterable over Strings
            Return info dicts iff doc_id in doc_id
        """
        base_iterable = super(TestCorpus, self).__iter__()
        for i, row in enumerate(base_iterable):
            if i == self.limit:
                raise StopIteration

            if self.corpus_doc_id[i] in self.doc_id:
                yield row


def get_words_docfreq(self):
    """
    Returns a df with token id, doc freq as columns and words as index.
    """
    id2token = dict(self.dictionary.items())
    words_df = pd.DataFrame(
            {id2token[tokenid]: [tokenid, docfreq] 
             for tokenid, docfreq in dictionary.dfs.iteritems()},
            index=['tokenid', 'docfreq']).T
    words_df = words_df.sort_index(by='docfreq', ascending=False)

    return words_df


def get_doc_topics(corpus, lda, doc_id=None):
    """
    Creates a delimited file with doc_id and topics scores.
    """
    topics_df = pd.concat((pd.Series(dict(doc)) for doc in 
        lda[corpus]), axis=1).fillna(0).T
    topics_df = topics_df.rename(
        columns={i: 'topic_' + str(i) for i in topics_df.columns})

    if doc_id is not None:
        topics_df.index = self.doc_id
        topics_df.index.name = 'doc_id'

    return topics_df
