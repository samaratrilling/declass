import sys
import pandas as pd
import numpy as np
import matplotlib.pylab as plt

from time import time
from gensim import corpora, models
from declass.utils import (
        text_processors, filefilter, streamers, gensim_helpers, common)
from common import lazyprop

from text_processors import TokenizerBasic


class Topics(object):
    """
    Convenience wrapper for for the gensim LDA module. 
    See http://radimrehurek.com/gensim/ for more details.
    """
    def __init__(
        self, text_base_path=None, limit=None, file_type='*.txt',
        shuffle=True, tokenizer_func=TokenizerBasic().text_to_token_list,
        verbose=False):
        """
        Parameters
        ----------
        text_base_path : string or None
            Base path to dir containing files.  Used as the default source
            for dictionaries and corpus if these are not specified.
        limit : int or None
            Limit files read in text_base_path to this many.
        file_type : string
            File types to filter by.
        shuffle : Boolean
            If True, shuffle paths in base_path
        tokenizer_func : function
            Converts text strings to lists of words.  Used to create dictionary
            and corpus.
        verbose : bool
        """
        self.verbose = verbose
        
        if text_base_path:
            self.streamer = streamers.TextFileStreamer(
                    text_base_path=text_base_path, file_type=file_type,
                    tokenizer_func=tokenizer_func, limit=limit,
                    shuffle=shuffle)

    def set_dictionary(
        self, doc_id=None, load_path=None, no_below=5, no_above=0.5,
        save_path=None):
        """
        Convert token stream into a dictionary, setting self.dictionary.
        
        Parameters
        ----------
        doc_id : List of doc_id
            Only use documents with these ids to build the dictionary
        load_path : string
            path to saved dictionary 
        no_below : Integer
            Do not keep words with total count below no_below
        no_above : Real number in [0, 1]
            Do not keep words whose total count is more than no_above fraction
            of the total word count.
        save_path : string
            path to save dictionary
        """
        t0 = time()
        # Either load a pre-made dict, or create a new one using __init__
        # parameters.
        if load_path:
            dictionary = corpora.Dictionary.load(load_path)
        else:
            token_stream = self.streamer.token_stream(doc_id=doc_id)
            dictionary = corpora.Dictionary(token_stream)
        dictionary.filter_extremes(no_below=no_below, no_above=no_above)
        dictionary.compactify()

        self._print('dictionary build time: %.2f' % (time() - t0))

        if save_path:
            dictionary.save(save_path)

        self.dictionary = dictionary

    def set_corpus(self, load_path=None, serialize_path=None, doc_id=None):
        """
        Creates a corpus and sets self.corpus

        Parameters
        ----------
        load_path : String
            Load an SvmLightPlusCorpus from here.
        serialize_path : String
            Create an SvmLightPlusCorpus using self.streamer and
            self.dictionary, then save it here.
        doc_id : List of strings
            Limit corpus building to documents with these ids
        """
        t0 = time()
        # Enforce one and only one of load_path, serialize_path
        load_nosave = (load_path is not None) and (serialize_path is None)
        noload_save = (load_path is None) and (serialize_path is not None)
        assert load_nosave or noload_save, (
            "Provide one and only one of load_path, serialize_path")

        # If you're loading, set streamer.doc_id_cache now
        # If you're streaming, self.streamer.doc_id_cache will be set when you
        # actually stream.
        if load_path:
            assert doc_id is None, "Can't filter by doc_id with loaded corpus"
            self.corpus = gensim_helpers.SvmLightPlusCorpus(
                load_path, doc_id=doc_id)
        else:
            self.corpus = gensim_helpers.StreamerCorpus.from_streamer_dict(
                self.streamer, self.dictionary, save_path, doc_id=doc_id)

        t1 = time()
        build_time = t1-t0
        self._print('corpus build time: %s'%build_time)

    def fit_lda(
        self, num_topics, alpha=None, eta=None, passes=1, chunksize=2000,
        update_every=1, corpus_save_path=None):
        """
        Buld the lda model on the current version of self.corpus.
        
        Parameters
        ----------
        num_topics : int
            number of topics 
        alpha : list of floats, None
            hyperparameter vector for topic distribution
        eta : list of floats, None
            hyperparameter vector for word distribution
        passes : int
            number of passes for model build
        chunksize : int
        update_every ; int
        corpus_save_path : string
            Path to save corpus used for this fit to disc in svmlight format.

        Notes
        -----
        If your are using a non-serialized corpus, then this may run slower.
        You can serialize using self.serialize_current_corpus(save_path)
        and then call self.set_corpus(save_path)
        """
        self.num_topics = num_topics
        t0 = time()
        lda = models.LdaModel(self.corpus, id2word=self.dictionary, 
                num_topics=num_topics, passes=passes, alpha=alpha, eta=eta, 
                chunksize=chunksize, update_every=update_every)
        t1=time()
        build_time = t1-t0
        self._print('lda build time: %s' % build_time)
        self.lda = lda

        if corpus_save_path:
            self.serialize_current_corpus(corpus_save_path)

        return lda

    def write_topics(self, path=None, num_words=5):
        """
        Writes the topics to disk.
        
        Parameters
        ----------
        path : string
            Designates file to write to.  If None, write to stdout.
        num_words : int
            number of words to write with each topic
        """
        outfile = common.get_outfile(path)
        for t in xrange(self.num_topics):
            outfile.write('topic %s'%t + '\n')
            outfile.write(self.lda.print_topic(t, topn=num_words) + '\n')
        common.close_outfile(outfile)

    def write_doc_topics(self, save_path, sep='|'):
        """
        Creates a delimited file with doc_id and topics scores.
        """
        topics_df = self._get_topics_df()
        topics_df.to_csv(save_path, sep=sep, header=True)
    
    def _get_topics_df(self):
        topics_df = pd.concat((pd.Series(dict(doc)) for doc in 
            self.lda[self.corpus]), axis=1).fillna(0).T
        topics_df.index = self.corpus.doc_id
        topics_df.index.name = 'doc_id'
        topics_df = topics_df.rename(
            columns={i: 'topic_' + str(i) for i in topics_df.columns})

        return topics_df
            
    def _print(self, msg):
        if self.verbose:
            sys.stdout.write(msg + '\n')

    def get_words_docfreq(self, plot_path=None):
        """
        Returns a df with token id, doc freq as columns and words as index.
        """
        id2token = dict(self.dictionary.items())
        words_df = pd.DataFrame(
                {id2token[tokenid]: [tokenid, docfreq] 
                 for tokenid, docfreq in self.dictionary.dfs.iteritems()},
                index=['tokenid', 'docfreq']).T
        words_df = words_df.sort_index(by='docfreq', ascending=False)
        if plot_path:
            plt.figure()
            words_df.docfreq.apply(np.log10).hist(bins=200)
            plt.xlabel('log10(docfreq)')
            plt.ylabel('Count')
            plt.savefig(plot_path)

        return words_df
