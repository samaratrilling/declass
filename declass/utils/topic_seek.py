import sys
import pandas as pd
import numpy as np
import matplotlib.pylab as plt

from time import time
from gensim import corpora, models
from declass.utils import (
        text_processors, filefilter, streamers, gensim_helpers, common)
from common import lazyprop


class Topics(object):
    """
    Convenience wrapper for for the gensim LDA module. 
    See http://radimrehurek.com/gensim/ for more details.
    """
    def __init__(
        self, text_base_path=None, file_type='*.txt', vw_corpus_path=None,
        shuffle=False, tokenizer=text_processors.TokenizerBasic(), limit=None, 
        verbose=False):
        """
        Parameters
        ----------
        text_base_path : string or None
            Base path to dir containing files.
        file_type : string
            File types to filter by.
        vw_corpus_path : string None
            Path to corpus saved in sparse VW format. 
        shuffle : Boolean
            If True, shuffle paths in base_path
            Not currently supported for vw_corpus_path
        tokenizer : function
        limit : int or None
            Limit for number of docs processed.
        verbose : bool
        
        Notes
        -----
        If text_base_path is None assumes that sparse_corpus_path will be 
        specified. Current supports on VW sparse format.
        """
        self.verbose = verbose
        self.limit = limit
        
        assert (text_base_path is None) or (vw_corpus_path is None)
        if text_base_path:
            self.streamer = streamers.TextFileStreamer(
                    text_base_path=text_base_path, file_type=file_type,
                    tokenizer=tokenizer, limit=limit, shuffle=shuffle)
        if vw_corpus_path:
            self.streamer = streamers.VWStreamer(
                    sfile=vw_corpus_path, limit=limit)

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

    def set_corpus(self, load_path=None, doc_id=None):
        """
        Creates a corpus and sets self.corpus, self.doc_id.  

        Parameters
        ----------
        load_path : String
            If provided, load corpus from load_path.
        doc_id : List of strings
            Limit corpus building to documents with these ids
        """
        t0 = time()

        # If you're loading, set streamer.doc_id_cache now
        # If you're streaming, self.streamer.doc_id_cache will be set when you
        # actually stream.
        if load_path:
            assert doc_id is None, "Can't filter by doc_id with loaded corpus"
            self.corpus = corpora.SvmLightCorpus(load_path)
            self.streamer.__dict__['doc_id_cache'] = (
                common.get_list_from_filerows(load_path + '.doc_id'))
        else:
            self.corpus = gensim_helpers.StreamerCorpus(
                self.dictionary, self.streamer, doc_id=doc_id,
                cache_list=['doc_id'])

        t1 = time()
        build_time = t1-t0
        self._print('corpus build time: %s'%build_time)

    def fit_lda(
        self, num_topics, alpha=None, eta=None, passes=1, chunksize=2000,
        update_every=1, corpus_save_path=None):
        """
        Buld the lda model on the current version of self.corpus.
        Sets self.doc_id equal to a list of doc_id encountered in building
        this corpus.
        
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
        self.doc_id = self.streamer.doc_id_cache
        self._print('lda build time: %s' % build_time)
        self.lda = lda

        if corpus_save_path:
            self.serialize_current_corpus(corpus_save_path)

        return lda

    def serialize_current_corpus(self, corpus_save_path):
        """
        Save the corpus that was set by calling self.set_corpus()
        to svmlight format (with additional .index and .doc_id files).
        """
        if hasattr(self.corpus, 'fname'):
            if self.corpus.fname == corpus_save_path:
                safepath = corpus_save_path + '.safety'
                self.serialize_current_corpus(safepath)
                raise ValueError(
                    "Corpus save path cannot equal corpus_load_path\n"
                    "File saved anyway to %s before exit" % safepath)
        # compact format save of the corpus, index, and doc_id
        corpora.SvmLightCorpus.serialize(corpus_save_path, self.corpus)
        with open(corpus_save_path + '.doc_id', 'w') as f:
            f.write('\n'.join(self.streamer.doc_id_cache))
 
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
        topics_df.index = self.doc_id
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
            plt.savefig(plot_path)

        return words_df
