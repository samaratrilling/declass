import sys
import pandas as pd
import numpy as np
import matplotlib.pylab as plt

from time import time
from gensim import corpora, models
from declass.utils import text_processors, filefilter, gensim_helpers


class Topics(object):
    def __init__(self, text_base_path=None, file_type='*.txt', 
            sparse_corpus_path=None, limit=None, verbose=True):
        """
        A wrapper class for the gensim LDA module. See 
        http://radimrehurek.com/gensim/ for more details.
        
        Parameters
        ----------
        text_base_path : string or None
            Base path to dir containing files.
        file_type : string
            File types to filter by.
        sparse_corpus_path : string None
            Path to corpus saved in sparse format. 
        limit : int or None
            Limit for number of docs processed.
        verbose : bool
        
        Notes
        -----
        If text_base_path is None assumes that sparse_corpus_path will be 
        specified. Current supports on VW sparse format.

        """
        self.verbose = verbose
        self.text_base_path = text_base_path
        self.file_type = file_type
        self.sparse_corpus_path = sparse_corpus_path
        self.limit = limit
        self.dictionary = None
        self.corpora = None
        if text_base_path:
            self.paths = self.get_paths()
        else:
            self.paths = None
        self.doc_ids = self.get_doc_ids()


    def get_paths(self):
        paths = filefilter.get_paths(self.text_base_path, self.file_type, 
                limit=self.limit)
        return paths

    def get_doc_ids(self):
        """
        Gets the doc ids. If self.paths is None, ass
        """
        if self.paths:
            doc_ids = [filefilter.path_to_name(path) for path in self.paths]
        else:
            assert self.sparse_corpus_path, (
            'neither text_base_path or sparse_corpus_path has been specified')
            with open(self.sparse_corpus_path) as f:
                doc_ids = [line.split('|')[0].split(' ')[-1] for line in f]
                doc_ids = doc_ids[:self.limit]
        return doc_ids

    def set_dictionary(self, tokenizer=None, filter_extremes=1, 
            load_path=None, save_path=None):
        """
        Convert token stream into a dictionary.
        
        Parameters
        ----------
        text_base_path : string
            path to dir with text file
        tokenizer : function
            tokenizer function
        limit : int
            limit number of docs to process
        filter_extremes : int
            filter out words of low count
        load_path : string
            path to saved dictionary 
        save_path : string
            path to save dictionary
        vw_corpus_path : string
            path to vw sparse format tokenized text 
       
        Notes
        -----
        If load_path is not provided the function will use self.paths to build
        the dictionary. For more information on vw see 
        https://github.com/JohnLangford/vowpal_wabbit/wiki

        """
        t0 = time()
        if not tokenizer:
            tokenizer = text_processors.TokenizerBasic()
        if load_path:
            dictionary = corpora.Dictionary.load(load_path)
        else:
            if self.paths:
                token_stream = text_processors.TokenStreamer(tokenizer, 
                        paths=self.paths, limit=self.limit)
            else:
                token_stream = text_processors.VWFormatter(
                        ).sfile_to_token_iter(self.sparse_corpus_path)
            dictionary = corpora.Dictionary(token_stream)
        if filter_extremes:
            low_freq_ids = [tokenid for tokenid, docfreq in 
                    dictionary.dfs.iteritems() if docfreq < filter_extremes]
            dictionary.filter_tokens(low_freq_ids)
        dictionary.compactify()
        t1 = time()
        build_time = t1-t0
        self._print('dictionary build time: %s'%build_time)
        if save_path:
            dictionary.save(save_path)
        self.dictionary=dictionary

    def set_corpus(self, tokenizer=None, save_path=None):
        """
        Creates a corpus. 
        
        Parameters
        ----------
        tokenizer : function
        save_path : string
            Path to save corpus to disc. 
        """
        assert self.dictionary, 'dictionary has not been created'
        t0 = time()
        if not tokenizer:
            tokenizer = text_processors.TokenizerBasic()
        if self.paths:
            self.corpus = gensim_helpers.TextFilesCorpus(tokenizer, 
                    self.dictionary, paths=self.paths, limit=self.limit)
        else: 
            self.corpus = SimpleCorpus(self.dictionary, self.sparse_corpus_path, 
                    limit=self.limit)
        t1 = time()
        build_time = t1-t0
        self._print('corpus build time: %s'%build_time)

        if save_path:
            #compact format save
            corpora.SvmLightCorpus.serialize(save_path, self.corpus)

    def build_lda(self, num_topics, alpha=None, eta=None, passes=1, 
            chunksize=2000, update_every=1):
        """
        Buld the lda model.
        
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
        """
        assert self.corpus, 'corpus has not been created'
        self.num_topics = num_topics
        t0 = time()
        lda = models.LdaModel(self.corpus, id2word=self.dictionary, 
                num_topics=num_topics, passes=passes, alpha=alpha, eta=eta, 
                chunksize=chunksize, update_every=update_every)
        t1=time()
        build_time = t1-t0
        self._print('lda build time: %s'%build_time)
        self.lda = lda
        return lda
 
    def write_topics(self, save_path, num_topics=None, num_words=5):
        """
        Prints the topics.
        
        Parameters
        ----------
        save_path : string
            topics file path
        num_topics : int
            number of topics to print
        num_words : int
            number of words to print with each topic
        """
        if not num_topics:
            num_topics = self.num_topics
        with open(save_path, 'w') as f:
            for t in xrange(num_topics):
                f.write('topic %s'%t + '\n')
                f.write(self.lda.print_topic(t, topn=num_words) + '\n')

    def write_doc_topics(self, save_path, sep='|'):
        """
        Creates a pipe separated file with doc_id|topics scores.
        """
        topics_df = self._format_output()
        topics_df.to_csv(save_path, sep=sep, header=False)
    
    def _format_output(self):
        topics_df = pd.concat((pd.Series(dict(doc)) for doc in 
            self.lda[self.corpus]), axis=1).fillna(0).T
        topics_df.index = self.doc_ids
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


class SimpleCorpus(object):
    """
    A simple corpus built with a dictionary and a token stream. 
    """
    def __init__(self, dictionary, sparse_corpus_path, limit):
        self.sparse_corpus_path = sparse_corpus_path
        self.dictionary = dictionary
        self.limit = limit
    
    def __iter__(self):
        """
        This method returns an iterator.
        This method is automatically called when you use MyCorpus in a for 
        loop. The returned value becomes the loop iterator.
        """
        token_streamer = text_processors.VWFormatter().sfile_to_token_iter(
                self.sparse_corpus_path)

        for index, token_list in enumerate(token_streamer):
            if index == self.limit:
                raise StopIteration
            yield self.dictionary.doc2bow(token_list)




