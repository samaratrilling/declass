"""
Helper objects/functions specifically for use with Gensim.
"""

from . import text_processors


class SimpleCorpus(object):
    """
    A gensim corpus-compatible object for streaming bag of word
    representations from token streams and dictionaries.

    Depending on your method for streaming tokens, this could be slow...
    so it's usually better to serialize this corpus first using 
    gensim.corpora.SvmLightCorpus.serialize(path, my_simple_corpus) 
    before doing modeling.
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
                doc_ids (list of strings)
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
