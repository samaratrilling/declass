"""
Helper objects/functions specifically for use with Gensim.
"""

from . import text_processors


class TextFilesCorpus(object):
    """
    A gensim corpus-compatible object for streaming bag of word representations
    from a directory structure containing text files.

    This means we read text files one by one, convert to bag-of-words using
    a dictionary, and then return the bag of words.  If you are repeatedly
    reading, this is slow...so it's better to serialize this using
    gensim.corpora.SvmLightCorpus.serialize() 
    before doing any analytics.
    """
    def __init__(
        self, tokenizer, dictionary, base_path=None, file_type='*',
        paths=None, limit=None):
        """
        Stream token lists from pre-defined path lists.

        Parameters
        ----------
        tokenizer : text_processors.Tokenizer object
        dictionary : gensim.corpora.Dictionary object
        base_path : String
            The base directory.  Get all files of file_type within this.
        file_type : String
            Glob filter on the file, e.g. '*.txt'
        paths : Iterable
            E.g. a list of paths.
        limit : Integer
            Raise StopIteration after returning limit token lists.
        """
        self.tokenizer = tokenizer
        self.dictionary = dictionary
        self.base_path = base_path
        self.file_type = file_type
        self.paths = paths
        self.limit = limit

        self.index = 0
        self.token_streamer = text_processors.TokenStreamer(
            self.tokenizer, base_path=self.base_path, file_type=self.file_type,
            paths=self.paths)

    def __iter__(self):
        # Defined for loops, which expect an iterable, not iterator
        return self

    def next(self):
        """
        x.next() -> the next value, or raise StopIteration
        """
        while True:
            if self.index == self.limit:
                raise StopIteration
            token_list = self.token_streamer.next()
            self.index += 1

            return self.dictionary.doc2bow(token_list)
