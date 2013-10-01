"""
Classes for streaming tokens/info from files/sparse files etc...
"""
from . import filefilter, nlp, common, text_processors

class BaseStreamer(object):
    """
    Base class...don't use this directly.
    """
    def single_stream(self, item, limit=None, cache_list=[]):
        """
        Stream a single item from source.

        Parameters
        ----------
        item : String
            The single item to pull from info and stream.
        limit : Integer
            Limit returned results to this number
        cache_list : Cache these items as they appear
            Call self.token_stream('doc_id', 'tokens') to cache
            info['doc_id'] and info['tokens'] (assuming both are available).
        """
        # Initialize the cached items as attributes
        for cache_item in cache_list:
            self.__dict__[cache_item] = []

        # Iterate through self.info_stream and pull off required information.
        stream = self.info_stream(limit=limit)
        for i, info in enumerate(stream):
            if i == limit:
                raise StopIteration
            for cache_item in cache_list:
                self.__dict__[cache_item].append(info[cache_item])

            yield info[item]

    def token_stream(self, limit=None, cache_list=[]):
        """
        Returns an iterator over tokens with possible caching of other info.

        Parameters
        ----------
        item : String
            The single item to pull from info and stream.
        limit : Integer
            Limit returned results to this number
        cache_list : Cache these items as they appear
            Call self.token_stream('doc_id', 'tokens') to cache
            info['doc_id'] and info['tokens'] (assuming both are available).
        """
        return self.single_stream('tokens', limit=limit, cache_list=cache_list)


class VWStreamer(BaseStreamer):
    """
    For streaming from a single VW file.
    """
    def __init__(self, sfile=None):
        """
        Parameters
        ----------
        sfile : File path or buffer
            Points to a sparse (VW) formatted file.
        """
        self.formatter = text_processors.VWFormatter()
        self.sfile = sfile

    def info_stream(self, limit=None):
        """
        Returns an iterator over info dicts.
        """
        infile, was_path = common.openfile_wrap(self.sfile, 'rb')

        for i, line in enumerate(infile):
            if i == limit:
                raise StopIteration
            
            yield self.formatter.sstr_to_info(line)

        if was_path:
            infile.close()


class TextFileStreamer(BaseStreamer):
    """
    For streaming from text files.
    """
    pass
