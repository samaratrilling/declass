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
        cache_list : List of strings
            Cache these items on every iteration
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
        limit : Integer
            Limit returned results to this number
        cache_list : Cache these items as they appear
            Call self.token_stream('doc_id', 'tokens') to cache
            info['doc_id'] and info['tokens'] (assuming both are available).
        """
        return self.single_stream('tokens', limit=limit, cache_list=cache_list)


class VWStreamer(BaseStreamer):
    """
    For streaming from a single VW file.  Since the VW file format does not
    preserve token order, all tokens are unordered.
    """
    def __init__(self, sfile=None, cache_sfile=False):
        """
        Parameters
        ----------
        sfile : File path or buffer
            Points to a sparse (VW) formatted file.
        cache_sfile : Boolean
            If True, cache the sfile in memory.  CAREFUL!!!
        """
        self.sfile = sfile
        self.cache_sfile = cache_sfile

        self.formatter = text_processors.VWFormatter()
        
        if cache_sfile:
            self.source = self._cached_stream
            self._init_cached_stream()
        else:
            self.source = self._sfile_stream

    def _init_cached_stream(self):
        records = {}
        for record_dict in self._sfile_stream():
            doc_id = record_dict['doc_id']
            records[doc_id] = record_dict

        self.records = records

    def _cached_stream(self, doc_ids=None, limit=None):
        records = self.records

        if doc_ids is None:
            for i, (doc_id, record_dict) in enumerate(records.iteritems()):
                if i == limit:
                    raise StopIteration
                yield record_dict
        else:
            if (limit is not None) and self.cache_sfile:
                raise ValueError(
                    "Cannot use both limit and doc_ids with cached stream")
            for doc_id in doc_ids:
                yield records[doc_id]

    def _sfile_stream(self, doc_ids=None, limit=None):
        """
        Stream record_dict from an sfile that sits on disk.
        """
        # Open file if path.  If buffer or StringIO, passthrough.
        infile, was_path = common.openfile_wrap(self.sfile, 'rb')

        if doc_ids is not None:
            doc_ids = set(doc_ids)

        for i, line in enumerate(infile):
            if i == limit:
                raise StopIteration
            
            record_dict = self.formatter.sstr_to_dict(line) 
            if doc_ids is not None:
                if record_dict['doc_id'] not in doc_ids:
                    continue
            yield record_dict

        if was_path:
            infile.close()

    def info_stream(self, doc_ids=None, limit=None):
        """
        Returns an iterator over info dicts.

        Parameters
        ----------
        doc_ids : Iterable over Strings
            Return info dicts iff doc_id in doc_ids
        limit : Integer
            Only return this many results
        """
        source = self.source(doc_ids=doc_ids, limit=limit)

        # Read record_dict and convert to info by adding tokens
        for record_dict in source:
            record_dict['tokens'] = self.formatter._dict_to_tokens(record_dict)

            yield record_dict


class TextFileStreamer(BaseStreamer):
    """
    For streaming from text files.
    """
    pass
