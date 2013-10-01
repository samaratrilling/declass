"""
Classes for streaming tokens/info from files/sparse files etc...
"""
from . import filefilter, nlp, common, text_processors
from common import lazyprop


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
    For streaming from a single VW file.
    """
    def __init__(self, sfile=None):
        """
        Parameters
        ----------
        sfile : File path or buffer
            Points to a sparse (VW) formatted file.
        """
        self.sfile = sfile

        self.formatter = text_processors.VWFormatter()

    def info_stream(self, limit=None):
        """
        Returns an iterator over info dicts.
        """
        # Open file if path.  If buffer or StringIO, passthrough.
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
    def __init__(self, text_base_path=None, file_type='*.txt'):
        self.text_base_path = text_base_path
        self.file_type = file_type
    
    @lazyprop
    def paths(self):
        """
        Get all paths that we will use.
        """
        if self.text_base_path:
            paths = filefilter.get_paths(
                self.text_base_path, self.file_type, limit=self.limit)
        else:
            paths = None

        return paths

    @lazyprop
    def doc_ids(self):
        """
        Get doc_ids corresponding to all paths.
        """
        regex = re.compile(self.name_strip)
        doc_ids = [
            regex.sub('', path_to_name(p, strip_ext=False))
            for p in self.paths]

        return doc_ids

    @lazyprop
    def _doc_id_to_path(self):
        """
        Build the dictionary mapping doc_id to path.  doc_id is based on
        the filename.
        """
        return dict(zip(self.doc_ids, self.paths))

    def info_stream(self, limit=None):
        """
        Returns an iterator over paths returning token lists.
        """

        for index, onepath in enumerate(paths):
            if index == self.limit:
                raise StopIteration

            with open(onepath, 'r') as f:
                text = f.read()
                doc_id = filefilter.path_to_name(onepath)
                yield {'item': text, 'info': {'path': onepath, 
                    'doc_id': doc_id}}            

        



