#!/usr/bin/env python
from optparse import OptionParser
import sys
import csv
# Set the limit to 1 billion columns
#csv.field_size_limit(10000000)

import jrl_utils.src.common as common
from jrl_utils.src.common import BadDataError


def main():
    r"""
    Generic example of a filter.
    Reads a csv file[s] or stdin, changes something, prints to stdout

    Examples
    ---------
    Read a comma delimited csv file, data.csv, 
    $ python generic_filter.py  data.csv

    Use cat to pipe through stdin, redirect to file.csv
    $ cat data.csv | generic_filter.py >  file.csv
    """
    usage = "usage: %prog [options] dataset"
    usage += '\n'+main.__doc__
    parser = OptionParser(usage=usage)
    parser.add_option(
        "-d", "--delimiter",
        help="Use DELIMITER as the column delimiter.  [default: %default]",
        action="store", dest='delimiter', default=',')
    parser.add_option(
        "-o", "--outfilename",
        help="Write to this file rather than stdout.  [default: %default]",
        action="store", dest='outfilename', default=None)

    (options, args) = parser.parse_args()

    ### Parse args
    # If an argument is given, then it is the 'infilename'
    # If no arguments are given, set infilename equal to None
    infilenames = args if args else [None]

    # Loop through all files and print
    for i, filename in enumerate(infilenames):

        # First time overwrite, after that append
        outmode = 'w' if i == 0 else 'a'

        # Get the infile/outfile
        infile, outfile = common.get_inout_files(
            filename, options.outfilename, outmode=outmode)

        ## Call the function that does the real work
        try:
            generic_filter(infile, outfile, delimiter=options.delimiter)
        except BadDataError as e:
            sys.stderr.write(e.message)

        ## Close the files iff not stdin, stdout
        common.close_files(infile, outfile)


def generic_filter(infile, outfile, delimiter=','):
    """
    Reads infile, prints to outfile.  Modify as needed.

    Parameters
    ----------
    infile : Open file
    outfile : Open file to write to
    delimiter : Delimiter for both in/outfiles
    """
    ## Get the csv reader and writer.  Use these to read/write the files.
    # reader.fieldnames gives you the header
    #reader = csv.DictReader(infile, delimiter=delimiter, fieldnames=header)
    #writer = csv.DictWriter(outfile, delimiter=delimiter, fieldnames=outheader)

    ## Use the standard reader/writer
    reader = csv.reader(infile, delimiter=delimiter)
    writer = csv.writer(outfile, delimiter=delimiter)
    # header = reader.next()

    ## Iterate through the file, printing out lines 
    for row in reader:
        _modify_row(row)
        writer.writerow(row)


def _modify_row(row):
    """
    Modify the row in place.  Currently does nothing but you can change that.

    Parameters
    ----------
    row : Dict or list (depending on the kind of reader we used)
    """
    # Do nothing since this is a stub
    pass


def _popheader(infile, delimiter):
    """
    Returns first row from infile, increments pointer.
    """
    ### Get the header from the first file, get outheader, write outheader.
    headerstr = infile.readline().strip()
    header = headerstr.split(delimiter)

    return header


if __name__=='__main__':
    main()


