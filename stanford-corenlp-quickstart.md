Stanford Core nlp Quick Start
=============================

The Stanford Core NLP library takes in a file or list of files, tokenizes, finds part of speech, lemmatizes, and much more.  Read about it on the [stanford website](http://nlp.stanford.edu/software/corenlp.shtml).  It is good and fast blah blah blah...

It comes with little in the way of setup instructions or a quick-start guide.  Start here then read the rest at the [stanford website][corenlp].


Command line usage
------------------

You can run the tagger on a bunch of files using the command line.  This produces an xml file that can be read with Python.  It isn't interactive, but it's simple, easy, fast, and parallel.

### Setting it up

1. Download the zip file [here](http://nlp.stanford.edu/software/corenlp.shtml#Download)
2. Unzip it anywhere.  The unzipped directory will be the entire installation (binaries, source, etc...), so put it somewhere where you won't lose it.
3. Make sure you have `java`, a `JDK`, and `ant` (sort of like `make` but for Java and using xml) installed.
4. `cd` into the unzipped directory and type `ant`.  This should build everything.
5. Take a look at the text file `input.txt`.  We will do some nlp action on this.
5. From the command line, type:

    ./corenlp.sh -file input.txt

This will analyze this sh$% out of `input.txt` and dump everything into a completely unreadable `input.txt.xml` file.  You can also try:

    ./corenlp.sh -file input.txt -outputFormat text
    ./corenlp.sh -file input.txt -outputFormat text -annotators tokenize,ssplit

See what this does.  At this time you should read through the [stanford website][corenlp].  Note that rather than copying the incomprehensible paths to `jar` files in the command line, you can use the `corenlp.sh` shell script and add command line options to them.

### Parsing the xml with Python

Assuming things worked, you can now analyze huge document collections and get unreadable `xml` output.  Let's use some Python magic to actually work with this...

**more coming soon...**


[corenlp]: http://nlp.stanford.edu/software/corenlp.shtml
