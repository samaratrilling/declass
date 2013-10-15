Working with Vowpal Wabbit (VW)
===============================

To work with the `declass` utilities you need to clone the [declass repo][declassrepo] and put it in your `PYTHONPATH`.  You should then set a shell variable with `export DECLASS=path-to-declass-directory`.

Create the sparse file (sfile)
------------------------------

Assume you have a `base_path` (directory), called `my_base_path`, under which you have all the documents you want to analyze.

### Method 1: `files_to_vw.py`
* Clone the [parallel easy][parallel_easy] module and put it in your `PYTHONPATH`.
* Try converting the first 5 files in `my_base_path`.  The following should print 5 lines of of results, in [vw format][vwinput]

    find my_base_path -type f | head -n 5 | python $DECLASS/cmd/files_to_vw.py


* Convert the entire directory quickly.  The `-o` option is the path to your output file.  The `--n_jobs -2` option means use all cores except for 1.

    python $DECLASS/cmd/files_to_vw.py --base_path my_base_path --n_jobs -2 -o my_output.vw


#### To use a custom tokenizer with Method 1
The default tokenizer removes stopwords and that's it.  You of course will want to create custom tokenizers.  A `Tokenizer` simply needs to be a subclass of `BaseTokenizer`.
* Create a `Tokenizer` and pickle it using `my_tokenizer.save(my_filename.pkl)`
* Pass this path as the `--tokenizer_pickle` option to `files_to_vw.py`
* If you think this tokenizer is useful for everyone, then submit an issue requesting this be added to the standard tokenizers, then it can be called with the `--tokenizer_type` argument.

### Method 2: From a `TextFileStreamer`


Quick test of VW on this `sfile`
--------------------------------

    rm -f *cache
    vw --lda 5 --cache_file my_output.cache --passes 10 -p prediction.dat --readable_model topics.dat -b 20

* The call `vw --lda 5` means run LDA and use 5 topics.  
* The `--cache_file` option means "during the first pass, convert the input to a binary 'cached' format and use that for subsequent results.  The `rm -f *cache` is important since if you don't erase the cache file, `VW` will re-use the old one, even if you specify a new input file!
* The `-b 20` option means: "Use 20 bits of precision" when [hashing][hashing] tokens.

This produces three files...

TODO Explain the files


[vwinput]: https://github.com/JohnLangford/vowpal_wabbit/wiki/Input-format
[declassrepo]: https://github.com/declassengine/declass
[parallel_easy]: https://github.com/langmore/parallel_easy
[hashing]: https://github.com/JohnLangford/vowpal_wabbit/wiki/Feature-Hashing-and-Extraction
