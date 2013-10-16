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

    python $DECLASS/cmd/files_to_vw.py --base_path my_base_path --n_jobs -2 -o doc_tokens.vw


#### To use a custom tokenizer with Method 1
The default tokenizer removes stopwords and that's it.  You of course will want to create custom tokenizers.  A `Tokenizer` simply needs to be a subclass of `BaseTokenizer`.  In particular, it (only) needs to have a method, `.text_to_token_list()` that takes in a string (representing a single document) and spits out a list of strings (the *tokens*).  If you already have such a function, then you can create a tokenizer by doing:

    my_tokenizer = MakeTokenizer(my_tokenizing_func)

In any case, the steps are:

* Create a `Tokenizer` and pickle it using `my_tokenizer.save(my_filename.pkl)`.  Note that an subclass of `BaseTokenizer` automatically has a `.save` method.
* Pass this path as the `--tokenizer_pickle` option to `files_to_vw.py`
* If you think this tokenizer is useful for everyone, then submit an issue requesting this be added to the standard tokenizers, then it can be called with the `--tokenizer_type` argument.

### Method 2: From a `TextFileStreamer`


Quick test of VW on this `sfile`
--------------------------------

    rm -f *cache
    vw --lda 5 --cache_file doc_tokens.cache --passes 10 -p prediction.dat --readable_model topics.dat --bit_precision 16 doc_tokens.vw

* The call `vw --lda 5` means run LDA and use 5 topics.
* The `--cache_file` option means "during the first pass, convert the input to a binary 'cached' format and use that for subsequent results.  The `rm -f *cache` is important since if you don't erase the cache file, `VW` will re-use the old one, even if you specify a new input file!
* The `--bit_precision 16` option means: "Use 16 bits of precision" when [hashing][hashing] tokens.  This will cause many collisions but won't effect the results much at all.  Vowpal Wabbit is *very* sensitive to bit precision!  If you make it bigger, you need many more passes to get decent results.
* See [this slideshow][vwlda] about LDA in VW.

This produces two files:

* `prediction.dat`.  Each row is one document.  The last column is the `doc_id`, the first columns are the (un-normalized) topics weights.  Dividing each row by the row sum, each row would be `P[topic | document]`.  Note that a new row is printed for *every* pass.  So if you run with `--passes 10`, there total number of rows will be 10 times the number of documents.
* `topics.dat`.  Each row is a word.  The first column is the hash value for that word.  The columns are, after normalization, `P[word | topic]`.  Note that the hash values run from 0 to `2^bit_precision`.  So even if the word corresponding to hash value 42 never appears in your documents, it will appear in this output file (probably with a complete garbage value).


Working with an `SFileFilter` and `LDAResults`
----------------------------------------------

There are some issues with using the raw `prediction.dat` and `topics.dat` files.  For one, the word hash values are not very interpretable--you want to work with actual English words.  Moreover, unless you allow for a very large hash space, you will have collisions.  Second, you will want some quick means to drop words from the vocabulary, or drop documents from the corpus without having to regenerate the VW file.  And finally, they need to be loaded into some suitable data structure for analysis.

### Step 1:  Make an `SFileFilter`

```python
sfile_filter = SFileFilter(text_processors.VWFormatter(), verbose=True)
sfile_filter.load_sfile('doc_tokens.vw')

sfile_filter.remove_extreme_tokens(doc_freq_min=5, doc_fraction_max=0.5)
sfile_filter.resolve_collisions()
sfile_filter.save('sfile_filter_file.pkl')
```

* `.remove_extreme_tokens` removes low/high frequency tokens from our filter's internal dictionaries.  It's just like those words were never present in the original text.
* `.resolve_collisions()` changes the hash values for tokens that collide.  If you have too many collisions an exception will be raised.  Note that if we didn't remove extreme tokens before resolving collisions, then we would have many words in our vocab, and there is a good chance the collisions would not be able to be resolved!

### Step 2a:  Run VW on filtered output
First save a "filtered" version of `doc_tokens.vw`.

```python
sfile_filter.filter_sfile('doc_tokens.vw', 'doc_tokens_filtered.vw')
```
Our filtered output, `doc_tokens_filtered.vw` has replaced words with the hash values that the `sfile_filter` chose.  This forces `vw` to use the hash values we chose.  Since we saved our filter with `.save`, we will have access to the `token2hash` mapping.  Also, since the `sfile_filter`resolves all collisions, we also have access to the inverse mapping `hash2token`.

Now run `vw`.

```
rm -f *cache
vw --lda 5 --cache_file ddrs.cache --passes 10 -p prediction.dat --readable_model topics.dat --bit_precision 16 doc_tokens_filtered.vw
```
It is very important that the bit precision for vw, set with `--bit_precision 16` matches the bit precision used for your `SFileFilter` (default = 16).  If you don't then the hash values used by `vw` will not match the ones stored in the filter.


### Step 2b:  Filter "on the fly" using a saved `sfile_filter`
The workflow in step 2a requires making the intermediate file `doc_tokens_filtered.vw`.  Keeping track of all these filtered outputs is an issue.  Since you already need to keep track of a saved sfile_filter, you might as well use that as your [one and only one][spot] reference.

```
rm -f *cache
python $DECLASS/cmd/filter_sfile.py -s sfile_filter_file.pkl  doc_tokens.vw  \
    | vw --lda 5 --cache_file ddrs.cache --passes 10 -p prediction.dat --readable_model topics.dat --bit_precision 16
```
The python function `filter_sfile.py` takes in `ddrs.vw` and streams a filtered sfile to stdout.  The `|` connects `vw` to this stream.  Notice we no longer specify an input file to `vw` (previously we passed it a `doc_tokens_filtered.vw` positional argument).

### Step 3:  Read the results with `LDAResults`

```python
num_topics = 5
lda = LDAResults('topics.dat', 'prediction.dat', num_topics, 'sfile_filter_file.pkl')
lda.topics.head()
lda.predictions.head()
lda.print_topics()
```


[vwinput]: https://github.com/JohnLangford/vowpal_wabbit/wiki/Input-format
[declassrepo]: https://github.com/declassengine/declass
[parallel_easy]: https://github.com/langmore/parallel_easy
[vwlda]: https://github.com/JohnLangford/vowpal_wabbit/wiki/lda.pdf
[hashing]: https://github.com/JohnLangford/vowpal_wabbit/wiki/Feature-Hashing-and-Extraction
[spot]: http://en.wikipedia.org/wiki/Single_Point_of_Truth
