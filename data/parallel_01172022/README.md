### Parallel Data (01/17/2022)

#### Raw Data
Dev/Test (in-domain development and testing sets): 1000 lines

Out_dev/Out_test (out-of-domain development and testing sets): 256 lines

Train (the training set): 16696 lines

#### Moses Tokenized Data

Moses Tokenizer (https://github.com/moses-smt/mosesdecoder) pre-processed data is under the "moses_tokenized" directory:

Dev/Test: 1000 lines

Out_dev/Out_test: 256 lines

Train: 16674 lines

Pre-processing scripts are provided in [tokenize.py](moses_tokenized/tokenize.py).

#### Transliteration

The "transliteration" directory contains transliterated (but untokenized) Cherokee text:

Dev/Test: 1000 lines

Out_dev/Out_test: 256 lines

Train: 16696 lines
