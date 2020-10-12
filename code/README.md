# ChrEn (EMNLP 2020) -- Code

## Structure

```
|__ OpenNMT-py/  --> modified OpenNMT-py (added BERT, etc.)
|__ scripts/  --> scripts for data preprocessing, SMT/NMT model training & evaluation
```

## Requirements

* Python 3.6
* requirements.txt
* [mosesdecoder](https://github.com/moses-smt/mosesdecoder)

## Quick Start

#### Installation
```
pip install -r requirements.txt
cd OpenNMT-py; python setup.py install
```

#### Parallel data preprocessing:
```
export DATA=../data/parallel
export MOSESDECODER=ABSOLUTE_PATH_TO_MOSESDECODER
python scripts/preprocess.py
```

#### SMT

```
python scripts/SMT.py --model enchr --output_dir enchr   # in-domain En-Chr SMT
python scripts/SMT.py --model chren --output_dir chren   # in-domain Chr-En SMT
python scripts/SMT.py --model enchr --output_dir enchr --dev_file out_dev --test_file out_test   # out-of-domain En-Chr SMT
python scripts/SMT.py --model chren --output_dir chren --dev_file out_dev --test_file out_test   # out-of-domain Chr-En SMT
```

#### NMT

prepare in-domain and out-of-domain En-Chr and Chr-En experimental data; vocabulary with minimun frequency = 0, 5, 10
```
python scripts/NMT/prepare.py --mode prepare --min_freq 0  
python scripts/NMT/prepare.py --mode prepare --min_freq 5 
python scripts/NMT/prepare.py --mode prepare --min_freq 10 
```

train in-domain and out-of-domain RNN-NMT models
```
# En-Chr 
python scripts/NMT/rnn_nmt_enchr.py --dev_file dev  --test_file test
python scripts/NMT/rnn_nmt_enchr.py --dev_file out_dev  --test_file out_test
# Chr-En
python scripts/NMT/rnn_nmt_chren.py --dev_file dev  --test_file test
python scripts/NMT/rnn_nmt_chren.py --dev_file out_dev  --test_file out_test
```


#### Todos
More scripts for BPE, BERT, Semi-supervised learning, Transfer learning, and Mulitlingual training will be added soon.