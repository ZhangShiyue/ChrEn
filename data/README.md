# ChrEn (EMNLP 2020) -- Data

## Structure

```
|__ raw/ --> raw data, including data sources, dialect, etc.
     |__ parallel_data.xlsx  --> parallel data
     |__ monolingual_data.xlsx --> monolingual data
|__ parallel/ -->  parallel data splits
     |__ train.chr, train.en  --> training set
     |__ dev.chr, dev.en --> in-domain development set
     |__ test.chr, test.en --> in-domain testing set
     |__ out_dev.chr, out_dev.en --> out-of-domain development set
     |__ out_test.chr, out_test.en --> out-of-domain testing set
|__ monolingual/  --> Cherokee and English monolingual data
     |__ chr  --> Cherokee monolingual data (from monolingual_data.xlsx)
     |__ en5000, en10000, en20000, en50000, en100000  --> English monolingual data (from News Crawl 2017)
|__ cherokee_old_testament/ --> newly added parallel data from Cherokee Old Testament (01/11/2021, pending to be verified by Cherokee speaker)
     |__ Old_Testament.xlsx 
     |__ Old_Testament.chr
     |__ Old_Testament.en
```

## Disclaimer

The copyright of the data belongs to original book/article authors or translators (hence, used for research purpose; 
and please contact Dr. Benjamin Frey for other copyright questions).

## Acknowledgement

We thank the Kituwah Preservation and Education Program (KPEP), the Eastern Band of Cherokee Indians, 
and the Cherokee Nation.