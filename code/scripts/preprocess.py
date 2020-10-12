import os

data_dir = os.getenv('DATA')
mosesdecoder_dir = os.getenv('MOSESDECODER')
files = ["train", "dev", "out_dev", "test", "out_test"]

# preprocess
# tokenize
for file in files:
    os.system(f"{mosesdecoder_dir}/scripts/tokenizer/tokenizer.perl -l en "
              f"< {data_dir}/{file}.en > {data_dir}/{file}.tok.en")
    os.system(f"{mosesdecoder_dir}/scripts/tokenizer/tokenizer.perl -l en "
              f"< {data_dir}/{file}.chr > {data_dir}/{file}.true.chr")
# truecaser
os.system(f"{mosesdecoder_dir}/scripts/recaser/train-truecaser.perl "
          f"--model {data_dir}/truecase-model.en "
          f"--corpus {data_dir}/train.tok.en")
for file in files:
    os.system(f"{mosesdecoder_dir}/scripts/recaser/truecase.perl "
              f"--model {data_dir}/truecase-model.en < {data_dir}/{file}.tok.en "
              f"> {data_dir}/{file}.true.en")
# clean
os.system(f"{mosesdecoder_dir}/scripts/training/clean-corpus-n.perl "
          f"{data_dir}/train.true en chr "
          f"{data_dir}/train.clean 1 80")