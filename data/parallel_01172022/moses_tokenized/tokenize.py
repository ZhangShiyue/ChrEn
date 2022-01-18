import os

MOSES_DECODER_PATH = "XXX"
data_dir = "../"

for data_file in ["train", "dev", "out_dev", "test", "out_test"]:
    os.system(f"{MOSES_DECODER_PATH}/scripts/tokenizer/tokenizer.perl -l en "
              f"< {data_dir}/{data_file}.en > {data_file}.tok.en")
    os.system(f"{MOSES_DECODER_PATH}/scripts/tokenizer/tokenizer.perl -l en "
              f"< {data_dir}/{data_file}.chr > {data_file}.true.chr")

    # truecaser
    os.system(f"{MOSES_DECODER_PATH}/scripts/recaser/truecase.perl "
              f"--model truecase-model.en < {data_file}.tok.en "
              f"> {data_file}.true.en")

    # clean
    os.system(f"{MOSES_DECODER_PATH}/scripts/training/clean-corpus-n.perl "
              f"train.true en chr "
              f"train.clean 1 80")