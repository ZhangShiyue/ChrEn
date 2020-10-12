import os
import argparse

data_dir = os.getenv('DATA')
mosesdecoder_dir = os.getenv('MOSESDECODER')

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="enchr", type=str, required=True, help="model, enchr or chren",)
parser.add_argument("--output_dir", default=None, type=str, required=True, help="Output directory",)
parser.add_argument("--dev_file", default="dev", type=str, required=False, help="Development set, dev or out_dev",)
parser.add_argument("--test_file", default="test", type=str, required=False, help="Testing set, test or out_test",)
args = parser.parse_args()

dev_file = args.dev_file
test_file = args.test_file
output_dir = args.output_dir
os.mkdir(f"{output_dir}")


def enchr():
    # lm
    os.mkdir(f"{output_dir}/lm")
    os.system(f"{mosesdecoder_dir}/bin/lmplz -o 3 < {data_dir}/train.true.chr > {output_dir}/lm/train.arpa.chr")
    os.system(f"{mosesdecoder_dir}/bin/build_binary {output_dir}/lm/train.arpa.chr {output_dir}/lm/train.blm.chr")

    # train
    os.system(f"{mosesdecoder_dir}/scripts/training/train-model.perl -root-dir {output_dir}/train "
              f"-corpus {data_dir}/train.clean -f en -e chr "
              "-alignment grow-diag-final-and -reordering msd-bidirectional-fe "
              f"-lm 0:3:$PWD/{output_dir}/lm/train.blm.chr:8 "
              f"-external-bin-dir {mosesdecoder_dir}/tools")

    # tuning
    os.system(f"{mosesdecoder_dir}/scripts/training/mert-moses.pl "
              f"{data_dir}/{dev_file}.true.en {data_dir}/{dev_file}.true.chr "
              f"{mosesdecoder_dir}/bin/moses {output_dir}/train/model/moses.ini "
              f"--mertdir {mosesdecoder_dir}/bin/ "
              f"--decoder-flags='-threads 4'")
    os.system(f"mv mert-work {output_dir}/")

    for file in [dev_file, test_file]:
        # translate
        os.system(f"{mosesdecoder_dir}/bin/moses -f {output_dir}/mert-work/moses.ini "
                  f"< {data_dir}/{file}.true.en "
                  f"> {output_dir}/{file}.translated.chr")
        # detokenize
        os.system(f"{mosesdecoder_dir}/scripts/tokenizer/detokenizer.perl < {output_dir}/{file}.translated.chr "
                  f"> {output_dir}/{file}.detok.chr")
        # sacrebleu
        os.system(f"cat {output_dir}/{file}.detok.chr | "
                  f"sacrebleu {data_dir}/{file}.chr > {output_dir}/{file}_bleu --short")


def chren():
    # lm
    os.mkdir(f"{output_dir}/lm")
    os.system(f"{mosesdecoder_dir}/bin/lmplz -o 3 < {data_dir}/train.true.en > {output_dir}/lm/train.arpa.en")
    os.system(f"{mosesdecoder_dir}/bin/build_binary {output_dir}/lm/train.arpa.en {output_dir}/lm/train.blm.en")

    # train
    os.system(f"{mosesdecoder_dir}/scripts/training/train-model.perl -root-dir {output_dir}/train "
              f"-corpus {data_dir}/train.clean -f chr -e en "
              "-alignment grow-diag-final-and -reordering msd-bidirectional-fe "
              f"-lm 0:3:$PWD/{output_dir}/lm/train.blm.en:8 "
              f"-external-bin-dir {mosesdecoder_dir}/tools")

    # tuning
    os.system(f"{mosesdecoder_dir}/scripts/training/mert-moses.pl "
              f"{data_dir}/{dev_file}.true.chr {data_dir}/{dev_file}.true.en "
              f"{mosesdecoder_dir}/bin/moses {output_dir}/train/model/moses.ini "
              f"--mertdir {mosesdecoder_dir}/bin/ "
              f"--decoder-flags='-threads 4'")
    os.system(f"mv mert-work {output_dir}/")

    for file in [dev_file, test_file]:
        # translate
        os.system(f"{mosesdecoder_dir}/bin/moses -f {output_dir}/mert-work/moses.ini "
                  f"< {data_dir}/{file}.true.chr "
                  f"> {output_dir}/{file}.translated.en")
        # detruecase
        os.system(f"{mosesdecoder_dir}/scripts/recaser/detruecase.perl < {output_dir}/{file}.translated.en "
                  f"> {output_dir}/{file}.detrue.en")
        # detokenize
        os.system(f"{mosesdecoder_dir}/scripts/tokenizer/detokenizer.perl < {output_dir}/{file}.detrue.en "
                  f"> {output_dir}/{file}.detok.en")
        # sacrebleu
        os.system(f"cat {output_dir}/{file}.detok.en | "
                  f"sacrebleu {data_dir}/{file}.en > {output_dir}/{file}_bleu --short")


if args.model == "enchr":
    enchr()
elif args.model == "chren":
    chren()
else:
    print("wrong model!")