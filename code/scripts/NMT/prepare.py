import os
import argparse


data_dir = os.getenv('DATA')
mosesdecoder_dir = os.getenv('MOSESDECODER')

parser = argparse.ArgumentParser()
parser.add_argument("--mode", default="get_vocab", type=str, required=True, help="Which function to run, prepare",)
parser.add_argument("--min_freq", default=0, type=int, required=False, help="Minimum frequency threshold",)
args = parser.parse_args()


def get_vocab(files, min_freq=0):
    # get vocabulary
    vocab = []
    vocab_dict = {}
    for file in files:
        with open(file, 'r') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                words = line.strip().split(' ')
                for word in words:
                    if word not in vocab_dict:
                        vocab.append(word)
                        vocab_dict[word] = 0
                    vocab_dict[word] += 1
    vocab = [(vocab_dict[word], word) for word in vocab]
    vocab = sorted(vocab, key=lambda x: x[0], reverse=True)
    if min_freq > 0:
        vocab = list(filter(lambda x: x[0] >= min_freq, vocab))
    vocab = [w for n, w in vocab]
    return vocab


def prepare(freq):
    # get vocabularies
    # Cherokee vocabulary
    files = [f"{data_dir}/train.true.chr"]
    vocab_chr = get_vocab(files, min_freq=args.min_freq)
    with open(f"{data_dir}/vocab.min{args.min_freq}.chr", 'w') as f:
        f.write('\n'.join(vocab_chr))
    # English vocabulary
    files = [f"{data_dir}/train.true.en"]
    vocab_en= get_vocab(files, min_freq=args.min_freq)
    with open(f"{data_dir}/vocab.min{args.min_freq}.en", 'w') as f:
        f.write('\n'.join(vocab_en))

    # prepare experimental enchr/chren data
    os.system(f"onmt_preprocess -train_src {data_dir}/train.true.en "
              f"-train_tgt {data_dir}/train.true.chr "
              f"-valid_src {data_dir}/dev.true.en "
              f"-valid_tgt {data_dir}/dev.true.chr "
              f"-src_vocab {data_dir}/vocab.min{freq}.en "
              f"-tgt_vocab {data_dir}/vocab.min{freq}.chr "
              f"-save_data {data_dir}/enchr_min{freq} "
              "-src_seq_length 80 "
              "-tgt_seq_length 80 "
              "-overwrite")
    os.system(f"onmt_preprocess -train_src {data_dir}/train.true.chr "
              f"-train_tgt {data_dir}/train.true.en "
              f"-valid_src {data_dir}/dev.true.chr "
              f"-valid_tgt {data_dir}/dev.true.en "
              f"-src_vocab {data_dir}/vocab.min{freq}.chr "
              f"-tgt_vocab {data_dir}/vocab.min{freq}.en "
              f"-save_data {data_dir}/chren_min{freq} "
              "-src_seq_length 80 "
              "-tgt_seq_length 80 "
              "-overwrite")
    os.system(f"onmt_preprocess -train_src {data_dir}/train.true.en "
              f"-train_tgt {data_dir}/train.true.chr "
              f"-valid_src {data_dir}/out_dev.true.en "
              f"-valid_tgt {data_dir}/out_dev.true.chr "
              f"-src_vocab {data_dir}/vocab.min{freq}.en "
              f"-tgt_vocab {data_dir}/vocab.min{freq}.chr "
              f"-save_data {data_dir}/out_enchr_min{freq} "
              "-src_seq_length 80 "
              "-tgt_seq_length 80 "
              "-overwrite")
    os.system(f"onmt_preprocess -train_src {data_dir}/train.true.chr "
              f"-train_tgt {data_dir}/train.true.en "
              f"-valid_src {data_dir}/out_dev.true.chr "
              f"-valid_tgt {data_dir}/out_dev.true.en "
              f"-src_vocab {data_dir}/vocab.min{freq}.chr "
              f"-tgt_vocab {data_dir}/vocab.min{freq}.en "
              f"-save_data {data_dir}/out_chren_min{freq} "
              "-src_seq_length 80 "
              "-tgt_seq_length 80 "
              "-overwrite")


if args.mode == "prepare":
    prepare(freq=args.min_freq)
else:
    print("wrong mode!")

