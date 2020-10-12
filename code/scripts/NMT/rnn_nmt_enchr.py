import os
import argparse

seed = 777
data_dir = os.getenv('DATA')
mosesdecoder_dir = os.getenv('MOSESDECODER')
train_dir = "rnn_nmt"

parser = argparse.ArgumentParser()
parser.add_argument("--dev_file", default="dev", type=str, required=True, help="Development set, dev or out_dev",)
parser.add_argument("--test_file", default="test", type=str, required=True, help="Testing set, test or out_test",)
args = parser.parse_args()

dev_file = args.dev_file
test_file = args.test_file


if dev_file == "dev":
    data = "enchr_min0"
    batch_size, batch_type = 1000, "tokens"
    layers, dropout = 2, 0.5
    rnn_size, rnn_type = 1024, "LSTM"
    label_smoothing = 0.2
elif dev_file == "out_dev":
    data = "out_enchr_min10"
    batch_size, batch_type = 32, "sents"
    layers, dropout = 2, 0.3
    rnn_size, rnn_type = 512, "LSTM"
    label_smoothing = 0.2
else:
    print("wrong!")
    exit()

outdir = f"{train_dir}/{data}_brnn_adam_0.0005_{batch_size}_{layers}_{dropout}_{rnn_size}_{rnn_type}_{label_smoothing}"
print(outdir)
# train
os.system(f"onmt_train -data {data_dir}/{data} -encoder_type brnn -gpu_ranks 0"
          f" -share_decoder_embeddings -word_vec_size {rnn_size}"
          f" -save_checkpoint_steps 0 -valid_steps 200 -train_steps 20000"
          f" -save_model {outdir}/model -tensorboard -tensorboard_log_dir {outdir}"
          f" -optim 'adam' -learning_rate 0.0005 -dropout {dropout}"
          f" -layers {layers} -rnn_size {rnn_size}"
          f" -batch_size {batch_size} -batch_type {batch_type} -early_stopping 10"
          f" -rnn_type {rnn_type} -label_smoothing {label_smoothing}"
          f" -average_decay 1e-4 -seed {seed} ")
# dev
# translate
os.system(f"onmt_translate -model {outdir}/model_step_0.pt -src {data_dir}/{dev_file}.true.en "
          f"-output {outdir}/{dev_file}.translated.chr -replace_unk -gpu 0")
# detokenize
os.system(f"{mosesdecoder_dir}/scripts/tokenizer/detokenizer.perl < {outdir}/{dev_file}.translated.chr "
          f"> {outdir}/{dev_file}.detok.chr")
# sacrebleu
os.system(f"cat {outdir}/{dev_file}.detok.chr | "
          f"sacrebleu {data_dir}/{dev_file}.chr > {outdir}/{dev_file}_bleu --short")

# test
# translate
os.system(f"onmt_translate -model {outdir}/model_step_0.pt -src {data_dir}/{test_file}.true.en "
          f"-output {outdir}/{test_file}.translated.chr -replace_unk -gpu 0")
# detokenize
os.system(f"{mosesdecoder_dir}/scripts/tokenizer/detokenizer.perl < {outdir}/{test_file}.translated.chr "
          f"> {outdir}/{test_file}.detok.chr")
# sacrebleu
os.system(f"cat {outdir}/{test_file}.detok.chr | "
          f"sacrebleu {data_dir}/{test_file}.chr > {outdir}/{test_file}_bleu --short")

# report
try:
    dev_bleu = open(f"{outdir}/{dev_file}_bleu", 'r').readline().strip()
    test_bleu = open(f"{outdir}/{test_file}_bleu", 'r').readline().strip()
    with open(f"{train_dir}/enchr.txt", 'a') as f:
        f.write(f"{outdir}\t{dev_bleu}\t{test_bleu}\n\n")
except:
    exit()