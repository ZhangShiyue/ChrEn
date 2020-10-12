"""Define RNN-based encoders."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

from onmt.encoders.encoder import EncoderBase
from onmt.utils.rnn_factory import rnn_factory

from transformers import BertModel, BertConfig


class RNNEncoder(EncoderBase):
    """ A generic recurrent neural network encoder.

    Args:
       rnn_type (str):
          style of recurrent unit to use, one of [RNN, LSTM, GRU, SRU]
       bidirectional (bool) : use a bidirectional RNN
       num_layers (int) : number of stacked layers
       hidden_size (int) : hidden size of each layer
       dropout (float) : dropout value for :class:`torch.nn.Dropout`
       embeddings (onmt.modules.Embeddings): embedding module to use
    """

    def __init__(self, rnn_type, bidirectional, num_layers,
                 hidden_size, word_dim, dropout=0.0, embeddings=None,
                 use_bridge=False, bert_model=None,
                 bert_layers=1, bert_input=False, bert_output=False,
                 bert_attn=False, cache_dir=None):
        super(RNNEncoder, self).__init__()
        assert embeddings is not None

        emb_dim = word_dim
        hidden_dim = hidden_size
        num_directions = 2 if bidirectional else 1
        assert hidden_size % num_directions == 0
        hidden_size = hidden_size // num_directions
        self.embeddings = embeddings

        self.rnn, self.no_pack_padded_seq = \
            rnn_factory(rnn_type,
                        input_size=embeddings.embedding_size,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        dropout=dropout,
                        bidirectional=bidirectional)

        # Initialize the bridge layer
        self.use_bridge = use_bridge
        if self.use_bridge:
            self._initialize_bridge(rnn_type,
                                    hidden_size,
                                    num_layers)

        self.bert_layers = bert_layers
        if self.bert_layers > 0:
            config = BertConfig.from_pretrained(bert_model, output_hidden_states=True, cache_dir=cache_dir)
            self.bert_model = BertModel.from_pretrained(bert_model, config=config, cache_dir=cache_dir)
            self.bert_model.cuda()
            self.bert_model.eval()
            # take bert as input
            self.bert_input = bert_input
            if self.bert_input:
                self.input_map = nn.Linear(emb_dim + 768, emb_dim)
            # take bert as output, either directly concat with RNN's output or use another attention
            self.bert_output = bert_output
            self.bert_attn = bert_attn
            if self.bert_output and not self.bert_attn:
                self.output_map = nn.Linear(hidden_dim + 768, hidden_dim)

    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor."""
        return cls(
            opt.rnn_type,
            opt.brnn,
            opt.enc_layers,
            opt.enc_rnn_size,
            opt.src_word_vec_size,
            opt.dropout[0] if type(opt.dropout) is list else opt.dropout,
            embeddings,
            opt.bridge,
            opt.bert_model,
            opt.bert_layers,
            opt.bert_input,
            opt.bert_output,
            opt.bert_attn,
            opt.cache_dir)

    def forward(self, src, lengths=None):
        """See :func:`EncoderBase.forward()`"""
        self._check_args(src, lengths)

        emb = self.embeddings(src)
        s_len, batch, emb_dim = emb.size()

        if self.bert_layers > 0:
            src_bert = src.transpose(0, 1).view(batch, s_len)
            start_tokens = torch.zeros((batch, 1), dtype=torch.long) + torch.tensor(101, dtype=torch.long)
            end_tokens = torch.zeros((batch, 1), dtype=torch.long) + torch.tensor(102, dtype=torch.long)
            src_bert = torch.cat([start_tokens.cuda(), src_bert, end_tokens.cuda()], dim=1)
            _, _, outputs = self.bert_model(input_ids=src_bert)
            outputs = outputs[::-1]
            bert_emb = outputs[0][:, 1:-1, :].transpose(0, 1)
            for i in range(1, self.bert_layers):
                bert_emb += outputs[i][:, 1:-1, :].transpose(0, 1)
            bert_emb /= self.bert_layers

        if self.bert_layers > 0 and self.bert_input:
            emb = torch.cat([emb, bert_emb], dim=2)
            emb = self.input_map(emb)

        packed_emb = emb
        if lengths is not None and not self.no_pack_padded_seq:
            # Lengths data is wrapped inside a Tensor.
            lengths_list = lengths.view(-1).tolist()
            packed_emb = pack(emb, lengths_list)

        memory_bank, encoder_final = self.rnn(packed_emb)
        if lengths is not None and not self.no_pack_padded_seq:
            memory_bank = unpack(memory_bank)[0]

        if self.bert_layers > 0 and self.bert_output:
            memory_bank = torch.cat([memory_bank, bert_emb], dim=2)
            if not self.bert_attn:
                memory_bank = self.output_map(memory_bank)

        if self.use_bridge:
            encoder_final = self._bridge(encoder_final)
        return encoder_final, memory_bank, lengths

    def _initialize_bridge(self, rnn_type,
                           hidden_size,
                           num_layers):

        # LSTM has hidden and cell state, other only one
        number_of_states = 2 if rnn_type == "LSTM" else 1
        # Total number of states
        self.total_hidden_dim = hidden_size * num_layers

        # Build a linear layer for each
        self.bridge = nn.ModuleList([nn.Linear(self.total_hidden_dim,
                                               self.total_hidden_dim,
                                               bias=True)
                                     for _ in range(number_of_states)])

    def _bridge(self, hidden):
        """Forward hidden state through bridge."""
        def bottle_hidden(linear, states):
            """
            Transform from 3D to 2D, apply linear and return initial size
            """
            size = states.size()
            result = linear(states.view(-1, self.total_hidden_dim))
            return F.relu(result).view(size)

        if isinstance(hidden, tuple):  # LSTM
            outs = tuple([bottle_hidden(layer, hidden[ix])
                          for ix, layer in enumerate(self.bridge)])
        else:
            outs = bottle_hidden(self.bridge[0], hidden)
        return outs

    def update_dropout(self, dropout):
        self.rnn.dropout = dropout
