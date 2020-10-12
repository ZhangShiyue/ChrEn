"""
Implementation of "Attention is All You Need"
"""
import torch
import torch.nn as nn

from onmt.encoders.encoder import EncoderBase
from onmt.modules import MultiHeadedAttention
from onmt.modules.position_ffn import PositionwiseFeedForward
from onmt.utils.misc import sequence_mask

from transformers import BertModel, BertConfig


class TransformerEncoderLayer(nn.Module):
    """
    A single layer of the transformer encoder.

    Args:
        d_model (int): the dimension of keys/values/queries in
                   MultiHeadedAttention, also the input size of
                   the first-layer of the PositionwiseFeedForward.
        heads (int): the number of head for MultiHeadedAttention.
        d_ff (int): the second-layer of the PositionwiseFeedForward.
        dropout (float): dropout probability(0-1.0).
    """

    def __init__(self, d_model, heads, d_ff, dropout, attention_dropout,
                 max_relative_positions=0):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = MultiHeadedAttention(
            heads, d_model, dropout=attention_dropout,
            max_relative_positions=max_relative_positions)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, mask):
        """
        Args:
            inputs (FloatTensor): ``(batch_size, src_len, model_dim)``
            mask (LongTensor): ``(batch_size, 1, src_len)``

        Returns:
            (FloatTensor):

            * outputs ``(batch_size, src_len, model_dim)``
        """
        input_norm = self.layer_norm(inputs)
        context, _ = self.self_attn(input_norm, input_norm, input_norm,
                                    mask=mask, attn_type="self")
        out = self.dropout(context) + inputs
        return self.feed_forward(out)

    def update_dropout(self, dropout, attention_dropout):
        self.self_attn.update_dropout(attention_dropout)
        self.feed_forward.update_dropout(dropout)
        self.dropout.p = dropout


class TransformerEncoder(EncoderBase):
    """The Transformer encoder from "Attention is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`

    .. mermaid::

       graph BT
          A[input]
          B[multi-head self-attn]
          C[feed forward]
          O[output]
          A --> B
          B --> C
          C --> O

    Args:
        num_layers (int): number of encoder layers
        d_model (int): size of the model
        heads (int): number of heads
        d_ff (int): size of the inner FF layer
        dropout (float): dropout parameters
        embeddings (onmt.modules.Embeddings):
          embeddings to use, should have positional encodings

    Returns:
        (torch.FloatTensor, torch.FloatTensor):

        * embeddings ``(src_len, batch_size, model_dim)``
        * memory_bank ``(src_len, batch_size, model_dim)``
    """

    def __init__(self, num_layers, d_model, word_dim, heads, d_ff, dropout,
                 attention_dropout, embeddings, max_relative_positions,
                 bert_model=None, bert_layers=1, bert_input=False,
                 bert_output=None, bert_attn=None, cache_dir=None):
        super(TransformerEncoder, self).__init__()

        self.embeddings = embeddings
        self.transformer = nn.ModuleList(
            [TransformerEncoderLayer(
                d_model, heads, d_ff, dropout, attention_dropout,
                max_relative_positions=max_relative_positions)
             for i in range(num_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        if word_dim == 768:
            self.emb_map = nn.Linear(word_dim, d_model)

        self.bert_layers = bert_layers
        if self.bert_layers > 0:
            config = BertConfig.from_pretrained(bert_model, output_hidden_states=True, cache_dir=cache_dir)
            self.bert_model = BertModel.from_pretrained(bert_model, config=config, cache_dir=cache_dir)
            self.bert_model.cuda()
            self.bert_model.eval()
            # take bert as input
            self.bert_input = bert_input
            if self.bert_input:
                self.input_map = nn.Linear(word_dim + 768, d_model)
            # take bert as output, either directly concat with RNN's output or use another attention
            self.bert_output = bert_output
            self.bert_attn = bert_attn
            if self.bert_output and not self.bert_attn:
                self.output_map = nn.Linear(d_model + 768, d_model)

    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor."""
        return cls(
            opt.enc_layers,
            opt.enc_rnn_size,
            opt.src_word_vec_size,
            opt.heads,
            opt.transformer_ff,
            opt.dropout[0] if type(opt.dropout) is list else opt.dropout,
            opt.attention_dropout[0] if type(opt.attention_dropout)
            is list else opt.attention_dropout,
            embeddings,
            opt.max_relative_positions,
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

        s_len, batch, emb_dim = emb.size()
        if emb_dim == 768:
            emb = self.emb_map(emb)

        out = emb.transpose(0, 1).contiguous()
        mask = ~sequence_mask(lengths).unsqueeze(1)
        # Run the forward pass of every layer of the tranformer.
        for layer in self.transformer:
            out = layer(out, mask)
        out = self.layer_norm(out)

        if self.bert_layers > 0 and self.bert_output:
            out = torch.cat([out, bert_emb.transpose(0, 1)], dim=2)
            if not self.bert_attn:
                out = self.output_map(out)

        return emb, out.transpose(0, 1).contiguous(), lengths

    def update_dropout(self, dropout, attention_dropout):
        self.embeddings.update_dropout(dropout)
        for layer in self.transformer:
            layer.update_dropout(dropout, attention_dropout)
