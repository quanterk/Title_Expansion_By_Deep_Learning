''' Define the Transformer model '''

import torch.nn as nn
import torch


def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)


def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    _, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask


class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, n_src_vocab, pad_idx, d_word_vec,
            d_model, d_inner, dropout=0.1):

        super().__init__()

        self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=pad_idx)

    def forward(self, src_seq, src_mask, return_attns=False):

        # -- Forward
        enc_output = self.src_word_emb(src_seq)
        return enc_output


class Fasttxt(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self, n_src_vocab, src_pad_idx, d_word_vec=512,
            d_model=512, d_inner=2048, dropout=0.1):

        super().__init__()

        self.n_src_vocab = n_src_vocab

        self.src_pad_idx = src_pad_idx

        self.d_model = d_model

        self.encoder = Encoder(
            n_src_vocab=n_src_vocab, d_word_vec=d_word_vec, d_model=d_model,
            d_inner=d_inner, pad_idx=src_pad_idx, dropout=dropout)

        self.fw2 = nn.Linear(self.d_model, self.n_src_vocab, bias=False)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        assert d_model == d_word_vec, \
            'To facilitate the residual connections, \
             the dimensions of all module outputs shall be the same.'

        self.x_logit_scale = 1.

    def forward(self, src_seq):

        src_mask = get_pad_mask(src_seq, self.src_pad_idx)
        enc_output, *_ = self.encoder(src_seq, src_mask)
        enc_output = enc_output.transpose(1, 2)
        enc_output = nn.AdaptiveAvgPool1d(1)(enc_output)
        enc_output = enc_output.view(enc_output.size(0), -1)
        enc_output = self.fw2(enc_output.cuda())
        enc_output = torch.sigmoid(enc_output)

        return enc_output
