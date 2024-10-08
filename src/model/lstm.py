# Copyright (c) 2020-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
# import math
# import itertools
# import numpy as np
import torch
# from torch._C import _set_backcompat_keepdim_warn
import torch.nn as nn
import torch.nn.functional as F
from .transformer import Embedding
from .transformer import BeamHypotheses


logger = getLogger()


def get_masks(slen, lengths):
    """
    Generate hidden states mask, and optionally an attention mask.
    """
    assert lengths.max().item() <= slen
    bs = lengths.size(0)
    alen = torch.arange(slen, dtype=torch.long, device=lengths.device)
    mask = alen < lengths[:, None]

    # sanity check
    assert mask.size() == (bs, slen)

    return mask


class LSTMModel(nn.Module):
    def __init__(self, params, id2word, is_encoder, with_output):
        """
        Transformer model (encoder or decoder).
        """
        super().__init__()

        # encoder / decoder, output layer
        self.dtype = torch.half if params.fp16 else torch.float
        self.is_encoder = is_encoder
        self.is_decoder = not is_encoder
        self.with_output = with_output

        self.lstm = params.lstm
        self.GRU = params.GRU
        
        # dictionary
        self.n_words = params.n_words
        self.eos_index = params.eos_index
        self.pad_index = params.pad_index
        self.id2word = id2word
        assert len(self.id2word) == self.n_words

        # model parameters
        assert params.enc_emb_dim == params.dec_emb_dim
        self.dim = params.enc_emb_dim if is_encoder else params.dec_emb_dim  # 512 by default
        self.src_dim = params.enc_emb_dim
        self.bidirectional = params.bidirectional
        self.hidden_dim = params.lstm_hidden_dim
        self.n_layers = params.n_enc_layers if is_encoder else params.n_dec_layers
        self.dropout = params.dropout
        
        # embeddings
        self.embeddings = Embedding(self.n_words, self.dim, padding_idx=self.pad_index)
        self.layer_norm_emb = nn.LayerNorm(self.dim, eps=1e-12)

        # transformer layers
        if self.GRU:
            self.layers = nn.GRU(self.dim, self.hidden_dim, self.n_layers, bidirectional=self.bidirectional, dropout=self.dropout, batch_first=True)
        else:
            self.layers = nn.LSTM(self.dim, self.hidden_dim, self.n_layers, bidirectional=self.bidirectional, dropout=self.dropout, batch_first=True)
        self.gru_out = nn.Linear(self.hidden_dim*(2 if self.bidirectional else 1) , self.dim, bias=True)
        
        # output layer
        if self.with_output:
            self.proj = nn.Linear(self.dim, params.n_words, bias=True)
            if params.share_inout_emb:
                self.proj.weight = self.embeddings.weight

    def forward(self, mode, **kwargs):
        """
        Forward function with different forward modes.
        ### Small hack to handle PyTorch distributed.
        """
        if mode == "fwd":
            return self.fwd(**kwargs)
        elif mode == "predict":
            return self.predict(**kwargs)
        else:
            raise Exception("Unknown mode: %s" % mode)

    def fwd(
        self,
        x,
        lengths,
        causal,
        src_enc=None,
        src_len=None,
        use_cache=False,
    ):
        """
        Inputs:
            `x` LongTensor(slen, bs), containing word indices
            `lengths` LongTensor(bs), containing the length of each sentence
            `causal` Boolean, if True, the attention is only done over previous hidden states
        """
        # lengths = (x != self.pad_index).float().sum(dim=1)
        # mask = x != self.pad_index

        # check inputs
        slen, bs = x.size()
        assert lengths.size(0) == bs
        assert lengths.max().item() <= slen
        x = x.transpose(0, 1)  # batch size as dimension 0
        if src_enc is not None:
            assert self.is_decoder
            #print(np.shape(src_enc))
            #assert src_enc.size(1) == bs

        # generate masks
        mask = get_masks(slen, lengths)

        # embeddings
        tensor = self.embeddings(x)
        tensor = self.layer_norm_emb(tensor)
        
        tensor = F.dropout(tensor, p=self.dropout, training=self.training)
        tensor *= mask.unsqueeze(-1).to(tensor.dtype)

        # transformer layers
        tensor, hidden = self.layers.forward(tensor, src_enc)
        tensor = self.gru_out(tensor)

        # move back sequence length to dimension 0
        tensor = tensor.transpose(0, 1)

        return tensor, hidden


    def predict(self, tensor, pred_mask, y, get_scores):
        """
        Given the last hidden state, compute word scores and/or the loss.
            `pred_mask` is a ByteTensor of shape (slen, bs), filled with 1 when
                we need to predict a word
            `y` is a LongTensor of shape (pred_mask.sum(),)
            `get_scores` is a boolean specifying whether we need to return scores
        """
        x = tensor[pred_mask.unsqueeze(-1).expand_as(tensor)].view(-1, self.dim)
        assert (y == self.pad_index).sum().item() == 0
        # print(np.shape(x))
        scores = self.proj(x).view(-1, self.n_words)
        # print(np.shape(scores))
        loss = F.cross_entropy(scores.float(), y, reduction="mean")
        return scores, loss

    def generate(self, src_enc, src_len, max_len=200, sample_temperature=None):
        """
        Decode a sentence given initial start.
        `x`:
            - LongTensor(bs, slen)
                <EOS> W1 W2 W3 <EOS> <PAD>
                <EOS> W1 W2 W3   W4  <EOS>
        `lengths`:
            - LongTensor(bs) [5, 6]
        """

        # input batch
        if self.GRU:
            bs = src_enc.size(1)
        else:
            bs = src_enc[0].size(1)


        # generated sentences
        generated = src_len.new(max_len, bs)  # upcoming output
        generated.fill_(self.pad_index)  # fill upcoming ouput with <PAD>
        generated[0].fill_(self.eos_index)  # we use <EOS> for <BOS> everywhere

        # current position / max lengths / length of generated sentences / unfinished sentences
        cur_len = 1
        gen_len = src_len.clone().fill_(1)
        unfinished_sents = src_len.clone().fill_(1)

        # cache compute states
        self.cache = {"slen": 0}

        while cur_len < max_len:

            # compute word scores
            tensor, _ = self.forward(
                "fwd",
                x=generated[:cur_len],
                lengths=gen_len.new(bs).fill_(cur_len),
                causal=True,
                src_enc=src_enc,
            )
            tensor = tensor[-1:,:,:]
            assert tensor.size() == (1, bs, self.dim), tensor.size()
            tensor = tensor.data[-1, :, :] #.to(self.dtype)  # (bs, dim)
            scores = self.proj(tensor)  # (bs, n_words)

            # select next words: sample or greedy
            if sample_temperature is None:
                next_words = torch.topk(scores, 1)[1].squeeze(1)
            else:
                next_words = torch.multinomial(
                    F.softmax(scores.float() / sample_temperature, dim=1), 1
                ).squeeze(1)
            assert next_words.size() == (bs,)

            # update generations / lengths / finished sentences / current length
            generated[cur_len] = next_words * unfinished_sents + self.pad_index * (
                1 - unfinished_sents
            )
            gen_len.add_(unfinished_sents)
            unfinished_sents.mul_(next_words.ne(self.eos_index).long())
            cur_len = cur_len + 1

            # stop when there is a </s> in each sentence, or if we exceed the maximul length
            if unfinished_sents.max() == 0:
                break

        # add <EOS> to unfinished sentences
        if cur_len == max_len:
            generated[-1].masked_fill_(unfinished_sents.byte(), self.eos_index)

        # sanity check
        assert (generated == self.eos_index).sum() == 2 * bs

        return generated[:cur_len].cpu(), gen_len.cpu()

    def generate_beam(
        self, src_enc, src_len, beam_size, length_penalty, early_stopping, max_len=200
    ):
        """
        Decode a sentence given initial start.
        `x`:
            - LongTensor(bs, slen)
                <EOS> W1 W2 W3 <EOS> <PAD>
                <EOS> W1 W2 W3   W4  <EOS>
        `lengths`:
            - LongTensor(bs) [5, 6]
        """

        # check inputs
        #assert src_enc.size(0) == src_len.size(0)
        assert beam_size == 1

        # batch size / number of words
        n_words = self.n_words
        if self.GRU:
            bs = src_enc.size(1)
            #src_enc = (
            #    src_enc.unsqueeze(1)
            #    .expand((bs, beam_size) + src_enc.shape[1:])
            #    .contiguous()
            #    .view((bs * beam_size,) + src_enc.shape[1:])
            #)
        else:
            bs = src_enc[0].size(1)
                
            # expand to beam size the source latent representations / source lengths
            #src_enc = (
            #    src_enc[0].unsqueeze(1)
            #    .expand((bs, beam_size) + src_enc[0].shape[1:])
            #    .contiguous()
            #    .view((bs * beam_size,) + src_enc[0].shape[1:]),
            #    src_enc[1].unsqueeze(1)
            #    .expand((bs, beam_size) + src_enc[1].shape[1:])
            #    .contiguous()
            #    .view((bs * beam_size,) + src_enc[1].shape[1:])                
            #)
        src_len = src_len.unsqueeze(1).expand(bs, beam_size).contiguous().view(-1)

        # generated sentences (batch with beam current hypotheses)
        generated = src_len.new(max_len, bs * beam_size)  # upcoming output
        generated.fill_(self.pad_index)  # fill upcoming ouput with <PAD>
        generated[0].fill_(self.eos_index)  # we use <EOS> for <BOS> everywhere

        # generated hypotheses
        generated_hyps = [
            BeamHypotheses(beam_size, max_len, length_penalty, early_stopping)
            for _ in range(bs)
        ]

        # scores for each sentence in the beam
        beam_scores = src_len.new(bs, beam_size).float().fill_(0)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view(-1)

        # current position
        cur_len = 1

        # cache compute states
        self.cache = {"slen": 0}

        # done sentences
        done = [False for _ in range(bs)]

        while cur_len < max_len:

            # compute word scores
            tensor, _ = self.forward(
                "fwd",
                x=generated[:cur_len],
                lengths=src_len.new(bs * beam_size).fill_(cur_len),
                causal=True,
                src_enc=src_enc,
            )
            tensor = tensor[-1:,:,:]
            assert tensor.size() == (1, bs * beam_size, self.dim), tensor.size()
            tensor = tensor.data[-1, :, :]  # .to(self.dtype)  # (bs * beam_size, dim)
            scores = self.proj(tensor)  # (bs * beam_size, n_words)
            scores = F.log_softmax(scores.float(), dim=-1)  # (bs * beam_size, n_words)
            assert scores.size() == (bs * beam_size, n_words)

            # select next words with scores
            _scores = scores + beam_scores[:, None].expand_as(
                scores
            )  # (bs * beam_size, n_words)
            _scores = _scores.view(bs, beam_size * n_words)  # (bs, beam_size * n_words)

            next_scores, next_words = torch.topk(
                _scores, 2 * beam_size, dim=1, largest=True, sorted=True
            )
            assert next_scores.size() == next_words.size() == (bs, 2 * beam_size)

            # next batch beam content
            # list of (bs * beam_size) tuple(next hypothesis score, next word, current position in the batch)
            next_batch_beam = []

            # for each sentence
            for sent_id in range(bs):

                # if we are done with this sentence
                done[sent_id] = done[sent_id] or generated_hyps[sent_id].is_done(
                    next_scores[sent_id].max().item()
                )
                if done[sent_id]:
                    next_batch_beam.extend(
                        [(0, self.pad_index, 0)] * beam_size
                    )  # pad the batch
                    continue

                # next sentence beam content
                next_sent_beam = []

                # next words for this sentence
                for idx, value in zip(next_words[sent_id], next_scores[sent_id]):

                    # get beam and word IDs
                    beam_id = idx // n_words
                    word_id = idx % n_words

                    # end of sentence, or next word
                    if word_id == self.eos_index or cur_len + 1 == max_len:
                        generated_hyps[sent_id].add(
                            generated[:cur_len, sent_id * beam_size + beam_id]
                            .clone()
                            .cpu(),
                            value.item(),
                        )
                    else:
                        next_sent_beam.append(
                            (value, word_id, sent_id * beam_size + beam_id)
                        )

                    # the beam for next step is full
                    if len(next_sent_beam) == beam_size:
                        break

                # update next beam content
                assert len(next_sent_beam) == 0 if cur_len + 1 == max_len else beam_size
                if len(next_sent_beam) == 0:
                    next_sent_beam = [
                        (0, self.pad_index, 0)
                    ] * beam_size  # pad the batch
                next_batch_beam.extend(next_sent_beam)
                assert len(next_batch_beam) == beam_size * (sent_id + 1)

            # sanity check / prepare next batch
            assert len(next_batch_beam) == bs * beam_size
            beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
            beam_words = generated.new([x[1] for x in next_batch_beam])
            beam_idx = src_len.new([x[2] for x in next_batch_beam])

            # re-order batch and internal states
            generated = generated[:, beam_idx]
            generated[cur_len] = beam_words
            for k in self.cache.keys():
                if k != "slen":
                    self.cache[k] = (
                        self.cache[k][0][beam_idx],
                        self.cache[k][1][beam_idx],
                    )

            # update current length
            cur_len = cur_len + 1

            # stop when we are done with each sentence
            if all(done):
                break

        # def get_coeffs(s):
        #     roots = [int(s[i + 2]) for i, c in enumerate(s) if c == 'x']
        #     poly = np.poly1d(roots, r=True)
        #     coeffs = list(poly.coefficients.astype(np.int64))
        #     return [c % 10 for c in coeffs], coeffs

        # visualize hypotheses
        # print([len(x) for x in generated_hyps], cur_len)
        # globals().update( locals() );
        # !import code; code.interact(local=vars())
        # for ii in range(bs):
        #     for ss, ww in sorted(generated_hyps[ii].hyp, key=lambda x: x[0], reverse=True):
        #         hh = " ".join(self.id2word[x] for x in ww.tolist())
        #         print(f"{ss:+.4f} {hh}")
        #         # cc = get_coeffs(hh[4:])
        #         # print(f"{ss:+.4f} {hh} || {cc[0]} || {cc[1]}")
        #     print("")

        # select the best hypotheses
        tgt_len = src_len.new(bs)
        best = []

        for i, hypotheses in enumerate(generated_hyps):
            best_hyp = max(hypotheses.hyp, key=lambda x: x[0])[1]
            tgt_len[i] = len(best_hyp) + 1  # +1 for the <EOS> symbol
            best.append(best_hyp)

        # generate target batch
        decoded = src_len.new(tgt_len.max().item(), bs).fill_(self.pad_index)
        for i, hypo in enumerate(best):
            decoded[: tgt_len[i] - 1, i] = hypo
            decoded[tgt_len[i] - 1, i] = self.eos_index

        # sanity check
        assert (decoded == self.eos_index).sum() == 2 * bs

        return decoded, tgt_len, generated_hyps
