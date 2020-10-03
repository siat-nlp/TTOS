#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
File: source/model/seq2seq.py
"""

import torch
import torch.nn as nn

from source.model.base_model import BaseModel
from source.module.embedder import Embedder
from source.module.rnn_encoder import RNNEncoder
from source.module.rnn_decoder import RNNDecoder
from source.utils.criterions import NLLLoss, MaskBCELoss
from source.utils.metrics import accuracy
from source.utils.misc import Pack
from source.utils.misc import sequence_mask


class Seq2Seq(BaseModel):
    """
    Seq2Seq
    """
    def __init__(self,
                 src_field,
                 tgt_field,
                 kb_field,
                 embed_size,
                 hidden_size,
                 padding_idx=None,
                 num_layers=1,
                 bidirectional=False,
                 attn_mode="mlp",
                 with_bridge=False,
                 tie_embedding=False,
                 max_hop=1,
                 dropout=0.0,
                 use_gpu=False):
        super(Seq2Seq, self).__init__()

        self.name = 'model_S'
        self.src_field = src_field
        self.tgt_field = tgt_field
        self.kb_field = kb_field
        self.src_vocab_size = src_field.vocab_size
        self.tgt_vocab_size = tgt_field.vocab_size
        self.kb_vocab_size = kb_field.vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.padding_idx = padding_idx
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.attn_mode = attn_mode
        self.with_bridge = with_bridge
        self.tie_embedding = tie_embedding
        self.max_hop = max_hop
        self.dropout = dropout
        self.use_gpu = use_gpu

        # TODO: used for rl training
        self.BOS = self.tgt_field.stoi[self.tgt_field.bos_token]

        self.enc_embedder = Embedder(num_embeddings=self.src_vocab_size,
                                     embedding_dim=self.embed_size,
                                     padding_idx=self.padding_idx)

        self.encoder = RNNEncoder(input_size=self.embed_size,
                                  hidden_size=self.hidden_size,
                                  embedder=self.enc_embedder,
                                  num_layers=self.num_layers,
                                  bidirectional=self.bidirectional,
                                  dropout=self.dropout)

        if self.with_bridge:
            self.bridge = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.Tanh(),
            )

        if self.tie_embedding:
            assert self.src_vocab_size == self.tgt_vocab_size
            self.dec_embedder = self.enc_embedder
            self.kb_embedder = self.enc_embedder
        else:
            self.dec_embedder = Embedder(num_embeddings=self.tgt_vocab_size,
                                         embedding_dim=self.embed_size,
                                         padding_idx=self.padding_idx)
            self.kb_embedder = Embedder(num_embeddings=self.kb_vocab_size,
                                        embedding_dim=self.embed_size,
                                        padding_idx=self.padding_idx)

        # TODO: change KB to MLP transform
        self.trans_layer = nn.Linear(3 * self.embed_size, self.hidden_size, bias=True)

        # init memory
        self.dialog_state_memory = None
        self.dialog_history_memory = None
        self.memory_masks = None
        self.kbs = None
        self.kb_state_memory = None
        self.kb_slot_memory = None
        self.history_index = None
        self.kb_slot_index = None
        self.kb_mask = None
        self.selector_mask = None

        self.decoder = RNNDecoder(embedder=self.dec_embedder,
                                  max_hop=self.max_hop,
                                  input_size=self.embed_size,
                                  hidden_size=self.hidden_size,
                                  output_size=self.tgt_vocab_size,
                                  kb_output_size=self.kb_vocab_size,
                                  num_layers=self.num_layers,
                                  attn_mode=self.attn_mode,
                                  memory_size=self.hidden_size,
                                  kb_memory_size=self.hidden_size,  # Note: hidden_size if MLP
                                  dropout=self.dropout,
                                  padding_idx=self.padding_idx,
                                  use_gpu=self.use_gpu)
        self.sigmoid = nn.Sigmoid()

        # loss definition
        if self.padding_idx is not None:
            weight = torch.ones(self.tgt_vocab_size)
            weight[self.padding_idx] = 0
        else:
            weight = None
        self.nll_loss = NLLLoss(weight=weight,
                                ignore_index=self.padding_idx,
                                reduction='mean')
        self.bce_loss = MaskBCELoss()

        if self.use_gpu:
            self.cuda()

    def reset_memory(self):
        """
        reset memory
        """
        self.dialog_state_memory = None
        self.dialog_history_memory = None
        self.memory_masks = None
        self.kbs = None
        self.kb_state_memory = None
        self.kb_slot_memory = None
        self.history_index = None
        self.kb_slot_index = None
        self.kb_mask = None
        self.selector_mask = None

    def load_kb_memory(self, kb_inputs):
        """
        load kb memory
        """
        kbs, kb_lengths = kb_inputs
        if self.use_gpu:
            kbs = kbs.cuda()
            kb_lengths = kb_lengths.cuda()

        batch_size, kb_num, kb_term = kbs.size()
        kbs = kbs[:, :, 1:-1]       # filter <bos> <eos>
        self.kbs = kbs

        # TODO: change kb_states
        #kb_states = kbs[:, :, :-1]  # <subject, relation>
        kb_states = kbs
        kb_slots = kbs[:, :, -1]    # <object>
        kb_states = kb_states.contiguous().view(batch_size, kb_num, -1)  # (batch_size, kb_num, 3)
        kb_slots = kb_slots.contiguous().view(batch_size, kb_num)  # (batch_size, kb_num)

        kb_mask = kb_lengths.eq(0)
        self.kb_mask = kb_mask      # (batch_size, kb_num)
        selector_mask = kb_mask.eq(0)
        self.selector_mask = selector_mask  # (batch_size, kb_num)

        embed_state = self.kb_embedder(kb_states)
        #self.kb_state_memory = embed_state.contiguous().view(
        #    batch_size, kb_num, -1)    # concat <subject, relation> embedding
        embed_state = embed_state.contiguous().view(batch_size, kb_num, -1)
        self.kb_state_memory = self.trans_layer(embed_state)
        self.kb_slot_memory = self.kb_state_memory.clone()

        #self.kb_state_memory = torch.mean(embed_state, dim=2)  # mean of <subject, relation, object> embedding
        #self.kb_slot_memory = self.kb_embedder(kb_slots)   # <object> embedding
        self.kb_slot_index = kb_slots

    def encode(self, enc_inputs, hidden=None):
        """
        encode
        """
        outputs = Pack()
        enc_outputs, enc_hidden = self.encoder(enc_inputs, hidden)
        inputs, lengths = enc_inputs
        batch_size = enc_outputs.size(0)
        max_len = enc_outputs.size(1)
        attn_mask = sequence_mask(lengths, max_len).eq(0)

        if self.with_bridge:
            enc_hidden = self.bridge(enc_hidden)

        # insert dialog memory
        if self.dialog_state_memory is None:
            assert self.dialog_history_memory is None
            assert self.history_index is None
            assert self.memory_masks is None
            self.dialog_state_memory = enc_outputs
            self.dialog_history_memory = enc_outputs
            self.history_index = inputs
            self.memory_masks = attn_mask
        else:
            batch_state_memory = self.dialog_state_memory[:batch_size, :, :]
            self.dialog_state_memory = torch.cat([batch_state_memory, enc_outputs], dim=1)
            batch_history_memory = self.dialog_history_memory[:batch_size, :, :]
            self.dialog_history_memory = torch.cat([batch_history_memory, enc_outputs], dim=1)
            batch_history_index = self.history_index[:batch_size, :]
            self.history_index = torch.cat([batch_history_index, inputs], dim=-1)
            batch_memory_masks = self.memory_masks[:batch_size, :]
            self.memory_masks = torch.cat([batch_memory_masks, attn_mask], dim=-1)

        batch_kb_inputs = self.kbs[:batch_size, :, :]
        batch_kb_state_memory = self.kb_state_memory[:batch_size, :, :]
        batch_kb_slot_memory = self.kb_slot_memory[:batch_size, :, :]
        batch_kb_slot_index = self.kb_slot_index[:batch_size, :]
        kb_mask = self.kb_mask[:batch_size, :]
        selector_mask = self.selector_mask[:batch_size, :]

        selector = self.decoder.initialize_kb_v2(enc_hidden=enc_hidden, kb_state_memory=batch_kb_state_memory,
                                                attn_kb_mask=kb_mask)
        # kb_memory, selector = self.decoder.initialize_kb_v3(kb_inputs=batch_kb_inputs, enc_hidden=enc_hidden)
        kb_memory=None
        dec_init_state = self.decoder.initialize_state(
            hidden=enc_hidden,
            state_memory=self.dialog_state_memory,
            history_memory=self.dialog_history_memory,
            kb_memory=kb_memory,
            kb_state_memory=batch_kb_state_memory,
            kb_slot_memory=batch_kb_slot_memory,
            history_index=self.history_index,
            kb_slot_index=batch_kb_slot_index,
            attn_mask=self.memory_masks,
            attn_kb_mask=kb_mask,
            selector=selector,
            selector_mask=selector_mask
        )

        return outputs, dec_init_state

    def decode(self, dec_inputs, state):
        """
        decode
        """
        prob, attn_prob, kb_prob, p_gen, p_con, state = self.decoder.decode(dec_inputs, state)

        # logits copy from dialog history
        batch_size, max_len, word_size = prob.size()
        copy_index = state.history_index.unsqueeze(1).expand_as(
            attn_prob).contiguous().view(batch_size, max_len, -1)
        copy_logits = attn_prob.new_zeros(size=(batch_size, max_len, word_size),
                                          dtype=torch.float)
        copy_logits = copy_logits.scatter_add(dim=2, index=copy_index, src=attn_prob)

        # logits copy from kb
        index = state.kb_slot_index[:batch_size, :].unsqueeze(1).expand_as(
            kb_prob).contiguous().view(batch_size, max_len, -1)
        kb_logits = kb_prob.new_zeros(size=(batch_size, max_len, word_size),
                                      dtype=torch.float)
        kb_logits = kb_logits.scatter_add(dim=2, index=index, src=kb_prob)

        # compute final distribution
        #copy_prob = p_con * copy_logits + (1 - p_con) * kb_logits
        #logits = p_gen * prob + (1 - p_gen) * copy_prob
        con_logits = p_gen * prob + (1 - p_gen) * copy_logits
        logits = p_con * kb_logits + (1 - p_con) * con_logits
        log_logits = torch.log(logits + 1e-12)

        return log_logits, state

    def forward(self, enc_inputs, dec_inputs, hidden=None):
        """
        forward

        """
        outputs, dec_init_state = self.encode(enc_inputs, hidden)
        prob, attn_prob, kb_prob, p_gen, p_con, dec_state = self.decoder(dec_inputs, dec_init_state)

        # logits copy from dialog history
        batch_size, max_len, word_size = prob.size()
        copy_index = dec_init_state.history_index.unsqueeze(1).expand_as(
            attn_prob).contiguous().view(batch_size, max_len, -1)
        copy_logits = attn_prob.new_zeros(size=(batch_size, max_len, word_size),
                                          dtype=torch.float)
        copy_logits = copy_logits.scatter_add(dim=2, index=copy_index, src=attn_prob)

        # logits copy from kb
        index = dec_init_state.kb_slot_index[:batch_size, :].unsqueeze(1).expand_as(
            kb_prob).contiguous().view(batch_size, max_len, -1)
        kb_logits = kb_prob.new_zeros(size=(batch_size, max_len, word_size),
                                      dtype=torch.float)
        kb_logits = kb_logits.scatter_add(dim=2, index=index, src=kb_prob)

        # compute final distribution
        #copy_prob = p_con * copy_logits + (1-p_con) * kb_logits
        #logits = p_gen * prob + (1-p_gen) * copy_prob
        con_logits = p_gen * prob + (1-p_gen) * copy_logits
        logits = p_con * kb_logits + (1 - p_con) * con_logits
        log_logits = torch.log(logits + 1e-12)

        gate_logits = p_con.squeeze(-1)
        selector_logits = dec_init_state.selector
        selector_mask = dec_init_state.selector_mask

        outputs.add(logits=log_logits, gate_logits=gate_logits,
                    selector_logits=selector_logits, selector_mask=selector_mask,
                    dialog_state_memory=dec_state.state_memory,
                    kb_state_memory=dec_state.kb_state_memory,
                    prob=logits)
        return outputs

    def collect_metrics(self, outputs, target, ptr_index, kb_index):
        """
        collect_metrics
        """
        num_samples = target.size(0)
        metrics = Pack(num_samples=num_samples)
        loss = 0

        # loss for generation
        logits = outputs.logits
        nll = self.nll_loss(logits, target)
        loss += nll

        '''
        # loss for gate
        pad_zeros = torch.zeros([num_samples, 1], dtype=torch.long)
        if self.use_gpu:
            pad_zeros = pad_zeros.cuda()
        ptr_index = torch.cat([ptr_index, pad_zeros], dim=-1).float()
        gate_logits = outputs.gate_logits
        loss_gate = self.bce_loss(gate_logits, ptr_index)
        loss += loss_gate
        '''
        # loss for selector
        # selector_target = kb_index.float()
        # selector_logits = outputs.selector_logits
        # selector_mask = outputs.selector_mask
        #
        # if selector_target.size(-1) < selector_logits.size(-1):
        #     pad_zeros = torch.zeros(size=(num_samples, selector_logits.size(-1)-selector_target.size(-1)),
        #                             dtype=torch.float)
        #     if self.use_gpu:
        #         pad_zeros = pad_zeros.cuda()
        #     selector_target = torch.cat([selector_target, pad_zeros], dim=-1)
        # loss_ptr = self.bce_loss(selector_logits, selector_target, mask=selector_mask)
        loss_ptr = torch.tensor(0.0)
        if self.use_gpu:
            loss_ptr = loss_ptr.cuda()
        loss += loss_ptr

        acc = accuracy(logits, target, padding_idx=self.padding_idx)
        metrics.add(loss=loss, ptr=loss_ptr, acc=acc, logits=logits, prob=outputs.prob)

        return metrics

    def update_memory(self, dialog_state_memory, kb_state_memory):
        self.dialog_state_memory = dialog_state_memory
        self.kb_state_memory = kb_state_memory

    def iterate(self, turn_inputs, kb_inputs,
                optimizer=None, grad_clip=None, use_rl=False, is_training=True):
        """
        iterate
        """
        self.reset_memory()

        self.load_kb_memory(kb_inputs)

        metrics_list = []
        total_loss = 0

        for i, inputs in enumerate(turn_inputs):
            if self.use_gpu:
                inputs = inputs.cuda()
            src, src_lengths = inputs.src
            tgt, tgt_lengths = inputs.tgt
            task_label = inputs.task
            gold_entity = inputs.gold_entity
            ptr_index, ptr_lengths = inputs.ptr_index
            kb_index, kb_index_lengths = inputs.kb_index
            enc_inputs = src[:, 1:-1], src_lengths - 2  # filter <bos> <eos>
            dec_inputs = tgt[:, :-1], tgt_lengths - 1  # filter <eos>
            target = tgt[:, 1:]  # filter <bos>
            target_mask = sequence_mask(tgt_lengths - 1)

            if use_rl:
                sample_outputs = self.sample(enc_inputs, dec_inputs, random_sample=True)
                with torch.no_grad():
                    greedy_outputs = self.sample(enc_inputs, dec_inputs, random_sample=False)
                    outputs = self.forward(enc_inputs, dec_inputs)
                metrics = self.collect_rl_metrics(sample_outputs, greedy_outputs, target,
                                                  gold_entity, ptr_index, kb_index, target_mask, task_label)
            else:
                outputs = self.forward(enc_inputs, dec_inputs)
                metrics = self.collect_metrics(outputs, target, ptr_index, kb_index)

            metrics_list.append(metrics)
            total_loss += metrics.loss

            # self.update_memory(dialog_state_memory=outputs.dialog_state_memory,
            #                    kb_state_memory=outputs.kb_state_memory)

        if torch.isnan(total_loss):
            raise ValueError("NAN loss encountered!")

        if is_training:
            assert optimizer is not None
            optimizer.zero_grad()
            total_loss.backward()
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(parameters=self.parameters(), max_norm=grad_clip)
            optimizer.step()

        return metrics_list
