#!/usr/bin/env python
"""
File: source/module/rnn_decoder.py
"""

import torch
import torch.nn as nn

from source.module.attention import Attention
from source.module.memory_helper import KnowledgeMemory
from source.module.memory_helper_v2 import KnowledgeMemoryv2
from source.module.memory_helper_v3 import KnowledgeMemoryv3
from source.module.decoder_state import DecoderState
from source.utils.misc import sequence_mask


class RNNDecoder(nn.Module):
    """
    A GRU recurrent neural network decoder.
    """
    def __init__(self,
                 embedder,
                 max_hop,
                 input_size,
                 hidden_size,
                 output_size,
                 kb_output_size,
                 num_layers=1,
                 attn_mode="mlp",
                 memory_size=None,
                 kb_memory_size=None,
                 dropout=0.0,
                 padding_idx=None,
                 use_gpu=False):
        super(RNNDecoder, self).__init__()

        self.embedder = embedder
        self.max_hop = max_hop
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.kb_output_size = kb_output_size
        self.num_layers = num_layers
        self.attn_mode = attn_mode
        self.memory_size = memory_size
        self.kb_memory_size = kb_memory_size
        self.dropout = dropout
        self.padding_idx = padding_idx
        self.use_gpu = use_gpu

        self.rnn_input_size = self.input_size
        self.out_input_size = self.hidden_size

        self.rnn_input_size += self.memory_size
        self.out_input_size += self.memory_size
        self.attention = Attention(max_hop=self.max_hop,
                                   query_size=self.hidden_size,
                                   memory_size=self.memory_size,
                                   hidden_size=self.hidden_size,
                                   num_layers=self.num_layers,
                                   dropout=self.dropout if self.num_layers > 1 else 0,
                                   mode=self.attn_mode,
                                   project=False)
        '''
        self.kb_memory = KnowledgeMemory(query_size=self.hidden_size,
                                         memory_size=self.kb_memory_size,
                                         hidden_size=self.hidden_size,
                                         max_hop=self.max_hop,
                                         num_layers=self.num_layers,
                                         dropout=self.dropout,
                                         use_gpu=self.use_gpu)
                                         
        self.kb_memory_v3 = KnowledgeMemoryv3(vocab_size=self.kb_output_size,
                                              query_size=self.hidden_size,
                                              memory_size=self.memory_size,
                                              max_hop=self.max_hop,
                                              padding_idx=self.padding_idx,
                                              use_gpu=self.use_gpu)
        '''

        self.kb_memory_v2 = KnowledgeMemoryv2(query_size=self.hidden_size,
                                              memory_size=self.kb_memory_size,
                                              hidden_size=self.hidden_size,
                                              max_hop=self.max_hop,
                                              num_layers=self.num_layers,
                                              dropout=self.dropout,
                                              use_gpu=self.use_gpu)

        # TODO: change kb into rn input
        # self.rnn_input_size += self.kb_memory_size
        # self.out_input_size += self.kb_memory_size

        self.rnn = nn.GRU(input_size=self.rnn_input_size,
                          hidden_size=self.hidden_size,
                          num_layers=self.num_layers,
                          dropout=self.dropout if self.num_layers > 1 else 0,
                          batch_first=True)
        map(nn.init.orthogonal_, self.rnn.all_weights)

        self.gate_layer = nn.Sequential(
            nn.Linear(self.out_input_size, 1, bias=True),
            nn.Sigmoid()
        )
        self.copy_gate_layer = nn.Sequential(
            nn.Linear(self.out_input_size, 1, bias=True),
            nn.Sigmoid()
        )

        if self.out_input_size > self.hidden_size:
            self.output_layer = nn.Sequential(
                nn.Dropout(p=self.dropout),
                nn.Linear(self.out_input_size, self.hidden_size),
                nn.Linear(self.hidden_size, self.output_size),
                nn.Softmax(dim=-1),
            )
        else:
            self.output_layer = nn.Sequential(
                nn.Dropout(p=self.dropout),
                nn.Linear(self.out_input_size, self.output_size),
                nn.Softmax(dim=-1),
            )

    def initialize_kb_v2(self, enc_hidden, kb_state_memory, attn_kb_mask):
        selector = self.kb_memory_v2.memory_point(enc_hidden, kb_state_memory, mask=attn_kb_mask)
        return selector

    def initialize_kb_v3(self, kb_inputs, enc_hidden):
        kb_memory, selector = self.kb_memory_v3.load_memory(kb_inputs, enc_hidden)
        return kb_memory, selector

    @staticmethod
    def initialize_state(hidden,
                         state_memory=None,
                         history_memory=None,
                         kb_memory=None,
                         kb_state_memory=None,
                         kb_slot_memory=None,
                         history_index=None,
                         kb_slot_index=None,
                         attn_mask=None,
                         attn_kb_mask=None,
                         selector=None,
                         selector_mask=None):
        """
        initialize_state
        """
        init_state = DecoderState(
            hidden=hidden,
            state_memory=state_memory,
            history_memory=history_memory,
            kb_memory=kb_memory,
            kb_state_memory=kb_state_memory,
            kb_slot_memory=kb_slot_memory,
            history_index=history_index,
            kb_slot_index=kb_slot_index,
            attn_mask=attn_mask,
            attn_kb_mask=attn_kb_mask,
            selector=selector,
            selector_mask=selector_mask
        )

        return init_state

    def decode(self, inputs, state, is_training=False):
        """
        decode
        """
        rnn_input_list = []
        out_input_list = []
        kb_input_list = []

        inputs = self.embedder(inputs)

        # shape: (batch_size, 1, input_size)
        inputs = inputs.unsqueeze(1)
        rnn_input_list.append(inputs)

        hidden = state.hidden
        query = hidden[-1].unsqueeze(1)

        weighted_context, attn, updated_memory = self.attention(query=query,
                                                                key_memory=state.state_memory.clone(),
                                                                value_memory=state.history_memory.clone(),
                                                                hidden=hidden,
                                                                mask=state.attn_mask)
        rnn_input_list.append(weighted_context)
        out_input_list.append(weighted_context)

        # generate from kb
        '''
        weighted_kb, kb_attn, updated_kb_memory = self.kb_memory(
            query=query,
            kb_state_memory=state.kb_state_memory.clone(),
            kb_slot_memory=state.kb_slot_memory.clone(),
            hidden=hidden,
            mask=state.attn_kb_mask)
        
        weighted_kb, kb_attn = self.kb_memory_v3(query=query,
                                                 kb_memory_db=state.kb_memory,
                                                 selector=state.selector,
                                                 mask=state.attn_kb_mask)
        '''
        weighted_kb, kb_attn, updated_kb_memory = self.kb_memory_v2(
            query=query,
            kb_state_memory=state.kb_state_memory.clone(),
            kb_slot_memory=state.kb_slot_memory.clone(),
            hidden=hidden,
            selector=None,
            mask=state.attn_kb_mask)

        kb_input_list.append(weighted_kb)

        # TODO: add kb context
        # rnn_input_list.append(weighted_kb)
        # out_input_list.append(weighted_kb)

        # state.state_memory = updated_memory.clone()
        # state.kb_state_memory = updated_kb_memory.clone()

        rnn_input = torch.cat(rnn_input_list, dim=-1)
        rnn_output, new_hidden = self.rnn(rnn_input, hidden)
        out_input_list.append(rnn_output)
        kb_input_list.append(rnn_output)
        out_input = torch.cat(out_input_list, dim=-1)
        kb_input = torch.cat(kb_input_list, dim=-1)
        state.hidden = new_hidden

        if is_training:
            return out_input, kb_input, attn, kb_attn, state
        else:
            prob = self.output_layer(out_input)
            p_gen = self.gate_layer(out_input)
            p_con = self.copy_gate_layer(kb_input)
            return prob, attn, kb_attn, p_gen, p_con, state

    def forward(self, dec_inputs, state):
        """
        forward
        """
        inputs, lengths = dec_inputs
        batch_size, max_len = inputs.size()

        out_inputs = inputs.new_zeros(
            size=(batch_size, max_len, self.out_input_size),
            dtype=torch.float)

        kb_inputs = inputs.new_zeros(
            size=(batch_size, max_len, self.out_input_size),
            dtype=torch.float)

        out_attn_size = state.history_memory.size(1)
        out_attn_probs = inputs.new_zeros(
            size=(batch_size, max_len, out_attn_size),
            dtype=torch.float)

        out_kb_size = state.kb_slot_memory.size(1)
        out_kb_probs = inputs.new_zeros(
            size=(batch_size, max_len, out_kb_size),
            dtype=torch.float)

        # sort by lengths
        sorted_lengths, indices = lengths.sort(descending=True)
        inputs = inputs.index_select(0, indices)
        state = state.index_select(indices)

        # number of valid inputs (i.e. not padding index) in each time step
        num_valid_list = sequence_mask(sorted_lengths).int().sum(dim=0)

        for i, num_valid in enumerate(num_valid_list):
            dec_input = inputs[:num_valid, i]
            valid_state = state.slice_select(num_valid)

            # decode for one step
            out_input, kb_input, attn, kb_attn, valid_state = self.decode(dec_input,
                                                                          valid_state,
                                                                          is_training=True)

            state.hidden[:, :num_valid] = valid_state.hidden
            state.state_memory[:num_valid, :, :] = valid_state.state_memory
            state.kb_state_memory[:num_valid, :, :] = valid_state.kb_state_memory

            out_inputs[:num_valid, i] = out_input.squeeze(1)
            kb_inputs[:num_valid, i] = kb_input.squeeze(1)
            out_attn_probs[:num_valid, i] = attn.squeeze(1)
            out_kb_probs[:num_valid, i] = kb_attn.squeeze(1)

        # Resort
        _, inv_indices = indices.sort()
        state = state.index_select(inv_indices)
        out_inputs = out_inputs.index_select(0, inv_indices)
        kb_inputs = kb_inputs.index_select(0, inv_indices)
        attn_probs = out_attn_probs.index_select(0, inv_indices)
        kb_probs = out_kb_probs.index_select(0, inv_indices)

        probs = self.output_layer(out_inputs)
        p_gen = self.gate_layer(out_inputs)
        p_con = self.copy_gate_layer(kb_inputs)

        return probs, attn_probs, kb_probs, p_gen, p_con, state
