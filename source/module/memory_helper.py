#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File: source/module/memory_helper.py
"""

import torch
import torch.nn as nn


class KnowledgeMemory(nn.Module):
    def __init__(self,
                 query_size,
                 memory_size,
                 hidden_size,
                 max_hop=1,
                 num_layers=1,
                 dropout=0.0,
                 use_gpu=False):
        super(KnowledgeMemory, self).__init__()
        self.query_size = query_size
        self.memory_size = memory_size
        self.hidden_size = hidden_size
        self.max_hop = max_hop
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_gpu = use_gpu

        self.rnn_input_size = self.query_size + self.memory_size

        self.rnn = nn.GRU(input_size=self.rnn_input_size,
                          hidden_size=self.hidden_size,
                          num_layers=self.num_layers,
                          dropout=self.dropout if self.num_layers > 1 else 0,
                          batch_first=True)
        map(nn.init.orthogonal_, self.rnn.all_weights)

        self.linear_query = nn.ModuleList([nn.Linear(self.query_size, self.query_size, bias=True)
                                           for _ in range(self.max_hop)])
        self.linear_memory = nn.ModuleList([nn.Linear(self.memory_size, self.query_size, bias=False)
                                            for _ in range(self.max_hop)])
        self.v = nn.ModuleList([nn.Linear(self.query_size, 1, bias=False)
                                for _ in range(self.max_hop)])
        self.tanh = nn.Tanh()

        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
        self.linear_forget = nn.ModuleList([nn.Linear(self.query_size, self.memory_size, bias=False)
                                            for _ in range(self.max_hop)])
        self.linear_add = nn.ModuleList([nn.Linear(self.query_size, self.memory_size, bias=False)
                                         for _ in range(self.max_hop)])

    def memory_address(self, query, key_memory, hop, mask=None):
        # (batch_size, query_length, memory_length, hidden_size)
        hidden_sum = self.linear_query[hop](query).unsqueeze(2) + \
                     self.linear_memory[hop](key_memory).unsqueeze(1)
        key = self.tanh(hidden_sum)
        attn = self.v[hop](key).squeeze(-1)  # (batch_size, query_length, memory_length)
        if mask is not None:
            attn.masked_fill_(mask, -float("inf"))

        weights = self.softmax(attn)  # (batch_size, query_length, memory_length)
        return weights

    def memory_update(self, query, key_memory, hop, mask=None):
        """
        query: Tensor(batch_size, query_length, query_size)
        key_memory: Tensor(batch_size, memory_length, memory_size)
        hop: int
        mask: Tensor(batch_size, memory_length)
        """
        weights = self.memory_address(query, key_memory, hop, mask=mask)  # (batch_size, query_length, memory_length)
        forget = self.linear_forget[hop](query)  # (batch_size, query_length, memory_size)
        forget_weights = self.sigmoid(forget)
        forget_memory = torch.bmm(weights.transpose(1, 2), forget_weights)  # (batch_size, memory_length, memory_size)
        temp_memory = key_memory * (1 - forget_memory)

        add = self.linear_add[hop](query)  # (batch_size, query_length, memory_size)
        add_weights = self.sigmoid(add)
        add_memory = torch.bmm(weights.transpose(1, 2), add_weights)  # (batch_size, memory_length, memory_size)
        final_memory = temp_memory + add_memory

        return final_memory

    def forward(self, query, kb_state_memory, kb_slot_memory, hidden, mask=None):
        """
        query: Tensor(batch_size, query_length, query_size)
        mask: Tensor(batch_size, memory_length)
        """
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, query.size(1), 1)  # (batch_size, query_length, memory_length)

        for hop in range(self.max_hop):
            weights = self.memory_address(query, kb_state_memory, hop, mask=mask)
            weighted_kb = torch.bmm(weights, kb_slot_memory)  # (batch_size, query_length, memory_size)

            # get intermediate hidden state
            rnn_input = torch.cat([weighted_kb, query], dim=-1)
            rnn_output, new_hidden = self.rnn(rnn_input, hidden)
            new_query = new_hidden[-1].unsqueeze(1)

            # key memory update
            kb_state_memory = self.memory_update(new_query, kb_state_memory, hop, mask=mask)

        final_weighted_kb = weighted_kb
        final_weights = weights
        final_kb_memory = kb_state_memory

        return final_weighted_kb, final_weights, final_kb_memory
