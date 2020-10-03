#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File: source/module/attention.py
"""

import torch
import torch.nn as nn


class Attention(nn.Module):
    """
    Attention
    """
    def __init__(self,
                 max_hop,
                 query_size,
                 memory_size=None,
                 hidden_size=None,
                 num_layers=1,
                 dropout=0.0,
                 mode="mlp",
                 project=False):
        super(Attention, self).__init__()
        assert (mode in ["dot", "general", "mlp"]), (
            "Unsupported attention mode: {mode}"
        )

        self.max_hop = max_hop
        self.query_size = query_size
        self.memory_size = memory_size or query_size
        self.hidden_size = hidden_size or query_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.mode = mode
        self.project = project

        self.rnn_input_size = self.query_size + self.memory_size

        self.rnn = nn.GRU(input_size=self.rnn_input_size,
                          hidden_size=self.hidden_size,
                          num_layers=self.num_layers,
                          dropout=self.dropout if self.num_layers > 1 else 0,
                          batch_first=True)
        map(nn.init.orthogonal_, self.rnn.all_weights)

        if mode == "general":
            self.linear_query = nn.ModuleList([nn.Linear(self.query_size, self.memory_size, bias=False)
                                               for _ in range(self.max_hop)])
        elif mode == "mlp":
            self.linear_query = nn.ModuleList([nn.Linear(self.query_size, self.hidden_size, bias=True)
                                               for _ in range(self.max_hop)])
            self.linear_memory = nn.ModuleList([nn.Linear(self.memory_size, self.hidden_size, bias=False)
                                                for _ in range(self.max_hop)])
            self.v = nn.ModuleList([nn.Linear(self.hidden_size, 1, bias=False)
                                    for _ in range(self.max_hop)])
            self.tanh = nn.Tanh()

        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
        self.linear_forget = nn.ModuleList([nn.Linear(self.query_size, self.memory_size, bias=False)
                                            for _ in range(self.max_hop)])
        self.linear_add = nn.ModuleList([nn.Linear(self.query_size, self.memory_size, bias=False)
                                         for _ in range(self.max_hop)])

        if self.project:
            self.linear_project = nn.Sequential(
                nn.Linear(in_features=self.hidden_size + self.memory_size,
                          out_features=self.hidden_size),
                nn.Tanh())

    def memory_address(self, query, key_memory, hop, mask=None):
        """
        query: Tensor(batch_size, query_length, query_size)
        key_memory: Tensor(batch_size, memory_length, memory_size)
        hop: int
        mask: Tensor(batch_size, memory_length)
        """
        if self.mode == "dot":
            assert query.size(-1) == key_memory.size(-1)
            attn = torch.bmm(query, key_memory.transpose(1, 2))  # (batch_size, query_length, memory_length)
        elif self.mode == "general":
            assert self.memory_size == key_memory.size(-1)
            key = self.linear_query[hop](query)  # (batch_size, query_length, memory_size)
            attn = torch.bmm(key, key_memory.transpose(1, 2))  # (batch_size, query_length, memory_length)
        else:
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

    def forward(self, query, key_memory, value_memory, hidden, mask=None):
        """
        query: Tensor(batch_size, query_length, query_size)
        key_memory: Tensor(batch_size, memory_length, memory_size)
        value_memory: Tensor(batch_size, memory_length, memory_size)
        mask: Tensor(batch_size, memory_length)
        """
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, query.size(1), 1)  # (batch_size, query_length, memory_length)

        for hop in range(self.max_hop):
            weights = self.memory_address(query, key_memory, hop, mask=mask)
            weighted_context = torch.bmm(weights, value_memory)  # (batch_size, query_length, memory_size)

            # get intermediate hidden state
            rnn_input = torch.cat([weighted_context, query], dim=-1)
            rnn_output, new_hidden = self.rnn(rnn_input, hidden)
            new_query = new_hidden[-1].unsqueeze(1)

            # key memory update
            key_memory = self.memory_update(new_query, key_memory, hop, mask=mask)

        final_weighted_context = weighted_context
        final_weights = weights
        final_key_memory = key_memory

        if self.project:
            project_output = self.linear_project(torch.cat([final_weighted_context, query], dim=-1))
            return project_output, final_weights, final_key_memory
        else:
            return final_weighted_context, final_weights, final_key_memory
