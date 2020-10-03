#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File: source/module/memory_helper.py
"""

import torch
import torch.nn as nn


class AttrProxy(object):
    """
    Translates index lookups into attribute lookups.
    To implement some trick which able to use list of nn.Module in a nn.Module
    see https://discuss.pytorch.org/t/list-of-nn-module-in-a-nn-module/219/2
    """

    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))


class KnowledgeMemoryv3(nn.Module):
    def __init__(self,
                 vocab_size,
                 query_size,
                 memory_size,
                 max_hop=1,
                 dropout=0.0,
                 padding_idx=None,
                 mode="mlp",
                 use_gpu=False):
        super(KnowledgeMemoryv3, self).__init__()
        assert (mode in ["general", "mlp"]), (
            "Unsupported attention mode: {mode}"
        )
        self.vocab_size = vocab_size
        self.query_size = query_size
        self.memory_size = memory_size
        self.max_hop = max_hop
        self.dropout_layer = nn.Dropout(dropout)
        self.padding_idx = padding_idx
        self.mode = mode
        self.use_gpu = use_gpu

        if self.mode == "general":
            self.linear_query = nn.ModuleList([nn.Linear(self.query_size, self.memory_size, bias=False)
                                               for _ in range(self.max_hop)])
        elif self.mode == "mlp":
            self.linear_query = nn.ModuleList([nn.Linear(self.query_size, self.memory_size, bias=True)
                                               for _ in range(self.max_hop)])
            self.linear_memory = nn.ModuleList([nn.Linear(self.memory_size, self.memory_size, bias=False)
                                                for _ in range(self.max_hop)])
            self.v = nn.ModuleList([nn.Linear(self.memory_size, 1, bias=False)
                                    for _ in range(self.max_hop)])
            self.tanh = nn.Tanh()

        for hop in range(self.max_hop + 1):
            C = nn.Embedding(num_embeddings=self.vocab_size,
                             embedding_dim=self.memory_size,
                             padding_idx=self.padding_idx)
            C.weight.data.normal_(0, 0.1)
            self.add_module("C_{}".format(hop), C)
        self.C = AttrProxy(self, "C_")
        # TODO: change KB to MLP transform
        #self.trans_layer = nn.ModuleList([nn.Linear(3 * self.memory_size, self.memory_size, bias=True)
        #                                  for _ in range(self.max_hop + 1)])

        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()

    def load_memory(self, kb_inputs, enc_hidden):
        kb_memory_list = []
        q = [enc_hidden.squeeze(0)]
        batch_size, kb_num = kb_inputs.size(0), kb_inputs.size(1)

        for hop in range(self.max_hop):
            embed_state = self.C[hop](kb_inputs)
            embed_state = torch.sum(embed_state, dim=2)  # (batch_size, kb_num, memory_size)
            #embed_state = torch.mean(embed_state, dim=2)  # (batch_size, kb_num, memory_size)
            #embed_state = self.trans_layer[hop](embed_state.contiguous().view(batch_size, kb_num, -1))

            q_state = q[-1].unsqueeze(1).expand_as(embed_state)
            prob_logit = torch.sum(embed_state * q_state, dim=-1)  # (batch_size, kb_num)
            attn_ = self.softmax(prob_logit)

            embed_state_next = self.C[hop+1](kb_inputs)
            embed_state_next = torch.sum(embed_state_next, dim=2)  # (batch_size, kb_num, memory_size)
            #embed_state_next = torch.mean(embed_state_next, dim=2)  # (batch_size, kb_num, memory_size)
            #embed_state_next = self.trans_layer[hop+1](embed_state_next.contiguous().view(batch_size, kb_num, -1))

            attn = attn_.unsqueeze(2).expand_as(embed_state_next)
            o_k = torch.sum(attn * embed_state_next, dim=1)
            q_k = q[-1] + o_k
            q.append(q_k)
            kb_memory_list.append(embed_state)
        kb_memory_list.append(embed_state_next)
        final_kb_memory = enc_hidden.new_zeros(
            size=(len(kb_memory_list), embed_state.size(0), embed_state.size(1), embed_state.size(2)),
            dtype=embed_state.dtype
        )
        for i, kb_memory in enumerate(kb_memory_list):
            final_kb_memory[i, :, :, :] = kb_memory
        final_kb_memory = final_kb_memory.transpose(0, 1)

        selector = self.sigmoid(prob_logit)   # (batch_size, kb_num)
        return final_kb_memory, selector

    def memory_address(self, query, key_memory, hop, mask=None):
        if self.mode == "general":
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

    def forward(self, query, kb_memory_db, selector=None, mask=None):
        """
        query: Tensor(batch_size, query_length, query_size)
        selector: Tensor(batch_size, memory_length)
        mask: Tensor(batch_size, memory_length)
        """
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, query.size(1), 1)  # (batch_size, query_length, memory_length)

        q = [query]
        batch_size = query.size(0)
        kb_memory_list = kb_memory_db.transpose(0, 1)

        for hop in range(self.max_hop):
            kb_memory = kb_memory_list[hop]
            kb_memory = kb_memory[:batch_size, :, :]
            if selector is not None:
                kb_memory = kb_memory * selector.unsqueeze(2).expand_as(kb_memory)

            q_temp = q[-1]
            attn_weights = self.memory_address(q_temp, kb_memory, hop, mask=mask)

            kb_memory_next = kb_memory_list[hop+1]
            kb_memory_next = kb_memory_next[:batch_size, :, :]
            if selector is not None:
                kb_memory_next = kb_memory_next * selector.unsqueeze(2).expand_as(kb_memory_next)

            o_k = torch.bmm(attn_weights, kb_memory_next)  # (batch_size, query_length, memory_size)
            q_k = q[-1] + o_k
            q.append(q_k)

        final_weights = attn_weights
        final_weighted_kb = o_k
        return final_weighted_kb, final_weights
