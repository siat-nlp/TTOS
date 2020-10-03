#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File: source/module/embedder.py
"""

import torch
import torch.nn as nn


class Embedder(nn.Embedding):
    """
    Embedder
    """
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None):
        super(Embedder, self).__init__(num_embeddings=num_embeddings,
                                       embedding_dim=embedding_dim,
                                       padding_idx=padding_idx)

    def load_embeddings(self, embeds, scale=0.05):
        """
        load_embeddings
        """
        assert len(embeds) == self.num_embeddings

        embeds = torch.tensor(embeds)
        num_known = 0
        for i in range(len(embeds)):
            if len(embeds[i].nonzero()) == 0:
                nn.init.uniform_(embeds[i], -scale, scale)
            else:
                num_known += 1
        self.weight.data.copy_(embeds)
        print("{} words have pretrained embeddings".format(num_known),
              "(coverage: {:.3f})".format(num_known / self.num_embeddings))
