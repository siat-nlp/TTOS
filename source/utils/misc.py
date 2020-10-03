#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File: source/utils/misc.py
"""

import torch
import argparse


class Pack(dict):
    """
    Pack
    """
    def __getattr__(self, name):
        return self.get(name)

    def add(self, **kwargs):
        """
        add
        """
        for k, v in kwargs.items():
            self[k] = v

    def flatten(self):
        """
        flatten
        """
        pack_list = []
        for vs in zip(*self.values()):
            pack = Pack(zip(self.keys(), vs))
            pack_list.append(pack)
        return pack_list

    def cuda(self, device=None):
        """
        cuda
        """
        pack = Pack()
        for k, v in self.items():
            if k in ['src', 'tgt', 'tgt_b', 'tgt_e', 'ptr_index', 'kb_index']:
                if isinstance(v, tuple):
                    pack[k] = tuple(x.cuda(device) for x in v)
                else:
                    pack[k] = v.cuda(device)
            else:
                pack[k] = v
        return pack


def sequence_mask(lengths, max_len=None):
    """
    Creates a boolean mask from sequence lengths.
    """
    if max_len is None:
        max_len = lengths.max().item()
    mask = torch.arange(0, max_len, dtype=torch.long).type_as(lengths)
    mask = mask.unsqueeze(0)
    mask = mask.repeat(1, *lengths.size(), 1)
    mask = mask.squeeze(0)
    mask = mask.lt(lengths.unsqueeze(-1))
    return mask


def sequence_kd_mask(lengths, target, name, ent_idx, nen_idx, max_len=None):
    """
    Creates a boolean mask from sequence lengths/target/name.
    """
    if max_len is None:
        max_len = lengths.max().item()
    mask = torch.arange(0, max_len, dtype=torch.long).type_as(lengths)
    mask = mask.unsqueeze(0)
    mask = mask.repeat(1, *lengths.size(), 1)
    mask = mask.squeeze(0)
    mask = mask.lt(lengths.unsqueeze(-1))

    if name == 'model_TB':
        mask_b = (target == ent_idx)
        mask = mask.masked_fill(mask_b, False)
        return mask
    elif name == 'model_TE':
        mask_b = (target == nen_idx)
        mask = mask.masked_fill(mask_b, False)
        return mask
    else:
        return mask


def max_lens(X):
    """
    max_lens
    """
    if not isinstance(X[0], list):
        return [len(X)]
    elif not isinstance(X[0][0], list):
        return [len(X), max(len(x) for x in X)]
    elif not isinstance(X[0][0][0], list):
        return [len(X), max(len(x) for x in X),
                max(len(x) for xs in X for x in xs)]
    else:
        raise ValueError(
            "Data list whose dim is greater than 3 is not supported!")


def list2tensor(X):
    """
    list2tensor
    """
    size = max_lens(X)
    if len(size) == 1:
        tensor = torch.tensor(X)
        return tensor

    tensor = torch.zeros(size, dtype=torch.long)
    lengths = torch.zeros(size[:-1], dtype=torch.long)
    if len(size) == 2:
        for i, x in enumerate(X):
            l = len(x)
            tensor[i, :l] = torch.tensor(x)
            lengths[i] = l
    else:
        for i, xs in enumerate(X):
            for j, x in enumerate(xs):
                l = len(x)
                tensor[i, j, :l] = torch.tensor(x)
                lengths[i, j] = l

    return tensor, lengths


def one_hot(indice, num_classes):
    """
    one_hot
    """
    I = torch.eye(num_classes).to(indice.device)
    T = I[indice]
    return T


def str2bool(v):
    """
    str2bool
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def compute_lamdba(metric_s, metric_t):
    res = 0.5 + max(-0.5, min(0.5, (metric_s - metric_t) / 10))
    return res


def close_train(models):
    if isinstance(models, tuple):
        for model in models:
            for _, para in model.named_parameters():
                para.requires_grad=False
    else:
        for _, para in models.named_parameters():
            para.requires_grad = False


if __name__ == '__main__':
    # X = [1, 2, 3]
    # print(X)
    # print(list2tensor(X))
    # X = [X, [2, 3]]
    # print(X)
    # print(list2tensor(X))
    # X = [X, [[1, 1, 1, 1, 1]]]
    # print(X)
    # print(list2tensor(X))
    #
    # data_list = [{'src': [1, 2, 3], 'tgt': [1, 2, 3, 4]},
    #              {'src': [2, 3], 'tgt': [1, 2, 4]}]
    # batch = Pack()
    # for key in data_list[0].keys():
    #     batch[key] = list2tensor([x[key] for x in data_list])
    #
    # print(batch)
    # print(batch.src)
    bleu_s, bleu_t = 42.03, 48.81
    lambda_s_t = compute_lamdba(bleu_s, bleu_t)
    print(lambda_s_t)