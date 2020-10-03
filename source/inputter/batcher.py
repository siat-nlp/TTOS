#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
File: source/inputter/batcher.py
"""
import numpy as np
from torch.utils.data import Dataset

from source.utils.misc import Pack
from source.utils.misc import list2tensor


class DialogDataset(Dataset):
    """
    DialogDataset
    """
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class DialogBatcher(object):
    """
    DialogBatcher
    """

    def __init__(self, batch_size, data_type="train", shuffle=False):
        self.batch_size = batch_size
        self.data_type = data_type
        self.shuffle = shuffle

        self.batch_data_list = []
        self.batch_size_list = []
        self.n_batch = None  # number of batches
        self.n_rows = None  # number of samples

    def __len__(self):
        return self.n_rows

    def prepare_epoch(self):
        if self.shuffle:
            np.random.shuffle(self.batch_data_list)

    def get_batch(self, batch_idx):
        local_data = self.batch_data_list[batch_idx]
        #local_size = self.batch_size_list[batch_idx]

        batch_data = self.create_batches(local_data)

        return batch_data

    def prepare_input_list(self, input_data_list):
        if self.shuffle:
            np.random.shuffle(input_data_list)

        self.n_rows = remain_rows = len(input_data_list)
        while remain_rows > 0:
            self.batch_data_list.append({})
            active_size = min(remain_rows, self.batch_size)
            self.batch_size_list.append(active_size)
            remain_rows -= active_size
        self.n_batch = len(self.batch_size_list)

        for batch_idx in range(self.n_batch):
            st_idx = batch_idx * self.batch_size
            ed_idx = st_idx + self.batch_size
            local_batch_input = input_data_list[st_idx: ed_idx]
            self.batch_data_list[batch_idx] = local_batch_input

        print('n_rows = %d, batch_size = %d, n_batch = %d.' % (self.n_rows, self.batch_size, self.n_batch))

    def create_batches(self, data):
        # sort by dialog turns
        sorted_data = sorted(data, key=lambda x: x['turn'], reverse=True)

        tasks = [sample['task'] for sample in sorted_data]
        turns = [sample['turn'] for sample in sorted_data]
        kbs = [sample['kb'] for sample in sorted_data]
        max_turn = max(turns)
        inputs = []
        for t in range(max_turn):
            turn_label = []
            turn_src = []
            turn_tgt = []
            turn_entity = []
            turn_ptr = []
            turn_kb_ptr = []
            for sample in sorted_data:
                if sample['turn'] >= t+1:
                    turn_label.append(t+1)
                    turn_src.append(sample['src'][t])
                    turn_tgt.append(sample['tgt'][t])
                    turn_entity.append(sample['gold_entity'][t])
                    turn_ptr.append(sample['ptr_index'][t])
                    turn_kb_ptr.append(sample['kb_index'][t])

            turn_batch_size = len(turn_src)
            task = tasks[:turn_batch_size]
            assert len(turn_tgt) == turn_batch_size
            if self.data_type == "test":
                kb = kbs[:turn_batch_size]
                turn_input = {"turn_label": turn_label,
                              "src": turn_src,
                              "tgt": turn_tgt,
                              "task": task,
                              "gold_entity": turn_entity,
                              "ptr_index": turn_ptr,
                              "kb_index": turn_kb_ptr,
                              "kb": kb
                              }
            else:
                # turn_input = {"src": turn_src,
                #               "tgt": turn_tgt,
                #               "tgt_e": turn_tgt_e,
                #               "tgt_b": turn_tgt_b,
                #               "task": task,
                #               "gold_entity": turn_entity,
                #               "ptr_index": turn_ptr,
                #               "kb_index": turn_kb_ptr
                #               }
                # TODO: we need kb to compute f1_score during the training
                kb = kbs[:turn_batch_size]
                turn_input = {"turn_label": turn_label,
                              "src": turn_src,
                              "tgt": turn_tgt,
                              "task": task,
                              "gold_entity": turn_entity,
                              "ptr_index": turn_ptr,
                              "kb_index": turn_kb_ptr,
                              "kb": kb
                              }
            inputs.append(turn_input)
        batch_data = {"tasks": tasks,
                      "max_turn": max_turn,
                      "inputs": inputs,
                      "kbs": kbs
                      }
        return batch_data


def create_turn_batch(data_list):
    """
    create_turn_batch
    """
    turn_batches = []
    for data_dict in data_list:
        batch = Pack()
        for key in data_dict.keys():
            if key in ['src', 'tgt', 'tgt_e', 'tgt_b', 'ptr_index', 'kb_index']:
                batch[key] = list2tensor([x for x in data_dict[key]])
            else:
                batch[key] = data_dict[key]
        turn_batches.append(batch)

    return turn_batches


def create_kb_batch(kb_list):
    """
    create_kb_batch
    """
    kb_batches = list2tensor(kb_list)
    return kb_batches
