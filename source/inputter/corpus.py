#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
File: source/inputter/corpus.py
"""

import os
import torch
import json
from tqdm import tqdm
from source.inputter.field import EOS
from source.inputter.field import tokenize
from source.inputter.field import TextField
from source.inputter.batcher import DialogBatcher


class KnowledgeCorpus(object):
    """
    KnowledgeCorpus
    """
    def __init__(self,
                 data_dir,
                 min_freq=0,
                 max_vocab_size=None,
                 min_len=0,
                 max_len=400,
                 embed_file=None,
                 share_vocab=False,
                 special_tokens=None):

        self.data_dir = data_dir
        self.prepared_data_file = "%s/data.all.pt" % data_dir
        self.prepared_vocab_file = "%s/vocab.%d.pt" % (data_dir, max_vocab_size)
        self.min_freq = min_freq
        self.max_vocab_size = max_vocab_size
        self.min_len = min_len
        self.max_len = max_len
        self.share_vocab = share_vocab
        self.special_tokens = special_tokens

        self.data = {}
        self.SRC = TextField(tokenize_fn=tokenize, embed_file=embed_file, special_tokens=special_tokens)
        if self.share_vocab:
            self.TGT = self.SRC
            self.KB = self.SRC
        else:
            self.TGT = TextField(tokenize_fn=tokenize, embed_file=embed_file, special_tokens=special_tokens)
            self.KB = TextField(tokenize_fn=tokenize, embed_file=embed_file, special_tokens=special_tokens)

        self.fields = {'src': self.SRC,
                       'tgt': self.TGT,
                       'kb': self.KB}

        def src_filter_pred(src):
            return min_len <= len(self.SRC.tokenize_fn(src)) <= max_len

        def tgt_filter_pred(tgt):
            return min_len <= len(self.TGT.tokenize_fn(tgt)) <= max_len

        self.filter_pred = lambda ex: src_filter_pred(" ".join(ex['src'])) and tgt_filter_pred(" ".join(ex['tgt']))

        # load vocab or build vocab if not exists
        self.ent_token = self.special_tokens[0]
        self.nen_token = self.special_tokens[-1]
        self.load_vocab()
        self.ent_idx = self.TGT.stoi[self.ent_token]
        self.nen_idx = self.TGT.stoi[self.nen_token]
        self.padding_idx = self.TGT.stoi[self.TGT.pad_token]

    def load_vocab(self):
        """
        load_vocab
        """
        if not os.path.exists(self.prepared_vocab_file):
            print("Building vocab ...")
            train_file = os.path.join(self.data_dir, "train.data.txt")
            valid_file = os.path.join(self.data_dir, "dev.data.txt")
            test_file = os.path.join(self.data_dir, "test.data.txt")
            train_raw = self.read_data(train_file, data_type="train")
            valid_raw = self.read_data(valid_file, data_type="valid")
            test_raw = self.read_data(test_file, data_type="test")
            data_raw = train_raw + valid_raw + test_raw

            vocab_dict = self.build_vocab(data_raw)
            torch.save(vocab_dict, self.prepared_vocab_file)
            print("Saved prepared vocab to '{}'".format(self.prepared_vocab_file))
        else:
            print("Loading prepared vocab from {} ...".format(self.prepared_vocab_file))
            vocab_dict = torch.load(self.prepared_vocab_file)

        for name, vocab in vocab_dict.items():
            if name in self.fields:
                self.fields[name].load_vocab(vocab)
        for name, field in self.fields.items():
            if isinstance(field, TextField):
                print("Vocabulary size of fields {}-{}".format(name.upper(), field.vocab_size))

    def build_vocab(self, data):
        """
        Args
        ----
        data: ``List[Dict]``
        """
        field_data_dict = {}
        for name in data[0].keys():
            field = self.fields.get(name)
            if isinstance(field, TextField):
                xs = [x[name] for x in data]
                if field not in field_data_dict:
                    field_data_dict[field] = xs
                else:
                    field_data_dict[field] += xs

        vocab_dict = {}
        for name, field in self.fields.items():
            if field in field_data_dict:
                print("Building vocabulary of field {} ...".format(name.upper()))
                if field.vocab_size == 0:
                    field.build_vocab(field_data_dict[field],
                                      min_freq=self.min_freq,
                                      max_size=self.max_vocab_size)
                vocab_dict[name] = field.dump_vocab()
        return vocab_dict

    def load(self):
        """
        load
        """
        if not os.path.exists(self.prepared_data_file):
            self.data = self.build_data()

        else:
            print("Loading prepared data from {} ...".format(self.prepared_data_file))
            self.data = torch.load(self.prepared_data_file)
            print("Number of examples:",
                  " ".join("{}-{}".format(k.upper(), len(v)) for k, v in self.data.items()))

    def build_data(self):
        """
        build
        """
        train_file = os.path.join(self.data_dir, "train.data.txt")
        valid_file = os.path.join(self.data_dir, "dev.data.txt")
        test_file = os.path.join(self.data_dir, "test.data.txt")

        print("Reading data ...")
        train_raw = self.read_data(train_file, data_type="train")
        valid_raw = self.read_data(valid_file, data_type="valid")
        test_raw = self.read_data(test_file, data_type="test")

        print("Building TRAIN examples ...")
        train_data = self.build_examples(train_raw)
        print("Building VALID examples ...")
        valid_data = self.build_examples(valid_raw)
        print("Building TEST examples ...")
        test_data = self.build_examples(test_raw)

        data = {"train": train_data,
                "valid": valid_data,
                "test": test_data}
        torch.save(data, self.prepared_data_file)
        print("Saved prepared data to '{}'".format(self.prepared_data_file))
        return data

    def read_data(self, data_file, data_type="train"):
        """
        read_data
        """
        data = []
        with open(data_file, "r") as fr:
            for line in fr:
                sample = json.loads(line.strip())
                task = sample['task']
                uids = sample['uid']
                turn = uids.count('0')
                dialogs = sample['dialog']
                kb = sample['kb']
                # kb = [" ".join([self.nen_token] * 3)] + sample['kb']
                gold_entity = sample['gold_entity']
                ptr_index = sample['ptr_index']
                kb_index = sample['kb_index']
                src = []
                tgt = []
                tgt_e = []
                tgt_b = []
                for i, t in enumerate(range(0, len(uids), 2)):
                    if t == 0:
                        u_sent = dialogs[t]
                        s_sent = dialogs[t + 1]
                        # s_sent_b = " ".join([self.ent_token if w in gold_entity[i] else w
                        #                    for w in dialogs[t + 1].split(" ")])
                        # s_sent_e = " ".join([w if w in gold_entity[i] else self.nen_token
                        #                    for w in dialogs[t + 1].split(" ")])
                    else:
                        u_sent = " ".join([dialogs[t - 1], dialogs[t]])
                        s_sent = dialogs[t + 1]
                        # s_sent_b = " ".join([self.ent_token if w in gold_entity[i] else w
                        #                    for w in dialogs[t + 1].split(" ")])
                        # s_sent_e = " ".join([w if w in gold_entity[i] else self.nen_token
                        #                    for w in dialogs[t + 1].split(" ")])
                    src.append(u_sent)
                    tgt.append(s_sent)
                    # tgt_e.append(s_sent_e)
                    # tgt_b.append(s_sent_b)
                assert len(src) == turn
                assert len(tgt) == turn
                data_sample = {'task': task,
                               'turn': turn,
                               'src': src,
                               'tgt': tgt,
                               'tgt_e': tgt_e,
                               'tgt_b': tgt_b,
                               'ptr_index': ptr_index,
                               'kb_index': kb_index,
                               'gold_entity': gold_entity,
                               'kb': kb,
                               }
                data.append(data_sample)

        filtered_num = len(data)
        if not data_type == "test" and self.filter_pred is not None:
            data = [ex for ex in data if self.filter_pred(ex)]
        filtered_num -= len(data)
        print("Read {} {} examples ({} filtered)".format(len(data), data_type.upper(), filtered_num))
        return data

    def build_examples(self, data):
        """
        Args
        ----
        data: ``List[Dict]``
        """
        examples = []
        for raw_data in tqdm(data):
            example = {}
            for name, strings in raw_data.items():
                if name in self.fields.keys():
                    example[name] = self.fields[name].numericalize(strings)
                elif name.startswith("tgt"):
                    example[name] = self.fields["tgt"].numericalize(strings)
                else:
                    example[name] = strings
            examples.append(example)

        return examples

    def create_batches(self, batch_size, data_type="train", shuffle=False):
        """
        create_batches
        """
        try:
            data = self.data[data_type]
            dialog_batcher = DialogBatcher(batch_size=batch_size,
                                           data_type=data_type,
                                           shuffle=shuffle)
            dialog_batcher.prepare_input_list(input_data_list=data)
            return dialog_batcher
        except KeyError:
            raise KeyError("Unsupported data type: {}!".format(data_type))
