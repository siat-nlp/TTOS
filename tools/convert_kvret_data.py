#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File: convert_kvret_data.py
"""

import json
import ast
import numpy as np


def convert_text_for_model(input_file, out_file):
    all_samples = []
    sample = {}
    uid = []
    dialog = []
    kb = []
    gold_entity = []
    ptr_index = []
    kb_index = []
    with open(input_file, 'r') as fr:
        for line in fr:
            line = line.strip()
            if line:
                if line.startswith('#'):
                    task = line.replace('#', '')
                    sample['task'] = task
                elif line.startswith('0'):
                    triples = line.split()[1:]
                    if len(triples) <= 4:
                        if len(triples) == 2:
                            kb_triple = triples[0] + " <pad> " + triples[1]
                        elif len(triples) == 4:
                            kb_triple = triples[0] + " " + triples[1] + "_" + triples[2] + " " + triples[3]
                        else:
                            if "poi_type" in triples:
                                kb_triple = triples[2] + " poi " + triples[0]
                            else:
                                kb_triple = " ".join(triples)
                        kb.append(kb_triple)
                else:
                    u, s, gold_ent = line.split('\t')
                    u = " ".join(u.split()[1:])
                    uid.append('1')
                    dialog.append(u)
                    uid.append('0')
                    dialog.append(s)
                    gold_ent = ast.literal_eval(gold_ent)
                    gold_entity.append(gold_ent)
                    ptr = [1 if (w in gold_ent and len(kb) > 0) else 0 for w in s.split()]
                    ptr_index.append(ptr)
                    if len(kb) == 0:
                        kb_ptr = [0]
                    else:
                        kb_ptr = []
                        for triple in kb:
                            tup = triple.split()
                            assert len(tup) == 3
                            sub, rel, obj = tup[0], tup[1], tup[2]
                            if obj in s.split():
                                kb_ptr.append(1)
                            else:
                                kb_ptr.append(0)
                    kb_index.append(kb_ptr)
            else:
                sample['uid'] = uid
                sample['dialog'] = dialog
                sample['gold_entity'] = gold_entity
                sample['ptr_index'] = ptr_index
                sample['kb_index'] = kb_index
                if len(kb) == 0:
                    sample['kb'] = ["<pad> <pad> <pad>"]
                else:
                    sample['kb'] = kb

                all_samples.append(sample)
                sample = {}
                uid = []
                dialog = []
                kb = []
                gold_entity = []
                ptr_index = []
                kb_index = []
    print("total samples:", len(all_samples))

    for i, s in enumerate(all_samples):
        if len(s['uid']) == 0:
            print("index=%d utterance is None! filtered." % i)
            del all_samples[i]
    print("max utterances:", max([len(s['uid']) for s in all_samples]))      # 12
    print("min utterances:", min([len(s['uid']) for s in all_samples]))      # 2
    print("avg utterances:", np.mean([len(s['uid']) for s in all_samples]))
    print("max kb triples:", max([len(s['kb']) for s in all_samples]))       # 148
    print("min kb triples:", min([len(s['kb']) for s in all_samples]))       # 0
    print("avg kb triples:", np.mean([len(s['kb']) for s in all_samples]))   # 62.3
    with open(out_file, 'w') as fw:
        for sample in all_samples:
            line = json.dumps(sample)
            fw.write(line)
            fw.write('\n')


if __name__ == '__main__':
    data_dir = "./data/KVR"
    modes = ['train', 'dev', 'test']
    for mode in modes:
        input_file = "%s/%s.txt" % (data_dir, mode)
        out_file = "%s/%s.data.txt" % (data_dir, mode)
        convert_text_for_model(input_file, out_file)
