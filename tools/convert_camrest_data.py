#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File: convert_camrest_data.py
"""

import json
import ast
import numpy as np


def convert_text_for_sample(input_file, out_file):
    kbs = []
    dialogs = []

    count = 0
    with open(input_file, 'r') as fr, open(out_file, 'w') as fw:
        for line in fr:
            line = line.strip()
            if line:
                if 'R_' in line:
                    triple = line.split()[1:]
                    triple_str = ' '.join(triple).replace('R_', '')
                    kbs.append(triple_str)
                elif 'api_call' in line:
                    usr_sent = line.split('\t')[0]
                    usr_sent = ' '.join(usr_sent.split()[1:])
                elif '<SILENCE>' in line:
                    sys_sent = line.split('\t')[1]
                    assert usr_sent is not None
                    dialog = usr_sent + '\t' + sys_sent
                    dialogs.append(dialog)
                else:
                    u, s = line.split('\t')
                    u = ' '.join(u.split()[1:])
                    dialog = u + '\t' + s
                    dialogs.append(dialog)
            else:
                new_kbs = []
                entities = []
                for triple in kbs:
                    subj, rel, obj = triple.split()
                    entities.append(subj)
                    entities.append(obj)
                    poi_triple = [subj, 'poi', subj]
                    poi_triple = ' '.join(poi_triple)
                    if poi_triple not in new_kbs:
                        new_kbs.append(poi_triple)
                    new_kbs.append(triple)
                gold_ents = []
                entities = set(entities)
                for i, dialog in enumerate(dialogs):
                    u, s = dialog.split('\t')
                    sys_toks = s.split()
                    gold_entity = []
                    for tok in sys_toks:
                        if tok in entities:
                            gold_entity.append(tok)
                    gold_ents.append(gold_entity)

                for triple in new_kbs:
                    kb_line = '0 ' + triple
                    fw.write(kb_line)
                    fw.write('\n')

                assert len(gold_ents) == len(dialogs)
                for i, dialog in enumerate(dialogs):
                    dialog_line = str(i+1) + ' ' + dialog + '\t' + str(gold_ents[i])
                    fw.write(dialog_line)
                    fw.write('\n')
                fw.write('\n')
                kbs = []
                dialogs = []
                count += 1
    print("total dialogs:", count)


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
                if line.startswith('0'):
                    triple = line.split()[1:]
                    kb_triple = ' '.join(triple)
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
                sample['task'] = 'restaurant'
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
    print("max utterances:", max([len(s['uid']) for s in all_samples]))      # 16
    print("min utterances:", min([len(s['uid']) for s in all_samples]))      # 4
    print("avg utterances:", np.mean([len(s['uid']) for s in all_samples]))  # 7.98 / 8.32 / 8.32
    print("max kb triples:", max([len(s['kb']) for s in all_samples]))       # 452 / 248 / 112
    print("min kb triples:", min([len(s['kb']) for s in all_samples]))       # 1
    print("avg kb triples:", np.mean([len(s['kb']) for s in all_samples]))   # 23.57 / 21.64 / 22.62
    with open(out_file, 'w') as fw:
        for sample in all_samples:
            line = json.dumps(sample)
            fw.write(line)
            fw.write('\n')


if __name__ == '__main__':
    data_dir = "./data/CamRest"
    modes = ['train', 'dev', 'test']

    for mode in modes:
        input_file1 = "%s/camrest676-%s.txt" % (data_dir, mode)
        out_file1 = "%s/%s.txt" % (data_dir, mode)
        convert_text_for_sample(input_file1, out_file1)

    for mode in modes:
        input_file2 = "%s/%s.txt" % (data_dir, mode)
        out_file2 = "%s/%s.data.txt" % (data_dir, mode)
        convert_text_for_model(input_file2, out_file2)
