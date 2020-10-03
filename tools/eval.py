#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File: eval.py
"""
import argparse
import json
import numpy as np
from source.utils.metrics import moses_multi_bleu
from source.utils.metrics import compute_prf


def eval_bleu(eval_fp):
    hyps = []
    refs = []
    with open(eval_fp, 'r') as fr:
        for line in fr:
            dialog = json.loads(line.strip())
            pred_str = dialog["result"]
            gold_str = dialog["target"]
            hyps.append(pred_str)
            refs.append(gold_str)
    assert len(hyps) == len(refs)
    hyp_arrys = np.array(hyps)
    ref_arrys = np.array(refs)

    bleu_score = moses_multi_bleu(hyp_arrys, ref_arrys, lowercase=True)
    return bleu_score


def eval_bleu_online(hyps, refs):
    assert len(hyps) == len(refs)

    hyps_new, refs_new = [], []
    for x, y in zip(hyps, refs):
        pred_str = ' '.join([w for w in x.split(" ") if w != '<ENT>'])
        gold_str = ' '.join([w for w in y.split(" ") if w != '<ENT>'])
        hyps_new.append(pred_str)
        refs_new.append(gold_str)

    hyp_arrys = np.array(hyps_new)
    ref_arrys = np.array(refs_new)

    bleu_score = moses_multi_bleu(hyp_arrys, ref_arrys, lowercase=True)
    return bleu_score


def eval_entity_f1_kvr(eval_fp, entity_fp):
    test_data = []
    with open(eval_fp, 'r') as fr:
        for line in fr:
            ent_idx_sch, ent_idx_wet, ent_idx_nav = [], [], []
            dialog = json.loads(line.strip())
            if dialog["task"] == "schedule":
                ent_idx_sch = dialog["gold_entity"]
            elif dialog["task"] == "weather":
                ent_idx_wet = dialog["gold_entity"]
            elif dialog["task"] == "navigate":
                ent_idx_nav = dialog["gold_entity"]
            ent_index = list(set(ent_idx_sch + ent_idx_wet + ent_idx_nav))
            dialog["ent_index"] = ent_index
            dialog["ent_idx_sch"] = list(set(ent_idx_sch))
            dialog["ent_idx_wet"] = list(set(ent_idx_wet))
            dialog["ent_idx_nav"] = list(set(ent_idx_nav))
            test_data.append(dialog)

    print("test data: ", len(test_data))

    with open(entity_fp, 'r') as fr:
        global_entity = json.load(fr)
        global_entity_list = []
        for key in global_entity.keys():
            if key != 'poi':
                global_entity_list += [item.lower().replace(' ', '_') for item in global_entity[key]]
            else:
                for item in global_entity['poi']:
                    global_entity_list += [item[k].lower().replace(' ', '_') for k in item.keys()]
    global_entity_list = list(set(global_entity_list))

    F1_pred, F1_sch_pred, F1_nav_pred, F1_wet_pred = 0, 0, 0, 0
    F1_count, F1_sch_count, F1_nav_count, F1_wet_count = 0, 0, 0, 0
    for dialog in test_data:
        pred_tokens = dialog["result"].replace('_', ' ').split()
        kb_arrys = dialog["kb"]

        gold_ents = dialog["ent_index"]
        if len(gold_ents) > 0:
            gold_ents = ' '.join(gold_ents).replace('_', ' ').split()
        single_f1, count = compute_prf(gold_ents, pred_tokens, global_entity_list, kb_arrys)
        F1_pred += single_f1
        F1_count += count

        gold_sch_ents = dialog["ent_idx_sch"]
        if len(gold_sch_ents) > 0:
            gold_sch_ents = ' '.join(gold_sch_ents).replace('_', ' ').split()
        single_f1, count = compute_prf(gold_sch_ents, pred_tokens, global_entity_list, kb_arrys)
        F1_sch_pred += single_f1
        F1_sch_count += count

        gold_wet_ents = dialog["ent_idx_wet"]
        if len(gold_wet_ents) > 0:
            gold_wet_ents = ' '.join(gold_wet_ents).replace('_', ' ').split()
        single_f1, count = compute_prf(gold_wet_ents, pred_tokens, global_entity_list, kb_arrys)
        F1_wet_pred += single_f1
        F1_wet_count += count

        gold_nav_ents = dialog["ent_idx_nav"]
        if len(gold_nav_ents) > 0:
            gold_nav_ents = ' '.join(gold_nav_ents).replace('_', ' ').split()
        single_f1, count = compute_prf(gold_nav_ents, pred_tokens, [], kb_arrys)
        F1_nav_pred += single_f1
        F1_nav_count += count

    F1_score = F1_pred / float(F1_count)
    F1_sch_score = F1_sch_pred / float(F1_sch_count)
    F1_wet_score = F1_wet_pred / float(F1_wet_count)
    F1_nav_score = F1_nav_pred / float(F1_nav_count)
    return F1_score, F1_sch_score, F1_wet_score, F1_nav_score


def eval_entity_f1_kvr_online(hyps, tasks, gold_entity, kb_word):

    entity_fp = "./data/KVR/kvret_entities.json"

    with open(entity_fp, 'r') as fr:
        global_entity = json.load(fr)
        global_entity_list = []
        for key in global_entity.keys():
            if key != 'poi':
                global_entity_list += [item.lower().replace(' ', '_') for item in global_entity[key]]
            else:
                for item in global_entity['poi']:
                    global_entity_list += [item[k].lower().replace(' ', '_') for k in item.keys()]
    global_entity_list = list(set(global_entity_list))

    test_data = []
    for hyp, tk, ent, kb in zip(hyps, tasks, gold_entity, kb_word):
        dialog = {"result": hyp,
                  "task": tk,
                  "gold_entity": ent,
                  "kb": kb}
        test_data.append(dialog)

    F1_pred, F1_count = 0, 0
    for dialog in test_data:
        pred_tokens = dialog["result"].replace('_', ' ').split()
        kb_arrys = dialog["kb"]

        gold_ents = dialog["gold_entity"]
        if len(gold_ents) > 0:
            gold_ents = ' '.join(gold_ents).replace('_', ' ').split()
        single_f1, count = compute_prf(gold_ents, pred_tokens, global_entity_list, kb_arrys)
        F1_pred += single_f1
        F1_count += count

    F1_score = F1_pred / float(F1_count)

    return F1_score


def eval_entity_f1_camrest(eval_fp, entity_fp):
    test_data = []
    with open(eval_fp, 'r') as fr:
        for line in fr:
            dialog = json.loads(line.strip())
            test_data.append(dialog)

    print("test data: ", len(test_data))

    with open(entity_fp, 'r') as fr:
        global_entity = json.load(fr)
        global_entity_list = []
        for key in global_entity.keys():
            global_entity_list += [item.lower().replace(' ', '_') for item in global_entity[key]]
    global_entity_list = list(set(global_entity_list))

    F1_pred, F1_count = 0, 0
    for dialog in test_data:
        pred_tokens = dialog["result"].replace('_', ' ').split()
        kb_arrys = dialog["kb"]

        gold_ents = dialog["gold_entity"]
        if len(gold_ents) > 0:
            gold_ents = ' '.join(gold_ents).replace('_', ' ').split()
        single_f1, count = compute_prf(gold_ents, pred_tokens, global_entity_list, kb_arrys)
        F1_pred += single_f1
        F1_count += count

    F1_score = F1_pred / float(F1_count)

    return F1_score


def eval_entity_f1_camrest_online(hyps, tasks, gold_entity, kb_word):

    entity_fp = "./data/CamRest/camrest676-entities.json"
    with open(entity_fp, 'r') as fr:
        global_entity = json.load(fr)
        global_entity_list = []
        for key in global_entity.keys():
            global_entity_list += [item.lower().replace(' ', '_') for item in global_entity[key]]
    global_entity_list = list(set(global_entity_list))

    test_data = []
    for hyp, tk, ent, kb in zip(hyps, tasks, gold_entity, kb_word):
        dialog = {"result": hyp,
                  "task": tk,
                  "gold_entity": ent,
                  "kb": kb}
        test_data.append(dialog)

    F1_pred, F1_count = 0, 0
    for dialog in test_data:
        pred_tokens = dialog["result"].replace('_', ' ').split()
        kb_arrys = dialog["kb"]

        gold_ents = dialog["gold_entity"]
        if len(gold_ents) > 0:
            gold_ents = ' '.join(gold_ents).replace('_', ' ').split()
        single_f1, count = compute_prf(gold_ents, pred_tokens, global_entity_list, kb_arrys)
        F1_pred += single_f1
        F1_count += count

    F1_score = F1_pred / float(F1_count)

    return F1_score


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str)
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--eval_dir", type=str)
    args = parser.parse_args()

    data_dir = args.data_dir
    eval_dir = args.eval_dir

    eval_file = "%s/output.txt" % eval_dir

    # cal bleu
    bleu = eval_bleu(eval_file)

    # cal entity F1
    if args.data_name == 'kvr':
        entity_file = "%s/kvret_entities.json" % data_dir
        f1_score, f1_sch, f1_wet, f1_nav = eval_entity_f1_kvr(eval_file, entity_file)

        output_str = "BLEU SCORE: %.3f\n" % bleu
        output_str += "F1 SCORE: %.2f%%\n" % (f1_score * 100)
        output_str += "SCH F1: %.2f%%\n" % (f1_sch * 100)
        output_str += "WET F1: %.2f%%\n" % (f1_wet * 100)
        output_str += "NAV F1: %.2f%%" % (f1_nav * 100)
        print(output_str)
    else:
        entity_file = "%s/camrest676-entities.json" % data_dir
        f1_score = eval_entity_f1_camrest(eval_file, entity_file)
        output_str = "BLEU SCORE: %.3f\n" % bleu
        output_str += "F1 SCORE: %.2f%%" % (f1_score * 100)
        print(output_str)

    # write evaluation results to file
    out_file = "%s/eval.result.txt" % eval_dir
    with open(out_file, 'w') as fw:
        fw.write(output_str)
    print("Saved evaluation results to '{}.'".format(out_file))
