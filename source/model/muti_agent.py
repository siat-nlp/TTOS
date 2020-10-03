#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
File: source/model/muti-agent.py
"""
import torch
import sys
from source.utils.misc import Pack
from source.model.base_model import BaseModel
from source.utils.misc import sequence_mask
from source.utils.misc import sequence_kd_mask
from source.utils.criterions import KDLoss, MaskBCELoss
from tools.eval import eval_bleu_online
from tools.eval import eval_entity_f1_kvr_online
from tools.eval import eval_entity_f1_camrest_online


class Muti_Agent(BaseModel):
    """
    Muti_Agent
    """
    def __init__(self, data_name, ent_idx, nen_idx, model_S, model_TB, model_TE=None,
                 generator_S=None, generator_TB=None, generator_TE=None,
                 lambda_s=0.5, lambda_tb=0.5, lambda_te=0.5, lambda_g=1.0,
                 discriminator_B=None, discriminator_E=None, use_gpu=False):
        super(Muti_Agent, self).__init__()

        self.name = "muti_agent"
        self.data_name = data_name

        self.model_S = model_S
        self.model_TB = model_TB
        self.model_TE = model_TE

        self.generator_S = generator_S
        self.generator_TB = generator_TB
        self.generator_TE = generator_TE

        self.discriminator_B = discriminator_B
        self.discriminator_E = discriminator_E

        self.ent_idx = ent_idx
        self.nen_idx = nen_idx

        # compute lambda for pre-training model (in all test data)
        # adding the ensemble model is still three lambda
        self.lambda_s = lambda_s
        self.lambda_tb = lambda_tb
        self.lambda_te = lambda_te

        self.lambda_g = lambda_g

        # compute KD between two single model
        self.kd_loss = KDLoss()
        # compute loss in discriminator
        self.bce_loss = MaskBCELoss()

        self.use_gpu = use_gpu

    def discriminator_update(self, netD, real_data, fake_data, lengths, mask, optimizerD=None):
        """ Update D network: maximize log(D(x)) + log(1 - D(G(Z))) """

        # Train with all-real batch
        netD.zero_grad()
        # Format batch
        label = torch.full((real_data.size(0), ), 1)
        if self.use_gpu:
            label = label.cuda()
        # Forward pass real batch through D
        output = netD((real_data, lengths))
        # Calculate loss on all-real batch
        errD_real = self.bce_loss(input=output, target=label)
        # Calculate gradients for D in backward pass
        # errD_real.backward()

        # Train with all-fake batch
        label = torch.full((fake_data.size(0),), 0)
        if self.use_gpu:
            label = label.cuda()
        # Classify all fake batch with D
        output = netD((fake_data.detach(), lengths))
        # Calculate D's loss on the all-fake batch
        errD_fake = self.bce_loss(input=output, target=label)
        # Calculate the gradients for this batch
        # errD_fake.backward()

        # Add the graidents from all-real and all-fake batches
        errD = errD_real + errD_fake
        # update D
        # optimizerD.step()
        return errD

    def generator_update(self, netG, netDB, netDE, fake_data, length, mask, nll, lambda_g=1.0, optimizerG=None):
        """ Update G network: maximize log(DB(G(z))) + log(DE(G(z))) + LL"""

        netG.zero_grad()
        # fake labels are real for generator cost
        label = torch.full((fake_data.size(0), ), 1)
        if self.use_gpu:
            label = label.cuda()
        # Since we just update D, perform another forward pass of all-fake batch through D
        output_B, output_E = netDB((fake_data, length)), netDE((fake_data, length))
        # Calculate G's loss based on two outputs
        errG_B = self.bce_loss(input=output_B, target=label)
        errG_E = self.bce_loss(input=output_E, target=label)
        # Calculate gradients for G
        errG = lambda_g * (errG_B + errG_E) + nll
        # errG.backward()
        # update G
        # optimizerG.step()
        return errG, errG_B, errG_E, nll

    def compare_metric(self, generator_1, generator_2, turn_inputs, kb_inputs, type='bleu', data_name='camrest'):
        """
        The metric of type in model_1 gt that in model_2 return True about a batch, otherwise False
        Default deal camrest dataset (ignore equal in metric because of low probability)
        """
        hyps_1, refs_1, tasks_1, gold_entity_1, kb_word_1 = generator_1.generate_batch(turn_inputs=turn_inputs,
                                                                                       kb_inputs=kb_inputs)
        hyps_2, refs_2, tasks_2, gold_entity_2, kb_word_2 = generator_2.generate_batch(turn_inputs=turn_inputs,
                                                                                       kb_inputs=kb_inputs)

        model_1_name, model_2_name = generator_1.model.name, generator_2.model.name

        if type == 'bleu':
            bleu_1 = eval_bleu_online(hyps=hyps_1, refs=refs_1)
            bleu_2 = eval_bleu_online(hyps=hyps_2, refs=refs_2)
            res = True if bleu_1 > bleu_2 else False
            report_str = type + ": " + model_1_name + '-' + str(bleu_1) + (' > ' if res else ' < ') + \
                         model_2_name + '-' + str(bleu_2)
            return res, report_str
        else:
            # default compute F1_score as metric
            if data_name == 'camrest':
                F1_score_1 = eval_entity_f1_camrest_online(hyps=hyps_1, tasks=tasks_1, gold_entity=gold_entity_1,
                                                           kb_word=kb_word_1)
                F1_score_2 = eval_entity_f1_camrest_online(hyps=hyps_2, tasks=tasks_2, gold_entity=gold_entity_2,
                                                           kb_word=kb_word_2)
            else:
                assert data_name == 'kvr'
                # default compute kvret as dataset todo complete like above camrest
                F1_score_1 = eval_entity_f1_kvr_online(hyps=hyps_1, tasks=tasks_1, gold_entity=gold_entity_1,
                                                           kb_word=kb_word_1)
                F1_score_2 = eval_entity_f1_kvr_online(hyps=hyps_2, tasks=tasks_2, gold_entity=gold_entity_2,
                                                           kb_word=kb_word_2)
            res = True if F1_score_1 > F1_score_2 else False
            report_str = type + ": " + model_1_name + '-' + str(F1_score_1) + (' > ' if res else ' < ') + \
                         model_2_name + '-' + str(F1_score_2)
            return res, report_str

    def iterate(self, turn_inputs, kb_inputs,
                optimizer=None, grad_clip=None, is_training=True, method="GAN", mask=False):
        """
        iterate
        note: this function iterate in the whole model (muti-agent) instead of single sub_model
        """

        if isinstance(optimizer, tuple):
            optimizerG, optimizerDB, optimizerDE = optimizer

        # clear all memory before the begin of a new batch computation
        for name, model in self.named_children():
            if name.startswith("model_"):
                model.reset_memory()
                model.load_kb_memory(kb_inputs)

        # store the whole model (muti_agent)'s metric
        metrics_list_S, metrics_list_TB, metrics_list_TE = [], [], []
        metrics_list_G, metrics_list_DB, metrics_list_DE = [], [], []
        mask_list_S, length_list = [], []
        # store the whole model (muti_agent)'s loss
        total_loss_DB, total_loss_DE, total_loss_G = 0, 0, 0
        # use to compute final loss (sum of each agent's loss) per turn for the cumulated total_loss in a batch
        loss = Pack()
        # use to store kb_mask for three single model
        kd_masks = Pack()

        # compare evaluation metric (bleu/f1score) among models
        if method in ('1-3', 'GAN'):
            # TODO complete
            bleu_ENS_gt_S, bleu_ENS_gt_TB, f1score_ENS_gt_TE = True, True, True
        else:
            # compute bleu_S_gt_TB per batch (compute metric for the following training batch)
            # (key: batch/following/training)
            res_bleu = self.compare_metric(generator_1=self.generator_S, generator_2=self.generator_TB,
                                           turn_inputs=turn_inputs, kb_inputs=kb_inputs, type='bleu',
                                           data_name=self.data_name)
            if isinstance(res_bleu, tuple):
                bleu_S_gt_TB, bleu_S_gt_TB_str = res_bleu
            else:
                assert isinstance(res_bleu, bool)
                bleu_S_gt_TB, bleu_S_gt_TB_str = res_bleu, ''
            if self.model_TE is not None:
                res_f1score = self.compare_metric(generator_1=self.generator_S, generator_2=self.generator_TE,
                                                  turn_inputs=turn_inputs, kb_inputs=kb_inputs, type='f1score',
                                                  data_name=self.data_name)
                if isinstance(res_f1score, tuple):
                    f1score_S_gt_TE, f1score_S_gt_TE_str = res_f1score
                else:
                    assert isinstance(res_f1score, bool)
                    f1score_S_gt_TE, f1score_S_gt_TE_str = res_f1score, ''

        """ update discriminator """

        # clear all memory again because of cumulation of the memory in the computation of the above generator
        for name, model in self.named_children():
            if name.startswith("model_"):
                model.reset_memory()
                model.load_kb_memory(kb_inputs)

        # begin iterate (a dialogue batch)
        for i, inputs in enumerate(turn_inputs):

            for name, model in self.named_children():
                if name.startswith("model_"):
                    if model.use_gpu:
                        inputs = inputs.cuda()
                    src, src_lengths = inputs.src
                    tgt, tgt_lengths = inputs.tgt
                    task_label = inputs.task
                    gold_entity = inputs.gold_entity
                    ptr_index, ptr_lengths = inputs.ptr_index
                    kb_index, kb_index_lengths = inputs.kb_index
                    enc_inputs = src[:, 1:-1], src_lengths - 2  # filter <bos> <eos>
                    dec_inputs = tgt[:, :-1], tgt_lengths - 1  # filter <eos>
                    target = tgt[:, 1:]  # filter <bos>
                    target_mask = sequence_mask(tgt_lengths - 1)
                    kd_mask = sequence_kd_mask(tgt_lengths - 1, target, name, self.ent_idx, self.nen_idx)

                    outputs = model.forward(enc_inputs, dec_inputs)
                    metrics = model.collect_metrics(outputs, target, ptr_index, kb_index)


                    if name == "model_S":
                        metrics_list_S.append(metrics)
                    elif name == "model_TB":
                        metrics_list_TB.append(metrics)
                    else:
                        metrics_list_TE.append(metrics)

                    kd_masks[name] = kd_mask if mask else target_mask
                    loss[name] = metrics

                    model.update_memory(dialog_state_memory=outputs.dialog_state_memory,
                                        kb_state_memory=outputs.kb_state_memory)

            # store necessary data for three single model
            if self.model_TE is not None:
                kd_mask_e = kd_masks.model_TE
            kd_mask_s = kd_masks.model_S
            kd_mask_b = kd_masks.model_TB
            mask_list_S.append(kd_mask_s)
            length_list.append(tgt_lengths - 1)

            assert False not in (kd_mask_b == kd_mask_e)

            errD_B = self.discriminator_update(netD=self.discriminator_B, real_data=loss.model_TB.prob,
                                               fake_data=loss.model_S.prob, lengths=tgt_lengths - 1,
                                               mask=kd_mask_b)
            errD_E = self.discriminator_update(netD=self.discriminator_E, real_data=loss.model_TE.prob,
                                               fake_data=loss.model_S.prob, lengths=tgt_lengths - 1,
                                               mask=kd_mask_e)
            # collect discriminator‘s total loss
            metrics_DB = Pack(num_samples=metrics.num_samples)
            metrics_DE = Pack(num_samples=metrics.num_samples)
            metrics_DB.add(loss=errD_B, logits=0.0, prob=0.0)
            metrics_DE.add(loss=errD_E, logits=0.0, prob=0.0)
            metrics_list_DB.append(metrics_DB)
            metrics_list_DE.append(metrics_DE)

            # update in a batch
            total_loss_DB = total_loss_DB + errD_B
            total_loss_DE = total_loss_DE + errD_E
            loss.clear()
            kd_masks.clear()

        # check loss
        if torch.isnan(total_loss_DB) or torch.isnan(total_loss_DE):
            raise ValueError("NAN loss encountered!")

        # compute and update gradient
        if is_training:
            assert not None in (optimizerDB, optimizerDE)
            optimizerDB.zero_grad()
            optimizerDE.zero_grad()
            total_loss_DB.backward()
            total_loss_DE.backward()
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(parameters=self.discriminator_B.parameters(), max_norm=grad_clip)
                torch.nn.utils.clip_grad_norm_(parameters=self.discriminator_E.parameters(), max_norm=grad_clip)
            optimizerDB.step()
            optimizerDE.step()


        """ update generator """

        # begin iterate (a dialogue batch)
        n_turn = len(metrics_list_S)
        assert n_turn == len(turn_inputs) == len(mask_list_S)
        for i in range(n_turn):
            errG, errG_B, errG_E, nll = self.generator_update(netG=self.model_S, netDB=self.discriminator_B,
                                                              netDE=self.discriminator_E,
                                                              fake_data=metrics_list_S[i].prob,
                                                              length=length_list[i],
                                                              mask=mask_list_S[i],
                                                              nll=metrics_list_S[i].loss,
                                                              lambda_g=self.lambda_g)

            # collect generator‘s total loss
            metrics_G = Pack(num_samples=metrics_list_S[i].num_samples)
            metrics_G.add(loss=errG, loss_gb=errG_B, loss_ge=errG_E, loss_nll=nll, logits=0.0, prob=0.0)
            metrics_list_G.append(metrics_G)

            # update in a batch
            total_loss_G += errG

        # check loss
        if torch.isnan(total_loss_G):
            raise ValueError("NAN loss encountered!")

        # compute and update gradient
        if is_training:
            assert optimizerG is not None
            optimizerG.zero_grad()
            total_loss_G.backward()
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(parameters=self.model_S.parameters(), max_norm=grad_clip)
            optimizerG.step()

        return metrics_list_S, metrics_list_G, metrics_list_DB, metrics_list_DE
