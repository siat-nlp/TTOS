#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File: source/utils/engine.py
"""

import os
import time
import shutil
import numpy as np
import torch
import sys

from collections import defaultdict
from source.inputter.batcher import create_turn_batch, create_kb_batch


class MetricsManager(object):
    """
    MetricsManager
    """

    def __init__(self):
        self.metrics_val = defaultdict(float)
        self.metrics_cum = defaultdict(float)
        self.num_samples = 0

    def update(self, metrics_list):
        """
        update
        """
        for i, metrics in enumerate(metrics_list):
            num_samples = metrics.pop("num_samples", 1)
            # TODO remove logits and prob from reporting
            metrics.pop("logits")
            metrics.pop("prob")
            self.num_samples += num_samples
            for key, val in metrics.items():
                if val is not None:
                    key_turn = str(key) + "-turn-{}".format(str(i + 1))
                    self.metrics_val[key_turn] = val

                    if isinstance(val, torch.Tensor):
                        val = val.item()
                        self.metrics_cum[key] += val * num_samples
                    elif isinstance(val, tuple):
                        assert len(val) == 2
                        val, num_words = val[0].item(), val[1]
                        self.metrics_cum[key] += np.array([val * num_samples, num_words])
                    else:
                        self.metrics_cum[key] += val * num_samples

    def clear(self):
        """
        clear
        """
        self.metrics_val = defaultdict(float)
        self.metrics_cum = defaultdict(float)
        self.num_samples = 0

    def get(self, name):
        """
        get
        """
        val = self.metrics_cum.get(name)
        if not isinstance(val, float):
            val = val[0]
        return val / self.num_samples

    def report_val(self):
        """
        report_val
        """
        metric_strs = []
        for key, val in self.metrics_val.items():
            metric_str = "{}={:.3f}".format(key.upper(), val)
            metric_strs.append(metric_str)
        metric_strs = "   ".join(metric_strs)
        return metric_strs

    def report_cum(self):
        """
        report_cum
        """
        metric_strs = []
        for key, val in self.metrics_cum.items():
            if isinstance(val, float):
                val, num_words = val, None
            else:
                val, num_words = val

            metric_str = "{}={:.3f}".format(key.upper(), val / self.num_samples)
            metric_strs.append(metric_str)

            if num_words is not None:
                ppl = np.exp(min(val / num_words, 100))
                metric_str = "{}-PPL={:.3f}".format(key.upper(), ppl)
                metric_strs.append(metric_str)

        metric_strs = "   ".join(metric_strs)
        return metric_strs


class Trainer(object):
    """
    Trainer
    """

    def __init__(self,
                 model,
                 optimizer,
                 train_iter,
                 valid_iter,
                 logger,
                 method,
                 valid_metric_name="-loss",
                 num_epochs=1,
                 pre_epochs=1,
                 save_dir=None,
                 pre_train_dir=None,
                 log_steps=None,
                 valid_steps=None,
                 grad_clip=None,
                 lr_scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.train_iter = train_iter
        self.valid_iter = valid_iter
        self.logger = logger

        self.is_decreased_valid_metric = valid_metric_name[0] == "-"
        self.valid_metric_name = valid_metric_name[1:]
        self.num_epochs = num_epochs
        self.pre_epochs = pre_epochs
        self.save_dir = save_dir
        self.pre_train_dir = pre_train_dir
        self.log_steps = log_steps
        self.valid_steps = valid_steps
        self.grad_clip = grad_clip
        self.lr_scheduler = lr_scheduler
        self.method = method

        self.best_valid_metric = float(
            "inf") if self.is_decreased_valid_metric else -float("inf")
        self.epoch = 0
        self.batch_num = 0
        self.use_rl = False

        self.train_start_message = "\n".join(["",
                                              "=" * 85,
                                              "=" * 34 + " Model Training " + "=" * 35,
                                              "=" * 85,
                                              ""])
        self.valid_start_message = "\n" + "-" * 33 + " Model Evaulation " + "-" * 33

    def train(self):
        """
        train
        """
        for epoch in range(self.epoch, self.pre_epochs):
            self.train_epoch()

    def train_epoch(self):
        """
        train_epoch
        """
        self.epoch += 1
        train_mm_S, train_mm_G, train_mm_DB, train_mm_DE = MetricsManager(), MetricsManager(), MetricsManager(), \
                                                           MetricsManager()
        num_batches = self.train_iter.n_batch
        self.train_iter.prepare_epoch()
        self.logger.info(self.train_start_message)

        for batch_idx in range(num_batches):
            self.model.train()
            start_time = time.time()

            local_data = self.train_iter.get_batch(batch_idx)
            turn_inputs = create_turn_batch(local_data['inputs'])
            kb_inputs = create_kb_batch(local_data['kbs'])
            assert len(turn_inputs) == local_data['max_turn']

            metrics_list_S, metrics_list_G, metrics_list_DB, metrics_list_DE \
                                                      = self.model.iterate(turn_inputs, kb_inputs,
                                                                           optimizer=self.optimizer,
                                                                           grad_clip=self.grad_clip,
                                                                           is_training=True,
                                                                           method=self.method)

            elapsed = time.time() - start_time
            train_mm_S.update(metrics_list_S)
            train_mm_G.update(metrics_list_G)
            train_mm_DB.update(metrics_list_DB)
            train_mm_DE.update(metrics_list_DE)
            self.batch_num += 1



            if (batch_idx + 1) % self.log_steps == 0:
                self.report_log_steps(train_mm=train_mm_S, elapsed=elapsed, batch_idx=batch_idx,
                                          num_batches=num_batches)
                self.report_log_steps(train_mm=train_mm_G, elapsed=elapsed, batch_idx=batch_idx,
                                          num_batches=num_batches)
                self.report_log_steps(train_mm=train_mm_DB, elapsed=elapsed, batch_idx=batch_idx,
                                          num_batches=num_batches)
                self.report_log_steps(train_mm=train_mm_DE, elapsed=elapsed, batch_idx=batch_idx,
                                          num_batches=num_batches)

            # TODO only evaluate model_S here
            if (batch_idx + 1) % self.valid_steps == 0:
                self.report_valid_steps(model=self.model.model_S, batch_idx=batch_idx, num_batches=num_batches)

        self.save()
        self.logger.info('')

    def report_log_steps(self, train_mm, elapsed, batch_idx, num_batches):
        message_prefix = "[Train][{:2d}][{}/{}]".format(self.epoch, batch_idx + 1, num_batches)
        metrics_message = train_mm.report_val()
        message_posfix = "TIME={:.2f}s".format(elapsed)
        self.logger.info("   ".join(
            [message_prefix, metrics_message, message_posfix]))

    def report_valid_steps(self, model, batch_idx, num_batches):
        self.logger.info(self.valid_start_message)
        if model.name == 'muti_agent':
            valid_mm_M, valid_mm_S, valid_mm_TB, valid_mm_TE = self.evaluate(model, self.valid_iter,
                                                                             use_rl=self.use_rl, method=self.method)
            message_prefix = "[Valid][{:2d}][{}/{}]".format(self.epoch, batch_idx + 1, num_batches)
            metrics_message_M = valid_mm_M.report_cum()
            metrics_message_S = valid_mm_S.report_cum()
            metrics_message_TB = valid_mm_TB.report_cum()
            metrics_message_TE = valid_mm_TE.report_cum()
            self.logger.info("   ".join([message_prefix, model.name, metrics_message_M]))
            self.logger.info("   ".join([message_prefix, model.model_S.name, metrics_message_S]))
            self.logger.info("   ".join([message_prefix, model.model_TB.name, metrics_message_TB]))
            if metrics_message_TE:
                self.logger.info("   ".join([message_prefix, model.model_TE.name, metrics_message_TE]))

            cur_valid_metric = valid_mm_M.get(self.valid_metric_name)

        else:
            valid_mm = self.evaluate(model, self.valid_iter, use_rl=self.use_rl)
            message_prefix = "[Valid][{:2d}][{}/{}]".format(self.epoch, batch_idx + 1, num_batches)
            metrics_message = valid_mm.report_cum()
            self.logger.info("   ".join([message_prefix, metrics_message]))
            cur_valid_metric = valid_mm.get(self.valid_metric_name)

        if self.is_decreased_valid_metric:
            is_best = cur_valid_metric < self.best_valid_metric
        else:
            is_best = cur_valid_metric > self.best_valid_metric
        if is_best:
            self.best_valid_metric = cur_valid_metric
        self.save(is_best, is_rl=self.use_rl)
        if self.lr_scheduler is not None:
            self.lr_scheduler.step(cur_valid_metric)

        self.logger.info("-" * 85 + "\n")

    @staticmethod
    def evaluate(model, data_iter, use_rl=False, method=None):
        """
        evaluate
        note: this function evalute single sub_model instead of the whole model (muti-agent)
        """
        model.eval()
        mm = MetricsManager()
        mm_M, mm_S, mm_TB, mm_TE = MetricsManager(), MetricsManager(), MetricsManager(), MetricsManager()
        num_batches = data_iter.n_batch
        with torch.no_grad():
            for batch_idx in range(num_batches):
                local_data = data_iter.get_batch(batch_idx)
                turn_inputs = create_turn_batch(local_data['inputs'])
                kb_inputs = create_kb_batch(local_data['kbs'])
                assert len(turn_inputs) == local_data['max_turn']

                if model.name == 'muti_agent':
                    if method == '1-3':
                        metrics_list_M, metrics_list_S, metrics_list_TB, metrics_list_TE = \
                                                            model.iterate(turn_inputs, kb_inputs,
                                                                          is_training=False,
                                                                          method=method)
                    else:
                        metrics_list_M, metrics_list_S, metrics_list_TB, metrics_list_TE, _, _ = \
                                                            model.iterate(turn_inputs, kb_inputs,
                                                                          is_training=False,
                                                                          method=method)
                    mm_M.update(metrics_list_M)
                    mm_S.update(metrics_list_S)
                    mm_TB.update(metrics_list_TB)
                    if metrics_list_TE:
                        mm_TE.update(metrics_list_TE)
                    return mm_M, mm_S, mm_TB, mm_TE
                else:
                    metrics_list = model.iterate(turn_inputs, kb_inputs,
                                             use_rl=use_rl, is_training=False)
                    mm.update(metrics_list)
                    return mm

    def save(self, is_best=False, is_rl=False):
        """
        save the whole model muti-agent todo save three single model separately after train muti-agent
        """
        model_file = os.path.join(self.save_dir, "state_epoch_{}.model".format(self.epoch))
        torch.save(self.model.state_dict(), model_file)
        self.logger.info("Saved model state to '{}'".format(model_file))

        train_file = os.path.join(self.save_dir, "state_epoch_{}.train".format(self.epoch))
        train_state = {"epoch": self.epoch,
                       "batch_num": self.batch_num,
                       "best_valid_metric": self.best_valid_metric,
                       "optimizer": self.optimizer[0].state_dict() if isinstance(self.optimizer, tuple) else
                       self.optimizer.state_dict()}
        if self.lr_scheduler is not None:
            train_state["lr_scheduler"] = self.lr_scheduler.state_dict()
        torch.save(train_state, train_file)
        self.logger.info("Saved train state to '{}'".format(train_file))

        if is_best:
            if is_rl:
                best_model_file = os.path.join(self.save_dir, "best_rl.model")
                best_train_file = os.path.join(self.save_dir, "best_rl.train")
            else:
                best_model_file = os.path.join(self.save_dir, "best.model")
                best_train_file = os.path.join(self.save_dir, "best.train")
            shutil.copy(model_file, best_model_file)
            shutil.copy(train_file, best_train_file)
            self.logger.info(
                "Saved best model state to '{}' with new best valid metric {}={:.3f}".format(
                    best_model_file, self.valid_metric_name.upper(), self.best_valid_metric))

    def load(self, file_ckpt):
        """
        load the whole model muti-agent
        """
        if os.path.isfile(os.path.join(self.save_dir, file_ckpt)):
            file_prefix = file_ckpt.split('.')[0]
            model_file = "{}/{}.model".format(self.save_dir, file_prefix)
            train_file = "{}/{}.train".format(self.save_dir, file_prefix)

            model_state_dict = torch.load(model_file, map_location=lambda storage, loc: storage)
            self.model.load_state_dict(model_state_dict)
            self.logger.info("Loaded model state from '{}'".format(model_file))

            train_state_dict = torch.load(train_file, map_location=lambda storage, loc: storage)
            self.epoch = train_state_dict["epoch"]
            self.best_valid_metric = train_state_dict["best_valid_metric"]
            self.batch_num = train_state_dict["batch_num"]
            if isinstance(self.optimizer, tuple):
                self.optimizer[0].load_state_dict(train_state_dict["optimizer"])
            else:
                self.optimizer.load_state_dict(train_state_dict["optimizer"])
            if self.lr_scheduler is not None and "lr_scheduler" in train_state_dict:
                self.lr_scheduler.load_state_dict(train_state_dict["lr_scheduler"])
            self.logger.info(
                "Loaded train state from '{}' with (epoch-{} best_valid_metric={:.3f})".format(
                    train_file, self.epoch, self.best_valid_metric))

    def load_per_agent(self, S_ckpt, TE_ckpt, TB_ckpt):
        """
        load pre_train three single model in the muti-agent model as train from scratch
        instead of load the whole model as load()
        """
        if os.path.isfile(os.path.join(self.pre_train_dir, S_ckpt)) and \
                os.path.isfile(os.path.join(self.pre_train_dir, TB_ckpt)):
                ckpts = [S_ckpt, TB_ckpt, TE_ckpt] if self.model.model_TE is not None else [S_ckpt, TB_ckpt]
                for file_ckpt in ckpts:
                    model_file = "{}/{}".format(self.pre_train_dir, file_ckpt)
                    model_state_dict = torch.load(model_file, map_location=lambda storage, loc: storage)
                    if file_ckpt == S_ckpt:
                        self.model.model_S.load_state_dict(model_state_dict)
                        self.logger.info("Loaded model state from '{}'".format(model_file))
                    elif file_ckpt == TB_ckpt:
                        self.model.model_TB.load_state_dict(model_state_dict)
                        self.logger.info("Loaded model state from '{}'".format(model_file))
                    elif file_ckpt == TE_ckpt:
                        if self.model.model_TE is not None:
                            assert os.path.isfile(os.path.join(self.pre_train_dir, TE_ckpt))
                            self.model.model_TE.load_state_dict(model_state_dict)
                            self.logger.info("Loaded model state from '{}'".format(model_file))
                    else:
                        self.logger.info("error!")
        else:
            self.logger.info("Training from scratch must load pre_train three single agent first!")
            sys.exit(0)




