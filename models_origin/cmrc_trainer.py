#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import collections
import json
import os
from pickle import TRUE, PicklingError
import time
import random
from functools import partial
import numpy as np 
import paddle
from paddle.nn.layer import CrossEntropyLoss
from paddle.nn import functional as F
# from tokenizers import BertWordPieceTokenizer,BertPreTokenizer
from paddle.io import DataLoader
# from transformers import AdamW, BertConfig, get_linear_schedule_with_warmup
from paddlenlp.transformers import LinearDecayWithWarmup

from datasets.cmrc_2018_dataset import CMRC2018Dataset
from datasets.collate_functions import collate_to_max_length
from modeling import GlyceBertForQuestionAnswering,GlyceBertModel
from utils.random_seed import set_random_seed

from paddlenlp.datasets import CMRC2018


set_random_seed(2333)
RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "start_logits", "end_logits"])

def get_parser():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--bert_path", required=True, type=str, help="bert config file")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--lr", type=float, default=3e-5, help="learning rate")
    parser.add_argument("--workers", type=int, default=4, help="num workers for dataloader")
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="warmup steps")
    parser.add_argument("--use_memory", action="store_true", help="load datasets to memory to accelerate.")
    parser.add_argument("--max_length", default=512, type=int, help="max length of datasets")
    parser.add_argument("--data_dir", required=True, type=str, help="train data path")
    parser.add_argument("--save_path", required=True, type=str, help="train data path")
    parser.add_argument("--save_topk", default=1, type=int, help="save topk checkpoint")
    parser.add_argument("--warmup_proporation", default=0.01, type=float, help="warmup proporation")
    parser.add_argument("--max_epoch", default=3, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--gpus", default="0,", type=str, help="the index of gpu")
    parser.add_argument("--device", choices=["cpu", "gpu", "xpu"], default="gpu", help="Select which device to train model, defaults to gpu.")
    parser.add_argument("--init_from_ckpt", type=str, default=None, help="The path of checkpoint to be loaded.")
    parser.add_argument("--is_init_from_ckpt", type=bool, default=False, help="Whether to load the existing checkpoint to continue to train.")
    return parser

class CrossEntropyLossForCMRC2018(paddle.nn.Layer):
    def __init__(self):
        super(CrossEntropyLossForCMRC2018, self).__init__()

    def forward(self, y, label):
        start_logits, end_logits = y
        
        start_position, end_position = label

        start_position = paddle.squeeze(start_position, axis=-1)
        end_position = paddle.squeeze(end_position, axis=-1)

        ignored_index = start_logits.shape[1]
        start_position = paddle.clip(start_position,0,ignored_index)
        end_position = paddle.clip(end_position,0,ignored_index)

        start_loss  = paddle.nn.functional.cross_entropy(input=start_logits, label=start_position,ignore_index=ignored_index)
        end_loss  = paddle.nn.functional.cross_entropy(input=end_logits, label=end_position,ignore_index=ignored_index)

        # start_loss  = paddle.nn.functional.cross_entropy(input=start_logits, label=start_position, soft_label=False)
        # end_loss  = paddle.nn.functional.cross_entropy(input=end_logits, label=end_position, soft_label=False)
        
        # start_loss = paddle.mean(start_loss)
        # end_loss = paddle.mean(end_loss)

        loss = (start_loss + end_loss) / 2
        return loss

def do_train():
    print("Start training...")
    parser = get_parser()
    args = parser.parse_args()
    paddle.set_device(args.device)
    rank = paddle.distributed.get_rank()
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    # model = GlyceBertForSequenceClassification(model_GlyceBertModel)
    model_GlyceBertModel = GlyceBertModel.from_pretrained("ChineseBERT-large")
    model = GlyceBertForQuestionAnswering(model_GlyceBertModel)
    

    # prefix = "train"

    # dataset = CMRC2018Dataset(data_path=os.path.join(args.data_dir, prefix + '.tsv'),
    #                                  chinese_bert_path=args.bert_path,
    #                                  max_length=args.max_length)
    train_dataset = CMRC2018Dataset(directory=args.data_dir, prefix="train")
    train_data_loader = DataLoader(
        dataset= train_dataset,
        batch_size= args.batch_size,
        num_workers= args.workers,
        shuffle=True
    )

    dev_dataset = CMRC2018Dataset(directory=args.data_dir, prefix="dev")
    dev_data_loader = DataLoader(
        dataset= dev_dataset,
        batch_size= args.batch_size,
        num_workers= args.workers
    )

    accumulate_grad_batches=4

    num_training_steps = len(train_data_loader) * args.max_epoch

    t_total = len(train_data_loader) //accumulate_grad_batches * args.max_epoch

    warmup_steps = int(args.warmup_proporation * t_total)

    lr_scheduler = LinearDecayWithWarmup(args.lr, t_total, warmup_steps)

    decay_params = [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ]
    # scaler = paddle.amp.GradScaler(init_loss_scaling=1024)

    optimizer = paddle.optimizer.AdamW(
        learning_rate= lr_scheduler,
        beta1 = 0.9,
        beta2= 0.98,
        epsilon = args.adam_epsilon,
        parameters= model.parameters(),
        weight_decay= args.weight_decay,
        apply_decay_param_fun= lambda x: x in decay_params)
    

    criterion = CrossEntropyLossForCMRC2018()

    global_step = 0
    losses = []
    tic_train = time.time()
    for epoch in range(1, args.max_epoch + 1):
        for step, batch in enumerate(train_data_loader, start=1):
            global_step += 1
            # input_ids, pinyin_ids, labels = batch
            # input_ids, pinyin_ids, input_mask, span_mask, segment_ids, start, end = batch
            input_ids, pinyin_ids, token_type_ids, start_positions, end_positions = batch

            batch_size, length = input_ids.shape
            pinyin_ids = paddle.reshape(pinyin_ids,[batch_size, length, 8])
            
            logits =  model(input_ids, pinyin_ids,token_type_ids)   # (start_logits,end_logits)

            loss = criterion(logits, (start_positions,end_positions))
            loss = paddle.mean(loss)

            # losses.append(loss.numpy()[0])

            if global_step % 100 == 0 and rank == 0:
                print(
                    "global step %d, epoch: %d, loss: %.5f, speed: %.2f step/s"
                    % (global_step, epoch, loss, 10 / (time.time() - tic_train)))
                tic_train = time.time()

            if global_step % 300 == 0 and rank == 0:
                evaluate(model, criterion, dev_data_loader)


            if global_step % 300 == 0 and rank == 0:
                save_dir = os.path.join(args.save_path, "model_%d" % global_step)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_param_path = os.path.join(save_dir, "model_state.pdparams")
                paddle.save(model.state_dict(), save_param_path)

            loss.backward()
            lr_scheduler.step()
            
            # compute loss
            if (step+1)% accumulate_grad_batches ==0:

                optimizer.step()
                optimizer.clear_grad()


@paddle.no_grad()
def evaluate(model, criterion, data_loader):
    """
    Given a dataset, it evals model and computes the metric.

    Args:
        model(obj:`paddle.nn.Layer`): A model to classify texts.
        criterion(obj:`paddle.nn.Layer`): It can compute the loss.
        metric(obj:`paddle.metric.Metric`): The evaluation metric.
        data_loader(obj:`paddle.io.DataLoader`): The dataset loader which generates batches.
    """
    model.eval()
    # metric.reset()
    losses = []
    for batch in data_loader:
        input_ids, pinyin_ids, token_type_ids, start_positions, end_positions = batch
        batch_size, length = input_ids.shape
        pinyin_ids = paddle.reshape(pinyin_ids,[batch_size, length, 8])

        logits =  model(input_ids, pinyin_ids,token_type_ids)   # (start_logits,end_logits)
        start_logits,end_logits = logits
        loss = criterion(logits, (start_positions,end_positions))

        losses.append(loss.numpy())

    print(" | Eval loss: %.5f" % (np.mean(losses)))
    model.train()


if __name__ == '__main__':
    do_train()
