#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os
import numpy as np
import random
from functools import partial
import time


import paddle

from paddle.nn import functional as F

from paddle.nn.layer import CrossEntropyLoss

from paddle.io import DataLoader

from paddlenlp.transformers import LinearDecayWithWarmup

from datasets.collate_functions import collate_to_max_length
from datasets.xnli_dataset import XNLIDataset
from modeling import GlyceBertForSequenceClassification
from utils.random_seed import set_random_seed

set_random_seed(2333)


def get_parser():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--bert_path", required=True, type=str, help="bert config file")
    parser.add_argument("--data_dir", required=True, type=str, help="train data path")
    parser.add_argument("--save_path", required=True, type=str, help="train data path")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="learning rate")
    parser.add_argument("--workers", type=int, default=3, help="num workers for dataloader")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="warmup steps")
    parser.add_argument("--use_memory", action="store_true", help="load datasets to memory to accelerate.")
    parser.add_argument("--max_length", default=512, type=int, help="max length of datasets")
    parser.add_argument("--checkpoint_path", type=str, help="train checkpoint")
    parser.add_argument("--save_topk", default=1, type=int, help="save topk checkpoint")
    parser.add_argument("--mode", default='train', type=str, help="train or evaluate")
    parser.add_argument("--warmup_proporation", default=0.01, type=float, help="warmup proporation")
    parser.add_argument("--max_epoch", default=3, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--gpus", default="0,", type=str, help="the index of gpu")
    parser.add_argument("--device", choices=["cpu", "gpu", "xpu"], default="gpu", help="Select which device to train model, defaults to gpu.")
    parser.add_argument("--init_from_ckpt", type=str, default=None, help="The path of checkpoint to be loaded.")
    parser.add_argument("--is_init_from_ckpt", type=bool, default=False, help="Whether the path of checkpoint to be loaded.")
    return parser

def do_train():
    parser = get_parser()
    args = parser.parse_args()
    # paddle.set_device(args.device)
    rank = paddle.distributed.get_rank()
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    model = GlyceBertForSequenceClassification.from_pretrained("ChineseBERT-large",num_classes=4)
    
    if args.is_init_from_ckpt and os.path.exists(args.init_from_ckpt):
        state_dict = paddle.load(args.init_from_ckpt)
        model.set_dict(state_dict)
         
    model = paddle.DataParallel(model)

    prefix = "train"

    train_dataset = XNLIDataset(data_path=os.path.join(args.data_dir, prefix + '.tsv'),
                                     chinese_bert_path=args.bert_path,
                                     max_length=args.max_length)

    dev_dataset = XNLIDataset(data_path=os.path.join(args.data_dir + "/dev" + '.tsv'),
                                     chinese_bert_path=args.bert_path,
                                     max_length=args.max_length)

    train_data_loader = DataLoader(
        dataset= train_dataset,
        batch_size= args.batch_size,
        num_workers= args.workers,
        collate_fn= partial(collate_to_max_length, fill_values=[0, 0, 0]),
        drop_last= False
    )
    dev_data_loader = DataLoader(
        dataset= dev_dataset,
        batch_size= args.batch_size,
        num_workers= args.workers,
        collate_fn= partial(collate_to_max_length, fill_values=[0, 0, 0]),
        drop_last= False
    )

    t_total = len(train_data_loader) * args.max_epoch

    warmup_steps = int(args.warmup_proporation * t_total)

    lr_scheduler = LinearDecayWithWarmup(args.lr, t_total, warmup_steps)

    decay_params = [
        p.name for n, p in model.named_parameters() 
        if not any(nd in n for nd in ["bias", "norm2.weight"])
    ]

    optimizer = paddle.optimizer.AdamW(
        learning_rate= lr_scheduler,
        parameters= model.parameters(),
        weight_decay= args.weight_decay,
        apply_decay_param_fun= lambda x: x in decay_params)
    

    criterion = CrossEntropyLoss()
    metric = paddle.metric.Accuracy()

    global_step = 0
    tic_train = time.time()
    for epoch in range(1, args.max_epoch + 1):
        for step, batch in enumerate(train_data_loader, start=1):
            input_ids, pinyin_ids, labels = batch
            batch_size, length = input_ids.shape
            pinyin_ids = paddle.reshape(pinyin_ids,[batch_size, length, 8])      
            
            y_hat = model(input_ids, pinyin_ids)
            # print("y_hat", y_hat)
            # print("labels", y_hat)
            
            loss = criterion(y_hat, labels)
          
            predict_scores = F.softmax(y_hat, axis=1)
      
            correct = metric.compute(predict_scores, labels)
            metric.update(correct)
            acc = metric.accumulate()

            global_step += 1
            if global_step % 100 == 0 and rank == 0:
                print(
                    "global step %d, epoch: %d, batch: %d, loss: %.5f, acc: %.5f, speed: %.2f step/s"
                    % (global_step, epoch, step, loss, acc, 10 / (time.time() - tic_train)))
                tic_train = time.time()
               
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.clear_grad()
            
            if global_step % 100 == 0:
                evaluate(model, criterion, metric, dev_data_loader) 
            
            if global_step % 300 == 0 and rank == 0:
                save_dir = os.path.join(args.save_path, "model_%d" % global_step)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                # evaluate(model, criterion, metric, dev_data_loader) 
                save_param_path = os.path.join(save_dir, "model_state.pdparams")
                paddle.save(model.state_dict(), save_param_path)
                ###??
                # tokenizer.save_pretrained(save_dir)

@paddle.no_grad()
def evaluate(model, criterion, metric, data_loader):
    """
    Given a dataset, it evals model and computes the metric.

    Args:
        model(obj:`paddle.nn.Layer`): A model to classify texts.
        criterion(obj:`paddle.nn.Layer`): It can compute the loss.
        metric(obj:`paddle.metric.Metric`): The evaluation metric.
        data_loader(obj:`paddle.io.DataLoader`): The dataset loader which generates batches.
    """
    model.eval()
    metric.reset()
    losses = []
    for batch in data_loader:
        input_ids, pinyin_ids, labels = batch
        batch_size, length = input_ids.shape
        pinyin_ids = paddle.reshape(pinyin_ids,[batch_size, length, 8])
        logits = model(input_ids, pinyin_ids)
        loss = criterion(logits, labels)
        losses.append(loss.numpy())
        correct = metric.compute(logits, labels)
        metric.update(correct)
        accu = metric.accumulate()
    print("eval loss: %.5f, accu: %.5f" % (np.mean(losses), accu))
    model.train()
    metric.reset()




if __name__ == '__main__':
    from multiprocessing import freeze_support

    freeze_support()
    do_train()
