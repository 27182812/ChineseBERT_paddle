# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2018 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import random
import time
import json
import math
import argparse
from functools import partial
import numpy as np
import paddle
from paddle.fluid.reader import QUEUE_GET_TIMEOUT

from paddle.io import DataLoader

import paddlenlp as ppnlp

from paddlenlp.data import Pad, Stack, Tuple, Dict
from paddlenlp.transformers import BertForQuestionAnswering, BertTokenizer, ErnieForQuestionAnswering, ErnieTokenizer
from modeling import GlyceBertForQuestionAnswering,GlyceBertModel
from tokenizer import ChineseBertTokenizer

from paddlenlp.transformers import LinearDecayWithWarmup
from paddlenlp.metrics.squad import squad_evaluate, compute_prediction
from paddlenlp.datasets import load_dataset
from datasets.cmrc_2018_dataset import CMRC2018Dataset
from utils.random_seed import set_random_seed


class CrossEntropyLossForSQuAD(paddle.nn.Layer):
    def __init__(self):
        super(CrossEntropyLossForSQuAD, self).__init__()

    def forward(self, y, label):

        start_logits, end_logits = y
        start_position, end_position = label
        # start_position = paddle.unsqueeze(start_position, axis=-1)
        # end_position = paddle.unsqueeze(end_position, axis=-1)
        start_position = paddle.squeeze(start_position, axis=-1)
        end_position = paddle.squeeze(end_position, axis=-1)
        
        start_loss = paddle.nn.functional.cross_entropy(input=start_logits, label=start_position)
        end_loss = paddle.nn.functional.cross_entropy(input=end_logits, label=end_position)
        
        loss = (start_loss + end_loss) / 2
        return loss


def run(args):
    print(args)
    set_random_seed(args.seed)
    paddle.set_device(args.device)
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()
    rank = paddle.distributed.get_rank()


    print(" | Loading pretrained ChineseBERT model pdparams...")
    tokenizer = ChineseBertTokenizer.from_pretrained(args.config_path)
    model = GlyceBertForQuestionAnswering.from_pretrained(args.model_name_or_path)

    print(" | Fineshed loading.")

    if paddle.distributed.get_world_size() > 1:
        model = paddle.DataParallel(model)

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

    num_training_steps = args.max_steps if args.max_steps > 0 else len(
        train_data_loader) * args.max_epoch
    max_epoch = math.ceil(num_training_steps /
                                    len(train_data_loader))
    
    print("Num of Train samples:",len(train_data_loader))
    print("Training Steps: ",num_training_steps)

    lr_scheduler = LinearDecayWithWarmup(
        args.lr, num_training_steps, args.warmup_proportion)

    # Generate parameter names needed to perform weight decay.
    # All bias and LayerNorm parameters are excluded.
    decay_params = [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm","LayerNorm.weight"])
    ]
    optimizer = paddle.optimizer.AdamW(
        beta1=0.9,
        beta2=0.98,
        learning_rate=lr_scheduler,
        epsilon=args.adam_epsilon,
        parameters=model.parameters(),
        weight_decay=args.weight_decay,
        apply_decay_param_fun=lambda x: x in decay_params)

    criterion = CrossEntropyLossForSQuAD()

    global_step = 0
    tic_train = time.time()
    print("-"*40)
    print(" | Start Training...")

    global_loss=99999
    for epoch in range(max_epoch):
        for step, batch in enumerate(train_data_loader):
            global_step += 1
            input_ids,pinyin_ids, token_type_ids, start_positions, end_positions = batch
            batch_size, length = input_ids.shape
            pinyin_ids = paddle.reshape(pinyin_ids,[batch_size, length, 8])

            logits = model(input_ids,pinyin_ids, token_type_ids)
            loss = criterion(logits, (start_positions, end_positions))

            if global_step % args.logging_steps == 0:
                print(
                    "global step %d, epoch: %d, batch: %d, loss: %f, speed: %.2f step/s"
                    % (global_step, epoch + 1, step + 1, loss,
                        args.logging_steps / (time.time() - tic_train)))
                tic_train = time.time()
            
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.clear_grad()

            if global_step % args.logging_steps == 0:
                
                tmp_loss = evaluate(model, criterion, dev_data_loader)
                
                if tmp_loss < 0.88:
                    # global_loss = tmp_loss
                    # if global_loss < 0.9:
                    output_dir = os.path.join(args.output_dir,
                                                "model_%d_loss%.2f" % (global_step,tmp_loss))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    # need better way to get inner model of DataParallel
                    model_to_save = model._layers if isinstance(
                        model, paddle.DataParallel) else model
                    model_to_save.save_pretrained(output_dir)
                    # tokenizer.save_pretrained(output_dir)
                    print('Saving checkpoint to:', output_dir)
               
                if global_step == num_training_steps:
                    break

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
    
    eval_loss = np.mean(losses)

    print(" | Eval loss: %.5f" % eval_loss)
    model.train()


    return eval_loss

def get_parser():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--model_name_or_path",default="ChineseBERT-large",type=str,help="Path to pre-trained model or shortcut name of model.")
    parser.add_argument("--data_dir",default="../data/CMRC2018",type=str,help="Path to datasets.")
    parser.add_argument("--config_path",default="../config",type=str,help="Path to pinyin config file.")
    parser.add_argument("--output_dir", default="../outputs/cmrc2018-v1", type=str, help="train data path")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--max_seq_length", default=512, type=int, help="max length of datasets")
    parser.add_argument("--lr", type=float, default=3e-5, help="learning rate")
    parser.add_argument("--warmup_proportion", default=0.1, type=float, help="warmup proporation")
    parser.add_argument("--workers", type=int, default=4, help="num workers for dataloader")
    parser.add_argument("--weight_decay", default=0.001, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="warmup steps")
    parser.add_argument("--use_memory", action="store_true", help="load datasets to memory to accelerate.")
    parser.add_argument("--save_topk", default=1, type=int, help="save topk checkpoint")
    parser.add_argument("--max_epoch", default=2, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps",default=-1,type=int,help="If > 0: set total number of training steps to perform. Override max_epoch.")
    parser.add_argument("--gpus", default="0,", type=str, help="the index of gpu")
    parser.add_argument("--device", choices=["cpu", "gpu", "xpu"], default="gpu", help="Select which device to train model, defaults to gpu.")
    parser.add_argument("--logging_steps",type=int,default=100,help="Log every X updates steps.")
    parser.add_argument("--save_steps",type=int,default=100,help="Save checkpoint every X updates steps.")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--doc_stride",type=int,default=128,help="When splitting up a long document into chunks, how much stride to take between chunks.")

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    run(args)