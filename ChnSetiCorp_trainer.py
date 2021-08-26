#!/usr/bin/env python
# -*- coding: utf-8 -*-


import argparse
import json
import time 
import os
import random
from functools import partial

import pytorch_lightning as pl
# import torch
import paddle
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

# from torch.nn import functional as F
from paddle.nn import functional as F
# from torch.nn.modules import CrossEntropyLoss
from paddle.nn.layer import CrossEntropyLoss

# from torch.utils.data.dataloader import DataLoader
from paddle.io import DataLoader

from transformers import AdamW, BertConfig, get_linear_schedule_with_warmup
from paddlenlp.transformers import LinearDecayWithWarmup


from datasets.chn_senti_corp_dataset import ChnSentCorpDataset
from datasets.collate_functions import collate_to_max_length
# from models.modeling_glycebert import GlyceBertForSequenceClassification
from modeling import GlyceBertForSequenceClassification,GlyceBertModel
import numpy as np

from utils.random_seed import set_random_seed

set_random_seed(2333)


class ChnSentiClassificationTask(pl.LightningModule):

    def __init__(
        self,
        args: argparse.Namespace
    ):
        """Initialize a models, tokenizer and config."""
        super().__init__()
        self.args = args
        if isinstance(args, argparse.Namespace):
            self.save_hyperparameters(args)
        self.bert_dir = args.bert_path
        # self.bert_config = BertConfig.from_pretrained(self.bert_dir, output_hidden_states=False)
        self.model = GlyceBertForSequenceClassification.from_pretrained(self.bert_dir)

        self.loss_fn = paddle.nn.loss.CrossEntropyLoss()
        self.metric = paddle.metric.Accuracy()

        # self.acc = pl.metrics.Accuracy(num_classes=self.bert_config.num_labels)

        # self.acc = pl.metrics.Accuracy(num_classes=2)

        gpus_string = self.args.gpus if not self.args.gpus.endswith(',') else self.args.gpus[:-1]
        self.num_gpus = len(gpus_string.split(","))

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
       

        # no_decay = ["bias", "LayerNorm.weight"]
        # optimizer_grouped_parameters = [
        #     {
        #         "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        #         "weight_decay": self.args.weight_decay,
        #     },
        #     {
        #         "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
        #         "weight_decay": 0.0,
        #     },
        # ]
        
        # optimizer = AdamW(optimizer_grouped_parameters,
        #                   betas=(0.9, 0.98),  # according to RoBERTa paper
        #                   lr=self.args.lr,
        #                   eps=self.args.adam_epsilon)
        # optimizer = paddle.optimizer.AdamW(model.parameters(),
        #                     beta1 = 0.9,
        #                     beta2 = 0.98,
        #                     learning_rate=self.args.lr,
        #                     epsilon=self.args.adam_epsilon)

        # t_total = len(self.train_dataloader()) // self.args.accumulate_grad_batches * self.args.max_epochs
        # warmup_steps = int(self.args.warmup_proporation * t_total)

        # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
        #                                             num_training_steps=t_total)

        t_total = len(self.train_dataloader()) // self.args.accumulate_grad_batches * self.args.max_epochs
        warmup_steps = int(self.args.warmup_proporation * t_total)

        lr_scheduler = LinearDecayWithWarmup(self.args.lr, t_total, warmup_steps)

        decay_params = [
            p.name for n, p in model.named_parameters() 
            if not any(nd in n for nd in ["bias", "norm2.weight"])
        ]
    
        optimizer = paddle.optimizer.AdamW(
            learning_rate=lr_scheduler,
            parameters=model.parameters(),
            weight_decay=self.args.weight_decay,
            apply_decay_param_fun=lambda x: x in decay_params)



        # return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
        return optimizer

    def forward(self, input_ids, pinyin_ids):
        """"""
        attention_mask = (input_ids != 0).long()
        return self.model(input_ids, pinyin_ids, attention_mask=attention_mask)

    def compute_loss_and_acc(self, batch):
        input_ids, pinyin_ids, labels = batch
        batch_size, length = input_ids.shape
        # pinyin_ids = pinyin_ids.view(batch_size, length, 8)
        pinyin_ids = paddle.reshape(pinyin_ids,[batch_size, length, 8])
        # y = labels.view(-1)
        y = paddle.reshape(labels,[-1])
        y_hat = self.forward(
            input_ids=input_ids,
            pinyin_ids=pinyin_ids
        )
        # compute loss
        loss = self.loss_fn(y_hat[0], y)
        # compute acc
        predict_scores = F.softmax(y_hat[0], axis=1)

        predict_labels = paddle.argmax(predict_scores, axis=-1)
        
        correct = self.metric.compute(predict_labels, y)
        self.metric.update(correct)
        acc = self.metric.accumulate()

        return loss, acc

    def training_step(self, batch, batch_idx):
        """"""
        loss, acc = self.compute_loss_and_acc(batch)
        tf_board_logs = {
            "train_loss": loss,
            "train_acc": acc,
        }
        return {'loss': loss, 'log': tf_board_logs}

    def validation_step(self, batch, batch_idx):
        """"""
        loss, acc = self.compute_loss_and_acc(batch)
        return {'val_loss': loss, "val_acc": acc}

    def validation_epoch_end(self, outputs):
        """"""
        avg_loss = paddle.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = paddle.stack([x['val_acc'] for x in outputs]).mean() / self.num_gpus
        tensorboard_logs = {'val_loss': avg_loss, 'val_acc': avg_acc}
        print(avg_loss, avg_acc)
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def train_dataloader(self) -> DataLoader:
        return self.get_dataloader("train")

    def val_dataloader(self):
        return self.get_dataloader("dev")

    def get_dataloader(self, prefix="train") -> DataLoader:
        """get training dataloader"""

        dataset = ChnSentCorpDataset(data_path=os.path.join(self.args.data_dir, prefix + '.tsv'),
                                     chinese_bert_path=self.args.bert_path,
                                     max_length=self.args.max_length)

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.workers,
            collate_fn=partial(collate_to_max_length, fill_values=[0, 0, 0]),
            drop_last=False
        )
        return dataloader

    def test_dataloader(self):
        return self.get_dataloader("test")

    def test_step(self, batch, batch_idx):
        loss, acc = self.compute_loss_and_acc(batch)
        return {'test_loss': loss, "test_acc": acc}

    def test_epoch_end(self, outputs):
        test_loss = paddle.stack([x['test_loss'] for x in outputs]).mean()
        test_acc = paddle.stack([x['test_acc'] for x in outputs]).mean() / self.num_gpus
        tensorboard_logs = {'test_loss': test_loss, 'test_acc': test_acc}
        print(test_loss, test_acc)
        return {'test_loss': test_loss, 'log': tensorboard_logs}


def get_parser():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--bert_path", type=str, default="E:/ChineseBERT/ChineseBERT_paddle/ChineseBERT-large",help="bert config file")
    parser.add_argument("--data_dir",  type=str, help="train data path")
    parser.add_argument("--save_path", type=str, help="train data path")
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
    parser.add_argument("--device", choices=["cpu", "gpu", "xpu"], default="cpu", help="Select which device to train model, defaults to gpu.")
    parser.add_argument("--init_from_ckpt", type=str, default=None, help="The path of checkpoint to be loaded.")
    return parser


def main():
    """main"""
    parser = get_parser()
    # print(parser.parse_args())
    # print("*"*20)
    # parser = Trainer.add_argparse_args(parser)
    # print(parser.parse_args())
    # exit()
    args = parser.parse_args()
    # print(args)
    # exit()

    # create save path if doesn't exit
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    model = ChnSentiClassificationTask(args)

    
    # for epoch in range(1, 1+ 1):
    #     for step, batch in enumerate(model.train_dataloader(), start=1):
    #         print(step, batch)

    # exit()

    # checkpoint_callback = ModelCheckpoint(
    #     dirpath=os.path.join(args.save_path, 'checkpoint', '{epoch}-{val_loss:.4f}-{val_acc:.4f}'),
    #     save_top_k=args.save_topk,
    #     save_last=False,
    #     monitor="val_acc",
    #     mode="max",
    # )
    logger = TensorBoardLogger(
        save_dir=args.save_path,
        name='log'
    )

    # save args
    with open(os.path.join(args.save_path, 'checkpoint', "args.json"), 'w') as f:
        args_dict = args.__dict__
        del args_dict['tpu_cores']
        json.dump(args_dict, f, indent=4)

    trainer = Trainer.from_argparse_args(args,
                                         checkpoint_callback=checkpoint_callback,
                                         distributed_backend="ddp",
                                         logger=logger)

    trainer.fit(model)

    # print
    result = trainer.test()
    print(result)

def do_train():
    parser = get_parser()
    args = parser.parse_args()
    paddle.set_device(args.device)
    rank = paddle.distributed.get_rank()
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    model_GlyceBertModel = GlyceBertModel.from_pretrained("./ChineseBERT-base")
    model = GlyceBertForSequenceClassification(model_GlyceBertModel)

    ### init_from_ckpt参数
    if args.init_from_ckpt and os.path.isfile(args.init_from_ckpt):
        state_dict = paddle.load(args.init_from_ckpt)
        model.set_dict(state_dict)
         
    # model = paddle.DataParallel(model)

    prefix = "train"

    dataset = ChnSentCorpDataset(data_path=os.path.join(args.data_dir, prefix + '.tsv'),
                                     chinese_bert_path=args.bert_path,
                                     max_length=args.max_length)

    train_data_loader = DataLoader(
        dataset= dataset,
        batch_size= args.batch_size,
        num_workers= args.workers,
        collate_fn= partial(collate_to_max_length, fill_values=[0, 0, 0]),
        drop_last= False
    )

    num_training_steps = len(train_data_loader) * args.max_epoch

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
            y = paddle.reshape(labels,[-1])
            
            # attention_mask = (input_ids != 0).long()
            attention_mask = paddle.to_tensor((input_ids != 0),dtype="int64")
            # print(input_ids.shape,attention_mask.shape)
            # y_hat = model(input_ids, pinyin_ids,attention_mask=attention_mask)
            y_hat = model(input_ids, pinyin_ids)
            ####### 改
            # y_me = paddle.transpose(y_hat, perm=[1, 0])
            # print(111,y_hat)
            # # print(111,y_me[0])
            # print(222,y)
            # print(333,labels)
            # exit()
            # compute loss
            loss = criterion(y_hat, labels)
            # compute acc
            predict_scores = F.softmax(y_hat, axis=1)

            # predict_labels = paddle.argmax(predict_scores, axis=-1)
        
            correct = metric.compute(predict_scores, labels)
            metric.update(correct)
            acc = metric.accumulate()
            print(acc)

            global_step += 1
            if global_step % 10 == 0 and rank == 0:
                print(
                    "global step %d, epoch: %d, batch: %d, loss: %.5f, acc: %.5f, speed: %.2f step/s"
                    % (global_step, epoch, step, loss, acc, 10 / (time.time() - tic_train)))
                tic_train = time.time()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.clear_grad()
            if global_step % 100 == 0 and rank == 0:
                save_dir = os.path.join(args.save_dir, "model_%d" % global_step)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_param_path = os.path.join(save_dir, "model_state.pdparams")
                paddle.save(model.state_dict(), save_param_path)
                ###??
                # tokenizer.save_pretrained(save_dir)

def do_test():
    parser = get_parser()
    args = parser.parse_args()
    # paddle.set_device(args.device)
    args.data_dir = "E:/ChineseBERT/ChineseBERT_paddle/data/ChnSetiCorp"

    model = GlyceBertForSequenceClassification.from_pretrained("ChineseBERT-large")
   
        

    prefix = "test"
    
    dataset = ChnSentCorpDataset(data_path=os.path.join(args.data_dir, prefix + '.tsv'),
                                     chinese_bert_path=args.bert_path,
                                     max_length=args.max_length)

    test_data_loader = DataLoader(
        dataset= dataset,
        batch_size= args.batch_size,
        num_workers= args.workers,
        collate_fn= partial(collate_to_max_length, fill_values=[0, 0, 0]),
        drop_last= False
    )

    # params_path = './ChineseBERT-large/ChineseBERT-large.pdparams'

    # if params_path and os.path.isfile(params_path):
    #     state_dict = paddle.load(params_path)
    #     model.set_dict(state_dict)
    #     print("Loaded parameters from %s" % params_path)

    label_map = {0: '0', 1: '1'}
    results = []
    model.eval()
    for batch in test_data_loader:
        input_ids, pinyin_ids, qids = batch
        batch_size, length = input_ids.shape
        pinyin_ids = paddle.reshape(pinyin_ids,[batch_size, length, 8])
        logits = model(input_ids, pinyin_ids)
        probs = F.softmax(logits, axis=1)
        print("probs",probs)
        idx = paddle.argmax(probs, axis=1).numpy()
        idx = idx.tolist()
        print("idx",idx)

        labels = [label_map[i] for i in idx]

        qids = qids.numpy().tolist()
        print("qids",qids)

        results.extend(zip(qids, labels))

    res_dir = "./results"
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    with open(os.path.join(res_dir, "ChnSetiCorp.tsv"), 'w', encoding="utf8") as f:
        f.write("true\tprediction\n")
        for qid, label in results:
            f.write(str(qid[0])+"\t"+label+"\n")


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
        input_ids, token_type_ids, labels = batch
        logits = model(input_ids, token_type_ids)
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
    # main()
    # do_train()
    do_test()
