#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os

import paddle
from paddle.io import Dataset,DataLoader
from torch._C import dtype
from tqdm import tqdm

from tasks.CMRC.processor import read_squad_examples, convert_examples_to_features


class CMRC2018Dataset(Dataset):

    def __init__(self, directory, prefix):
        super().__init__()
        file = os.path.join(directory, prefix + '.json')
        with open(file, 'r', encoding="utf8") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data['input_ids'])

    def __getitem__(self, idx):
        # input_ids = torch.LongTensor(self.data['input_ids'][idx])
        input_ids = paddle.to_tensor(self.data['input_ids'][idx],dtype="int64")

        # pinyin_ids = torch.LongTensor(self.data['pinyin_ids'][idx]).view(-1)
        pinyin_ids = paddle.reshape(paddle.to_tensor(self.data['pinyin_ids'][idx],dtype="int64"),[-1])

        # input_mask = torch.LongTensor(self.data['input_mask'][idx])
        input_mask = paddle.to_tensor(self.data['input_mask'][idx],dtype="int64")

        # span_mask = torch.LongTensor(self.data['span_mask'][idx])
        span_mask = paddle.to_tensor(self.data['span_mask'][idx],dtype="int64")

        # segment_ids = torch.LongTensor(self.data['segment_ids'][idx])
        segment_ids = paddle.to_tensor(self.data['segment_ids'][idx],dtype="int64")

        # start = torch.LongTensor([self.data['start'][idx]])
        start = paddle.to_tensor([self.data['start'][idx]], dtype="int64")
        
        # end = torch.LongTensor([self.data['end'][idx]])
        end = paddle.to_tensor([self.data['end'][idx]],dtype="int64")

        return input_ids, pinyin_ids, input_mask, span_mask, segment_ids, start, end


class CMRC2018EvalDataset(Dataset):

    def __init__(self, bert_path, test_file):
        super().__init__()
        self.examples = read_squad_examples(input_file=test_file, is_training=False)
        vocab_file = os.path.join(bert_path, 'vocab.txt')
        self.samples = convert_examples_to_features(
            bert_path=bert_path,
            examples=self.examples,
            max_seq_length=512,
            doc_stride=128,
            max_query_length=64,
            is_training=False,
            vocab_file=vocab_file,
            do_lower_case=False)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # input_ids = torch.LongTensor(self.samples[idx].input_ids)
        input_ids = paddle.to_tensor(self.samples[idx].input_ids, dtype="int64")

        # pinyin_ids = torch.LongTensor(self.samples[idx].pinyin_ids).view(-1)
        pinyin_ids = paddle.reshape(paddle.to_tensor(self.samples[idx].pinyin_ids, dtype="int64"),[-1])

        # input_mask = torch.LongTensor(self.samples[idx].input_mask)
        input_mask = paddle.to_tensor(self.samples[idx].input_mask, dtype="int64")

        # span_mask = torch.LongTensor(self.samples[idx].input_span_mask)
        span_mask = paddle.to_tensor(self.samples[idx].input_span_mask, dtype="int64")

        # segment_ids = torch.LongTensor(self.samples[idx].segment_ids)
        segment_ids = paddle.to_tensor(self.samples[idx].segment_ids,dtype="int64")

        # unique_ids = torch.LongTensor([self.samples[idx].unique_id])
        unique_ids = paddle.to_tensor([self.samples[idx].unique_id], dtype="int64")

        # indexes = torch.LongTensor([idx])
        indexes = paddle.to_tensor([idx],dtype="int64")
        return input_ids, pinyin_ids, input_mask, span_mask, segment_ids, unique_ids, indexes


def unit_test():
    root_path = "/home/sunzijun/cmrc"
    vocab_file = "/data/nfsdata2/sunzijun/glyce/glyce/bert_chinese_base_large_vocab/vocab.txt"
    config_path = "/data/nfsdata2/sunzijun/glyce/glyce/config"
    prefix = "train"
    dataset = CMRC2018Dataset(directory=root_path, prefix=prefix)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=10,
        num_workers=0,
        shuffle=True
    )
    for input_ids, pinyin_ids, input_mask, span_mask, segment_ids, start, end in tqdm(dataloader):
        bs, length = input_ids.shape
        # print(input_ids.shape)
        # print(pinyin_ids.reshape(bs, length, -1).shape)
        # print(start.view(-1).shape)
        # print(end.view(-1).shape)
        # print()


def eval():
    root_path = "/data/nfsdata2/sunzijun/squad-style-data"
    vocab_file = "/data/nfsdata2/sunzijun/glyce/glyce/bert_chinese_base_large_vocab/vocab.txt"
    config_path = "/data/nfsdata2/sunzijun/glyce/glyce/config"
    prefix = "dev"
    dataset = CMRC2018EvalDataset(directory=root_path, prefix=prefix, vocab_file=vocab_file)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=10,
        num_workers=0,
        shuffle=True
    )
    for input_ids, pinyin_ids, input_mask, span_mask, segment_ids, unique_ids, indexes in tqdm(dataloader):
        bs, length = input_ids.shape
        # print(input_ids.shape)
        # print(pinyin_ids.reshape(bs, length, -1).shape)
        # print(start.view(-1).shape)
        # print(end.view(-1).shape)
        # print()


if __name__ == '__main__':
    eval()
    unit_test()
