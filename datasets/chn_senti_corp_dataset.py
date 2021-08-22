#!/usr/bin/env python
# -*- coding: utf-8 -*-

from functools import partial

# import torch
import paddle
# from torch.utils.data import DataLoader
from paddle.io import DataLoader

from chinese_bert_dataset import ChineseBertDataset
from collate_functions import collate_to_max_length


class ChnSentCorpDataset(ChineseBertDataset):

    def get_lines(self):
        with open(self.data_path, 'r', encoding="utf-8") as f:
            lines = f.readlines()
        return lines[1:]

    def __len__(self):
        return len(self.lines)

    def test(self):
        line = self.lines[1]
        label, sentence = line.split('\t', 1)
        sentence = sentence[:self.max_length - 2]
        # convert sentence to ids
        tokenizer_output = self.tokenizer.encode(sentence)
        print(tokenizer_output)
      

    def __getitem__(self, idx):
        line = self.lines[idx]
        label, sentence = line.split('\t', 1)
        sentence = sentence[:self.max_length - 2]
        # convert sentence to ids
        tokenizer_output = self.tokenizer.encode(sentence)
        # print(tokenizer_output)
        # exit()
        bert_tokens = tokenizer_output.ids
        pinyin_tokens = self.convert_sentence_to_pinyin_ids(sentence, tokenizer_output)
        # assert
        assert len(bert_tokens) <= self.max_length
        assert len(bert_tokens) == len(pinyin_tokens)
        # convert list to tensor
        # input_ids = torch.LongTensor(bert_tokens)
        input_ids = paddle.to_tensor(bert_tokens,dtype="int64")
        # pinyin_ids = torch.LongTensor(pinyin_tokens).view(-1)
        pinyin_ids = paddle.reshape(paddle.to_tensor(pinyin_tokens,dtype="int64"),[-1])
        # label = torch.LongTensor([int(label)])
        label = paddle.to_tensor([int(label)],dtype="int64")
        return input_ids, pinyin_ids, label


def unit_test():
    data_path = "E:/ChineseBERT/ChineseBERT_paddle/data/ChnSetiCorp/train.tsv"
    model_path = "E:/ChineseBERT/ChineseBERT_paddle/ChineseBERT-base"
    dataset = ChnSentCorpDataset(data_path=data_path,
                                 chinese_bert_path=model_path)

    print(dataset.test())
    exit()

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=10,
        num_workers=0,
        shuffle=False,
        collate_fn=partial(collate_to_max_length, fill_values=[0, 0, 0])
    )
    for input_ids, pinyin_ids, label in dataloader:
        bs, length = input_ids.shape
        
        print(input_ids.shape)
        print(input_ids)
        # print(pinyin_ids.reshape(bs, length, -1).shape)
        print(paddle.reshape(pinyin_ids,[bs, length, -1]).shape)
        print(pinyin_ids)
        # print(label.view(-1).shape)
        print(paddle.reshape(label,[-1]).shape)
        print(label)
        print()
        break


if __name__ == '__main__':
    unit_test()
