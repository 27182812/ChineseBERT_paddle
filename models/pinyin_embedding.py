#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
import numpy as np
import paddle
# from torch import nn
from paddle import nn
# from torch.nn import functional as F
from paddle.nn import functional as F


class PinyinEmbedding(nn.Layer):
    def __init__(self, embedding_size: int, pinyin_out_dim: int, config_path):
        """
            Pinyin Embedding Module
        Args:
            embedding_size: the size of each embedding vector
            pinyin_out_dim: kernel number of conv
        """
        super(PinyinEmbedding, self).__init__()
        with open(os.path.join(config_path, 'pinyin_map.json')) as fin:
            pinyin_dict = json.load(fin)
        self.pinyin_out_dim = pinyin_out_dim
        self.embedding = nn.Embedding(len(pinyin_dict['idx2char']), embedding_size)
        self.conv = nn.Conv1D(in_channels=embedding_size, out_channels=self.pinyin_out_dim, kernel_size=2,
                              stride=1, padding=0)

    def forward(self, pinyin_ids):
        """
        Args:
            pinyin_ids: (bs*sentence_length*pinyin_locs)

        Returns:
            pinyin_embed: (bs,sentence_length,pinyin_out_dim)
        """
        # input pinyin ids for 1-D conv
        embed = self.embedding(pinyin_ids)  # [bs,sentence_length,pinyin_locs,embed_size]
        bs, sentence_length, pinyin_locs, embed_size = embed.shape
        
        # view_embed = embed.view(-1, pinyin_locs, embed_size)  # [(bs*sentence_length),pinyin_locs,embed_size]
        view_embed = paddle.reshape(embed,[-1, pinyin_locs, embed_size])
        
        # input_embed = view_embed.permute(0, 2, 1)  # [(bs*sentence_length), embed_size, pinyin_locs]
        input_embed = paddle.transpose(view_embed,perm=[0,2,1]) # [(bs*sentence_length), embed_size, pinyin_locs]
        
        # conv + max_pooling
        pinyin_conv = self.conv(input_embed)  # [(bs*sentence_length),pinyin_out_dim,H]
        
        pinyin_embed = F.max_pool1d(pinyin_conv, pinyin_conv.shape[-1])  # [(bs*sentence_length),pinyin_out_dim,1]
        
        # return pinyin_embed.view(bs, sentence_length, self.pinyin_out_dim)  # [bs,sentence_length,pinyin_out_dim]
        return paddle.reshape(pinyin_embed,[bs, sentence_length, self.pinyin_out_dim])  # [bs,sentence_length,pinyin_out_dim]


if __name__ == "__main__":
    pyemb = PinyinEmbedding(384,128,"E:\code\比赛\ChineseBERT\ChineseBERT-base\config")

    # pinyinids = paddle.rand([4,20,10])
    pinyinids = paddle.to_tensor(np.random.randint(10,size=(4,20,10)))
    print(pinyinids.shape)
    out = pyemb(pinyinids)
    print(out.shape)