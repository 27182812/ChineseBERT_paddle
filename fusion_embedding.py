#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

# import torch
# from torch import nn
import paddle
from paddle import nn 

from glyph_embedding import GlyphEmbedding
from pinyin_embedding import PinyinEmbedding


class FusionBertEmbeddings(nn.Layer):
    """
    Construct the embeddings from word, position, glyph, pinyin and token_type embeddings.
    """

    def __init__(self,
                 vocab_size=23236,
                 hidden_size=768,
                 hidden_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=16,
                 name_or_path="C:/Users/QYS/Desktop/ChineseBert-main/ChineseBERT-base/",
                 layer_norm_eps=1e-12):

        super(FusionBertEmbeddings, self).__init__()
        config_path = os.path.join(name_or_path, 'config')
        font_files = []
        for file in os.listdir(config_path):
            if file.endswith(".npy"):
                font_files.append(os.path.join(config_path, file))
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        self.pinyin_embeddings = PinyinEmbedding(embedding_size=128, pinyin_out_dim=hidden_size,
                                                 config_path=config_path)
        self.glyph_embeddings = GlyphEmbedding(font_npy_files=font_files)

        # self.LayerNorm is not snake-cased to stick with TensorFlow models variable name and be able to load
        # any TensorFlow checkpoint file
        self.glyph_map = nn.Linear(1728, hidden_size)
        self.map_fc = nn.Linear(hidden_size * 3, hidden_size)
        #self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.LayerNorm = nn.LayerNorm(hidden_size, epsilon=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        # self.register_buffer("position_ids", torch.arange(max_position_embeddings).expand((1, -1)))
        self.register_buffer("position_ids", paddle.expand(paddle.arange(max_position_embeddings),[1, -1]))

    def forward(self, input_ids=None, pinyin_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        # print(input_ids)
        # exit()
        # print(input_ids.size())
        # print(input_ids.shape)
        # exit()
        if input_ids is not None:
            # input_shape = input_ids.size()
            input_shape = input_ids.shape
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if token_type_ids is None:
            # token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)
            token_type_ids = paddle.zeros_like(input_ids, dtype="int64")

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        # get char embedding, pinyin embedding and glyph embedding
        word_embeddings = inputs_embeds  # [bs,l,hidden_size]
        pinyin_embeddings = self.pinyin_embeddings(pinyin_ids)  # [bs,l,hidden_size]
        glyph_embeddings = self.glyph_map(self.glyph_embeddings(input_ids))  # [bs,l,hidden_size]
        # fusion layer
        # concat_embeddings = torch.cat((word_embeddings, pinyin_embeddings, glyph_embeddings), 2)
        concat_embeddings = paddle.concat((word_embeddings, pinyin_embeddings, glyph_embeddings), 2)
        inputs_embeds = self.map_fc(concat_embeddings)

        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        #embeddings = self.LayerNorm(embeddings)
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
