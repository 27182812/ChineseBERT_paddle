#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import List

import numpy as np
import paddle
from paddle import nn 

class GlyphEmbedding(nn.Layer):
    """Glyph2Image Embedding"""

    def __init__(self, font_npy_files: List[str]):
        super(GlyphEmbedding, self).__init__()
        font_arrays = [
            np.load(np_file).astype(np.float32) for np_file in font_npy_files
        ]
        self.vocab_size = font_arrays[0].shape[0]
        self.font_num = len(font_arrays)
        self.font_size = font_arrays[0].shape[-1]
        # N, C, H, W
        font_array = np.stack(font_arrays, axis=1)
        # print(torch.from_numpy(font_array.reshape([self.vocab_size, -1])))
        # print(paddle.to_tensor(font_array.reshape([self.vocab_size, -1])))
        # exit()

        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.font_size ** 2 * self.font_num,
            # _weight=torch.from_numpy(font_array.reshape([self.vocab_size, -1]))
            #weight_attr=paddle.to_tensor(font_array.reshape([self.vocab_size, -1]))
        )
        self.embedding.weight.set_value(paddle.reshape(paddle.to_tensor(font_array),[self.vocab_size, -1]))


    def forward(self, input_ids):
        """
            get glyph images for batch inputs
        Args:
            input_ids: [batch, sentence_length]
        Returns:
            images: [batch, sentence_length, self.font_num*self.font_size*self.font_size]
        """
        # return self.embedding(input_ids).view([-1, self.font_num, self.font_size, self.font_size])
        return self.embedding(input_ids)


