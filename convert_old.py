#encoding=utf8
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import torch
import numpy as np

torch_model_path = "pytorch_model.bin"
torch_state_dict = torch.load("E:\code\Code-NLP\ChineseBERT\ChineseBERT-base\pytorch_model.bin")

paddle_model_path = "E:\code\Code-NLP\ChineseBERT\ChineseBERT-base\chinesebert-base.pdparams"
paddle_state_dict = {}

# State_dict's keys mapping: from torch to paddle
keys_dict = {
    # about embeddings
    # "embeddings.LayerNorm.weight": "embeddings.layer_norm.weight",
    # "embeddings.LayerNorm.bias": "embeddings.layer_norm.bias",

    # about encoder layer
    'encoder.layer': 'encoder.layers',
    'attention.self.query': 'self_attn.q_proj',
    'attention.self.key': 'self_attn.k_proj',
    'attention.self.value': 'self_attn.v_proj',
    'attention.output.dense': 'self_attn.out_proj',
    'attention.output.LayerNorm.weight': 'norm1.weight',
    'attention.output.LayerNorm.bias': 'norm1.bias',
    'intermediate.dense': 'linear1',
    'output.dense': 'linear2',
    'output.LayerNorm.weight': 'norm2.weight',
    'output.LayerNorm.bias': 'norm2.bias',

    # about cls predictions
    'cls.predictions.transform.dense': 'cls.predictions.transform',
    'cls.predictions.decoder.weight': 'cls.predictions.decoder_weight',
    'cls.predictions.decoder.bias': 'cls.predictions.decoder_bias',
    'cls.predictions.transform.LayerNorm.weight': 'cls.predictions.layer_norm.weight',
    'cls.predictions.transform.LayerNorm.bias': 'cls.predictions.layer_norm.bias',
}


for torch_key in torch_state_dict:
    paddle_key = torch_key
    for k in keys_dict:
        if k in paddle_key:
            paddle_key = paddle_key.replace(k, keys_dict[k])
    paddle_key = paddle_key.replace("bert","chinesebert")
    if ('linear' in paddle_key) or ('proj' in  paddle_key) or ('vocab' in  paddle_key and 'weight' in  paddle_key) or ("dense.weight" in paddle_key) or ('transform.weight' in paddle_key) or ('seq_relationship.weight' in paddle_key) or ('map_fc' in  paddle_key and 'weight' in  paddle_key) or ('glyph_map' in  paddle_key and 'weight' in  paddle_key):
        paddle_state_dict[paddle_key] = paddle.to_tensor(torch_state_dict[torch_key].cpu().numpy().transpose())
    else:
        paddle_state_dict[paddle_key] = paddle.to_tensor(torch_state_dict[torch_key].cpu().numpy())

    print("torch: ", torch_key,"\t", torch_state_dict[torch_key].shape)
    print("paddle: ", paddle_key, "\t", paddle_state_dict[paddle_key].shape, "\n")

paddle.save(paddle_state_dict, paddle_model_path)
print("Convert model Sucessfully.")
