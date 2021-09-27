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
from collections import OrderedDict
import argparse

huggingface_to_paddle = {

    "embeddings.LayerNorm": "embeddings.layer_norm",
    "encoder.layer": "encoder.layers",
    "attention.self.query": "self_attn.q_proj",
    "attention.self.key": "self_attn.k_proj",
    "attention.self.value": "self_attn.v_proj",
    "attention.output.dense": "self_attn.out_proj",
    "intermediate.dense": "linear1",
    "output.dense": "linear2",
    "attention.output.LayerNorm": "norm1",
    "output.LayerNorm": "norm2",
    "predictions.decoder.": "predictions.decoder_",
    "predictions.transform.dense": "predictions.transform",
    "predictions.transform.LayerNorm": "predictions.layer_norm",
}

donot_transpose = [
    "embeddings.word_embeddings.weight",
    "embeddings.position_embeddings.weight",
    "embeddings.token_type_embeddings.weight",
    "embeddings.pinyin_embeddings.embedding",
    "embeddings.glyph_embeddings.embedding.weight",
    ]

def convert_pytorch_checkpoint_to_paddle(pytorch_checkpoint_path,
                                         paddle_dump_path):

    import torch
    import paddle
    pytorch_state_dict = torch.load(pytorch_checkpoint_path, map_location="cpu")
    paddle_state_dict = OrderedDict()
    for k, v in pytorch_state_dict.items():
        is_transpose = False
        if k[-7:] == ".weight" and not any([d in k for d in donot_transpose]):
            if ".LayerNorm." not in k:
                if v.ndim==2:
                    v = v.transpose(0, 1)
                    is_transpose = True
        oldk = k
        for huggingface_name, paddle_name in huggingface_to_paddle.items():
            k = k.replace(huggingface_name, paddle_name)

        k = k.replace("bert","chinesebert")

        
        print(f"Converting: {oldk} => {k} is_transpose: {is_transpose}")
        paddle_state_dict[k] = v.data.numpy()

    paddle.save(paddle_state_dict, paddle_dump_path)

from paddlenlp.transformers import BertModel
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pytorch_checkpoint_path",
        default="E:\code\Code-NLP\ChineseBERT\ChineseBERT-base\pytorch_model.bin",
        type=str,
        required=False,
        help="Path to the Pytorch checkpoint path.")
    parser.add_argument(
        "--paddle_dump_path",
        default="E:\code\Code-NLP\ChineseBERT\ChineseBERT-base\chinesebert-base.pdparams",
        type=str,
        required=False,
        help="Path to the output Paddle model.")
    args = parser.parse_args()
    convert_pytorch_checkpoint_to_paddle(args.pytorch_checkpoint_path,
                                         args.paddle_dump_path)
