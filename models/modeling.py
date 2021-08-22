# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from math import pi
import paddle
import paddle.nn as nn
import paddle.tensor as tensor
import paddle.nn.functional as F
from paddle.nn import TransformerEncoder, Linear, Layer, Embedding, LayerNorm, Tanh
from paddlenlp.transformers.bert.modeling import BertPooler,BertPretrainedModel,BertLMPredictionHead,BertPretrainingHeads
from fusion_embedding import FusionBertEmbeddings


__all__ = [
    'GlyceBertModel',
    "GlyceBertPretrainedModel",
    'GlyceBertForPretraining',
    'GlyceBertPretrainingCriterion',
    'GlyceBertPretrainingHeads',
    'GlyceBertForSequenceClassification',
    'GlyceBertForTokenClassification',
    'GlyceBertForQuestionAnswering',
]


# class BertEmbeddings(Layer):
#     """
#     Include embeddings from word, position and token_type embeddings
#     """

#     def __init__(self,
#                  vocab_size,
#                  hidden_size=768,
#                  hidden_dropout_prob=0.1,
#                  max_position_embeddings=512,
#                  type_vocab_size=16):
#         super(BertEmbeddings, self).__init__()
#         self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
#         self.position_embeddings = nn.Embedding(max_position_embeddings,
#                                                 hidden_size)
#         self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)
#         self.layer_norm = nn.LayerNorm(hidden_size)
#         self.dropout = nn.Dropout(hidden_dropout_prob)

#     def forward(self, input_ids, token_type_ids=None, position_ids=None):
#         if position_ids is None:
#             ones = paddle.ones_like(input_ids, dtype="int64")
#             seq_length = paddle.cumsum(ones, axis=-1)

#             position_ids = seq_length - ones
#             position_ids.stop_gradient = True
#         if token_type_ids is None:
#             token_type_ids = paddle.zeros_like(input_ids, dtype="int64")

#         input_embedings = self.word_embeddings(input_ids)
#         position_embeddings = self.position_embeddings(position_ids)
#         token_type_embeddings = self.token_type_embeddings(token_type_ids)

#         embeddings = input_embedings + position_embeddings + token_type_embeddings
#         embeddings = self.layer_norm(embeddings)
#         embeddings = self.dropout(embeddings)
#         return embeddings


class GlyceBertPooler(BertPooler):
    """
    """

    def __init__(self, hidden_size, pool_act="tanh"):
        super(GlyceBertPooler, self).__init__(hidden_size=768)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
        self.pool_act = pool_act

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        if self.pool_act == "tanh":
            pooled_output = self.activation(pooled_output)
        return pooled_output


class GlyceBertPretrainedModel(BertPretrainedModel):
    """
    An abstract class for pretrained BERT models. It provides BERT related
    `model_config_file`, `resource_files_names`, `pretrained_resource_files_map`,
    `pretrained_init_configuration`, `base_model_prefix` for downloading and
    loading pretrained models. See `PretrainedModel` for more details.
    """

    model_config_file = "config.json"
    pretrained_init_configuration = {
        "ChineseBERT-base": {
            "vocab_size": 23236,
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 512,
            "type_vocab_size": 2,
            "initializer_range": 0.02,
            "pad_token_id": 0,

        },
        "ChineseBERT-large": {
            "vocab_size": 23236,
            "hidden_size": 1024,
            "num_hidden_layers": 24,
            "num_attention_heads": 16,
            "intermediate_size": 4096,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 512,
            "type_vocab_size": 2,
            "initializer_range": 0.02,
            "pad_token_id": 0,
        },
    }
    resource_files_names = {"model_state": "model_state.pdparams"}
    pretrained_resource_files_map = {
        "model_state": {
            "ChineseBERT-base":
            "C:/Users/QYS/Desktop/ChineseBert-main/ChineseBERT-base/chinesebert-base.pdparams",
            "ChineseBERT-large":
            "https://paddlenlp.bj.bcebos.com/models/transformers/bert-large-uncased.pdparams",
        }
    }
    # base_model_prefix = "bert"

    def init_weights(self, layer):
        """ Initialization hook """
        if isinstance(layer, (nn.Linear, nn.Embedding)):
            # In the dygraph mode, use the `set_value` to reset the parameter directly,
            # and reset the `state_dict` to update parameter in static mode.
            if isinstance(layer.weight, paddle.Tensor):
                layer.weight.set_value(
                    paddle.tensor.normal(
                        mean=0.0,
                        std=self.initializer_range
                        if hasattr(self, "initializer_range") else
                        self.bert.config["initializer_range"],
                        shape=layer.weight.shape))
        elif isinstance(layer, nn.LayerNorm):
            layer._epsilon = 1e-12


# @register_base_model
class GlyceBertModel(GlyceBertPretrainedModel):
    """
    The bare BERT Model transformer outputting raw hidden-states without any specific head on top.

    This model inherits from :class:`~paddlenlp.transformers.model_utils.PretrainedModel`.
    Check the superclass documentation for the generic methods and the library implements for all its model.

    This model is also a Paddle `paddle.nn.Layer <https://www.paddlepaddle.org.cn/documentation
    /docs/en/api/paddle/fluid/dygraph/layers/Layer_en.html>`__ subclass. Use it as a regular Paddle Layer
    and refer to the Paddle documentation for all matter related to general usage and behavior.

    Args:
        vocab_size (`int`):
            Vocabulary size of the XLNet model. Defines the number of different tokens that can
            be represented by the `inputs_ids` passed when calling XLNetModel.
        hidden_size (`int`, optional):
            Dimensionality of the encoder layers and the pooler layer. Defaults to ``768``.
        num_hidden_layers (`int`, optional):
            Number of hidden layers in the Transformer encoder. Defaults to ``12``.
        num_attention_heads (`int`, optional):
            Number of attention heads for each attention layer in the Transformer encoder.
            Defaults to ``12``.
        intermediate_size (`int`, optional):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
            Defaults to ``3072``.
        hidden_act (`str`, optional):
            The non-linear activation function in the feed-forward layer.
            ``"gelu"``, ``"relu"`` and any other paddle supported activation functions
            are supported. Defaults to ``"gelu"``.
        hidden_dropout_prob (`float`, optional):
            The dropout probability for all fully connected layers in the embeddings and encoder.
            Defaults to ``0.1``.
        attention_probs_dropout_prob (`float`, optional):
            The dropout probability for all fully connected layers in the pooler.
            Defaults to ``0.1``.
        initializer_range (`float`, optional):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
            Defaults to ``0.02``.
    """

    def __init__(self,
                 vocab_size = 23236,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=16,
                 initializer_range=0.02,
                 pad_token_id=0,
                 pool_act="tanh"):
                 
        super(GlyceBertModel, self).__init__()
        self.pad_token_id = pad_token_id
        self.initializer_range = initializer_range
        vocab_size = 23236
        self.embeddings = FusionBertEmbeddings(
            vocab_size, hidden_size, hidden_dropout_prob,
            max_position_embeddings, type_vocab_size)
        encoder_layer = nn.TransformerEncoderLayer(
            hidden_size,
            num_attention_heads,
            intermediate_size,
            dropout=hidden_dropout_prob,
            activation=hidden_act,
            attn_dropout=attention_probs_dropout_prob,
            act_dropout=0)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_hidden_layers)
        self.pooler = GlyceBertPooler(hidden_size, pool_act)
        self.apply(self.init_weights)

    def forward(self,
                input_ids,
                pinyin_ids=None,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None,
                output_hidden_states=False):
        if attention_mask is None:
            attention_mask = paddle.unsqueeze(
                (input_ids == self.pad_token_id
                 ).astype(self.pooler.dense.weight.dtype) * -1e9,
                axis=[1, 2])
        # print(pinyin_ids)
        # print("*"*20)
        embedding_output = self.embeddings(
            input_ids=input_ids,
            pinyin_ids=pinyin_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids)
        if output_hidden_states:
            output = embedding_output
            encoder_outputs = []
            for mod in self.encoder.layers:
                output = mod(output, src_mask=attention_mask)
                encoder_outputs.append(output)
            if self.encoder.norm is not None:
                encoder_outputs[-1] = self.encoder.norm(encoder_outputs[-1])
            pooled_output = self.pooler(encoder_outputs[-1])
        else:
            sequence_output = self.encoder(embedding_output, attention_mask)
            pooled_output = self.pooler(sequence_output)
        if output_hidden_states:
            return encoder_outputs, pooled_output
        else:
            return sequence_output, pooled_output


class GlyceBertForQuestionAnswering(GlyceBertPretrainedModel):
    def __init__(self, bert, dropout=None):
        super(GlyceBertForQuestionAnswering, self).__init__()
        self.bert = bert  # allow bert to be config
        self.classifier = nn.Linear(self.bert.config["hidden_size"], 2)
        self.apply(self.init_weights)

    def forward(self, input_ids,pinyin_ids=None ,token_type_ids=None):
        sequence_output, _ = self.bert(
            input_ids,
            pinyin_ids,
            token_type_ids=token_type_ids,
            position_ids=None,
            attention_mask=None)

        logits = self.classifier(sequence_output)
        logits = paddle.transpose(logits, perm=[2, 0, 1])
        start_logits, end_logits = paddle.unstack(x=logits, axis=0)

        return start_logits, end_logits


class GlyceBertForSequenceClassification(GlyceBertPretrainedModel):
    """
    Model for sentence (pair) classification task with BERT.
    Args:
        bert (BertModel): An instance of BertModel.
        num_classes (int, optional): The number of classes. Default 2
        dropout (float, optional): The dropout probability for output of BERT.
            If None, use the same value as `hidden_dropout_prob` of `BertModel`
            instance `bert`. Default None
    """

    def __init__(self, bert, num_classes = 2, dropout=None):
        super(GlyceBertForSequenceClassification, self).__init__()
        self.num_classes = num_classes
        self.bert = bert  # allow bert to be config
        self.dropout = nn.Dropout(dropout if dropout is not None else
                                  self.bert.config["hidden_dropout_prob"])
        self.classifier = nn.Linear(self.bert.config["hidden_size"],
                                    self.num_classes)
        self.apply(self.init_weights)

    def forward(self,
                input_ids,
                pinyin_ids=None,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None):
        _, pooled_output = self.bert(
            input_ids,
            pinyin_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask)
     
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


class GlyceBertForTokenClassification(GlyceBertPretrainedModel):
    def __init__(self, bert, num_classes = 2, dropout=None):
        super(GlyceBertForTokenClassification, self).__init__()
        self.num_classes = num_classes
        self.bert = bert  # allow bert to be config
        self.dropout = nn.Dropout(dropout if dropout is not None else
                                  self.bert.config["hidden_dropout_prob"])
        self.classifier = nn.Linear(self.bert.config["hidden_size"],
                                    self.num_classes)
        self.apply(self.init_weights)

    def forward(self,
                input_ids,
                pinyin_ids=None,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None):
        sequence_output, _ = self.bert(
            input_ids,
            pinyin_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask)

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        return logits


class GlyceBertLMPredictionHead(BertLMPredictionHead):
    def __init__(self,
                 hidden_size,
                 vocab_size,
                 activation,
                 embedding_weights=None):
        super(GlyceBertLMPredictionHead, self).__init__()
        self.transform = nn.Linear(hidden_size, hidden_size)
        self.activation = getattr(nn.functional, activation)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.decoder_weight = self.create_parameter(
            shape=[vocab_size, hidden_size],
            dtype=self.transform.weight.dtype,
            is_bias=False) if embedding_weights is None else embedding_weights
        self.decoder_bias = self.create_parameter(
            shape=[vocab_size], dtype=self.decoder_weight.dtype, is_bias=True)

    def forward(self, hidden_states, masked_positions=None):
        if masked_positions is not None:
            hidden_states = paddle.reshape(hidden_states,
                                           [-1, hidden_states.shape[-1]])
            hidden_states = paddle.tensor.gather(hidden_states,
                                                 masked_positions)
        # gather masked tokens might be more quick
        hidden_states = self.transform(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = paddle.tensor.matmul(
            hidden_states, self.decoder_weight,
            transpose_y=True) + self.decoder_bias
        return hidden_states


class GlyceBertPretrainingHeads(BertPretrainingHeads):
    def __init__(self,
                 hidden_size,
                 vocab_size,
                 activation,
                 embedding_weights=None):
        super(GlyceBertPretrainingHeads, self).__init__()
        self.predictions = BertLMPredictionHead(hidden_size, vocab_size,
                                                activation, embedding_weights)
        self.seq_relationship = nn.Linear(hidden_size, 2)

    def forward(self, sequence_output, pooled_output, masked_positions=None):
        prediction_scores = self.predictions(sequence_output, masked_positions)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class GlyceBertForPretraining(GlyceBertPretrainedModel):
    def __init__(self, bert):
        super(GlyceBertForPretraining, self).__init__()
        self.bert = bert
        self.cls = BertPretrainingHeads(
            self.bert.config["hidden_size"],
            self.bert.config["vocab_size"],
            self.bert.config["hidden_act"],
            embedding_weights=self.bert.embeddings.word_embeddings.weight)

        self.apply(self.init_weights)

    def forward(self,
                input_ids,
                pinyin_ids=None,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None,
                masked_positions=None):
        with paddle.static.amp.fp16_guard():
            outputs = self.bert(
                input_ids,
                pinyin_ids,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                attention_mask=attention_mask)
            sequence_output, pooled_output = outputs[:2]
            prediction_scores, seq_relationship_score = self.cls(
                sequence_output, pooled_output, masked_positions)
            return prediction_scores, seq_relationship_score


class GlyceBertPretrainingCriterion(paddle.nn.Layer):
    def __init__(self, vocab_size):
        super(GlyceBertPretrainingCriterion, self).__init__()
        # CrossEntropyLoss is expensive since the inner reshape (copy)
        self.loss_fn = paddle.nn.loss.CrossEntropyLoss(ignore_index=-1)
        self.vocab_size = vocab_size

    def forward(self, prediction_scores, seq_relationship_score,
                masked_lm_labels, next_sentence_labels, masked_lm_scale):
        with paddle.static.amp.fp16_guard():
            masked_lm_loss = F.cross_entropy(
                prediction_scores,
                masked_lm_labels,
                reduction='none',
                ignore_index=-1)
            masked_lm_loss = masked_lm_loss / masked_lm_scale
            next_sentence_loss = F.cross_entropy(
                seq_relationship_score, next_sentence_labels, reduction='none')
        return paddle.sum(masked_lm_loss) + paddle.mean(next_sentence_loss)
