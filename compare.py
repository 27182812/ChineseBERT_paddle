from datasets.bert_dataset import BertDataset
from models_origin.modeling_glycebert import GlyceBertModel
import numpy as np

sentence = '欢迎使用paddle'

# pytorch 版模型加载、推理
model_path="E:\code\Code-NLP\ChineseBERT\ChineseBERT-base"
torch_tokenizer = BertDataset(model_path)
chinese_bert = GlyceBertModel.from_pretrained(model_path)

input_ids, pinyin_ids = torch_tokenizer.tokenize_sentence(sentence)
length = input_ids.shape[0]
input_ids = input_ids.view(1, length)
pinyin_ids = pinyin_ids.view(1, length, 8)
# print(input_ids)
# print(pinyin_ids)
# print(input_ids.shape)
# print(pinyin_ids.shape)

chinese_bert.eval()
# # print(chinese_bert)

output_hidden = chinese_bert.forward(input_ids, pinyin_ids)[0]
torch_array = output_hidden.cpu().detach().numpy()
# print("torch_prediction_logits shape:{}".format(torch_array.shape))
# print("torch_prediction_logits:{}".format(torch_array))
# print(output_hidden.shape)
# print(output_hidden)

# paddle 版模型加载、推理
import paddle
from pdchinesebert.modeling import ChineseBertModel
from pdchinesebert.tokenizer import ChineseBertTokenizer

paddle_model_name = "ChineseBERT-base"


paddle_model = ChineseBertModel.from_pretrained(paddle_model_name)
paddle_tokenizer = ChineseBertTokenizer.from_pretrained(paddle_model_name)


paddle_model.eval()

# print(paddle_model)

paddle_inputs = paddle_tokenizer(sentence)
paddle_inputs = {k:paddle.to_tensor([v]) for (k, v) in paddle_inputs.items()}

input_ids = paddle_inputs["input_ids"]
batch_size, length = input_ids.shape
pinyin_ids = paddle.reshape(paddle_inputs["pinyin_ids"], [batch_size, length, 8])
# print(input_ids)
# print(pinyin_ids)
# print(input_ids.shape)
# print(pinyin_ids.shape)

paddle_outputs = paddle_model(input_ids,pinyin_ids)

paddle_logits = paddle_outputs[0]
paddle_array = paddle_logits.numpy()
# print("paddle_prediction_logits shape:{}".format(paddle_array.shape))
# print("paddle_prediction_logits:{}".format(paddle_array))


# the output logits should have the same shape
assert torch_array.shape == paddle_array.shape, "the output logits should have the same shape, but got : {} and {} instead".format(torch_array.shape, paddle_array.shape)
diff = torch_array - paddle_array
print(np.amax(abs(diff)))

print("mean difference:", np.mean(abs(diff)))
print("max difference:", np.amax(abs(diff)))