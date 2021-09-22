from paddlenlp.transformers.bert.modeling import BertModel

from datasets.bert_dataset import BertDataset
from models_origin.modeling_glycebert import GlyceBertModel
import numpy as np

sentence = '欢迎使用paddle'

# pytorch 版模型加载、推理
model_path="E:/code/Code-NLP/ChineseBERT/ChineseBERT-large"
torch_tokenizer = BertDataset(model_path)
chinese_bert = GlyceBertModel.from_pretrained(model_path)

input_ids, pinyin_ids = torch_tokenizer.tokenize_sentence(sentence)
length = input_ids.shape[0]
input_ids = input_ids.view(1, length)
pinyin_ids = pinyin_ids.view(1, length, 8)
# print(input_ids)
# print(pinyin_ids)

chinese_bert.eval()
# # print(chinese_bert)

output_hidden = chinese_bert.forward(input_ids, pinyin_ids)[0]
torch_array = output_hidden.cpu().detach().numpy()

import paddle
import paddlenlp
import numpy as np
from modeling import GlyceBertModel

paddle_model_name = "ChineseBERT-large"


#paddle_model = BertModel.from_pretrained("bert-base-uncased")
paddle_model = GlyceBertModel.from_pretrained(paddle_model_name)


# for v in paddle_model.parameters():
#     print(v.shape)
# print(2222,paddle_model.parameters()[11])
# exit()

# exit()
# for i,j in paddle_model.state_dict().items():
#     print(i,j.shape)
# exit()



from datasets.bert_dataset1 import BertDataset



# tokenizer = BertDataset("./ChineseBERT-large")
tokenizer = BertDataset("E:/code/Code-NLP/ChineseBERT/ChineseBERT-large")
input_ids, pinyin_ids = tokenizer.tokenize_sentence(sentence)

length = input_ids.shape[0]


input_ids = paddle.reshape(input_ids,[1,length])
pinyin_ids = paddle.reshape(pinyin_ids,[1, length, 8])

# print(input_ids)
# print(pinyin_ids)

paddle_model.eval()

# print(paddle_model)



# paddle_inputs = paddle_tokenizer(text)
# paddle_inputs = {k:paddle.to_tensor([v]) for (k, v) in paddle_inputs.items()}
# # print(paddle_inputs)
paddle_outputs = paddle_model(input_ids,pinyin_ids)

paddle_logits = paddle_outputs[0]
paddle_array = paddle_logits.numpy()
print("paddle_prediction_logits shape:{}".format(paddle_array.shape))
print("paddle_prediction_logits:{}".format(paddle_array))


# the output logits should have the same shape
assert torch_array.shape == paddle_array.shape, "the output logits should have the same shape, but got : {} and {} instead".format(torch_array.shape, paddle_array.shape)
diff = torch_array - paddle_array
print(np.amax(abs(diff)))
print(np.mean(abs(diff)))