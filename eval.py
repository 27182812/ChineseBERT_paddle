import paddle
import paddlenlp
from paddlenlp.transformers.bert.modeling import *
from paddlenlp.transformers.bert.tokenizer import *
import numpy as np
from modeling import GlyceBertModel, GlyceBertForSequenceClassification


paddle_model_name = "./ChineseBERT-base"


paddle_model = GlyceBertModel.from_pretrained(paddle_model_name)

# paddle_model = GlyceBertForSequenceClassification(paddle_model)
# print(paddle_model)
# #print(paddle_model.parameters())
# exit()



from datasets.bert_dataset1 import BertDataset



tokenizer = BertDataset("E:/ChineseBERT/ChineseBERT_paddle/ChineseBERT-base")

sentence = '我喜欢猫'

input_ids, pinyin_ids = tokenizer.tokenize_sentence(sentence)
length = input_ids.shape[0]

input_ids = paddle.reshape(input_ids,[1,length])
pinyin_ids = paddle.reshape(pinyin_ids,[1, length, 8])

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
# assert torch_array.shape == paddle_array.shape, "the output logits should have the same shape, but got : {} and {} instead".format(torch_array.shape, paddle_array.shape)
# diff = torch_array - paddle_array
# print(np.amax(abs(diff)))