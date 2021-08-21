from datasets.bert_dataset1 import BertDataset
import paddle


tokenizer = BertDataset("E:/ChineseBERT/ChineseBERT_paddle/ChineseBERT-base")

sentence = '我喜欢猫'

input_ids, pinyin_ids = tokenizer.tokenize_sentence(sentence)
length = input_ids.shape[0]

input_ids = paddle.reshape(input_ids,[1,length])
pinyin_ids = paddle.reshape(pinyin_ids,[1, length, 8])
# input_ids = input_ids.view(1, length)
# pinyin_ids = pinyin_ids.view(1, length, 8)

print(length)
print(input_ids)
print(pinyin_ids)

