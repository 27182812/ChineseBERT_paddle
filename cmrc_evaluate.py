#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import collections
import os
import paddle
from tokenizers import BertWordPieceTokenizer
from paddle.io import DataLoader

from datasets.cmrc_2018_dataset import CMRC2018EvalDataset
from modeling_cmrc import GlyceBertForQuestionAnswering,GlyceBertModel

from tasks.CMRC.processor import write_predictions
from utils.random_seed import set_random_seed

set_random_seed(2333)
RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "start_logits", "end_logits"])



def get_parser():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--bert_path", required=True, type=str, help="bert config file")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--lr", type=float, default=3e-5, help="learning rate")
    parser.add_argument("--workers", type=int, default=0, help="num workers for dataloader")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="warmup steps")
    parser.add_argument("--use_memory", action="store_true", help="load datasets to memory to accelerate.")
    parser.add_argument("--max_length", default=512, type=int, help="max length of datasets")
    parser.add_argument("--test_file", required=True, type=str, help="train data path")
    parser.add_argument("--save_path", required=True, type=str, help="train data path")
    parser.add_argument("--save_topk", default=1, type=int, help="save topk checkpoint")
    parser.add_argument("--task", default='cmrc', type=str, help="checkpoint path")
    parser.add_argument("--pretrain_checkpoint", default="", type=str, help="train data path")
    parser.add_argument("--warmup_proporation", default=0.01, type=float, help="warmup proporation")
    return parser


def evaluate():
    parser = get_parser()
    args = parser.parse_args()
    # paddle.device.set_device("cpu")

    # checkpoint = torch.load(args.pretrain_checkpoint, map_location=torch.device('cpu'))
    # model_GlyceBertModel = GlyceBertModel.from_pretrained(args.pretrain_checkpoint,config_path=config_path)
    # model = GlyceBertForQuestionAnswering.from_pretrained(args.pretrain_checkpoint)
    model_GlyceBertModel = GlyceBertModel.from_pretrained("ChineseBERT-large")
    model = GlyceBertForQuestionAnswering(model_GlyceBertModel)

    state_dict = paddle.load(args.pretrain_checkpoint)
    model.set_state_dict(state_dict)
    model.eval()

    dataset = CMRC2018EvalDataset(bert_path = args.bert_path, test_file = args.test_file)
    dataloader = DataLoader(
        dataset = dataset,
        batch_size = args.batch_size,
        num_workers = args.workers
    )
    all_results = []
    for step, batch in enumerate(dataloader):
        input_ids, pinyin_ids, input_mask, span_mask, segment_ids, unique_ids, indexes = batch
        batch_size, length = input_ids.shape
        pinyin_ids = paddle.reshape(pinyin_ids,[batch_size, length, 8])
        output =  model(input_ids, pinyin_ids,segment_ids) 

        start_logits, end_logits = output[0], output[1]
        for i in range(input_ids.shape[0]):
            all_results.append(
                RawResult(
                    unique_id=int(unique_ids[i][0]),
                    start_logits=start_logits[i].cpu().tolist(),
                    end_logits=end_logits[i].cpu().tolist()))

    eval_examples = dataset.examples
    eval_features = dataset.samples

    n_best_size = 20
    max_answer_length = 30
    do_lower_case = False

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    output_prediction_file = os.path.join(args.save_path, "test_predictions.json")
    output_nbest_file = os.path.join(args.save_path, "test_nbest_predictions.json")
    write_predictions(eval_examples, eval_features, all_results,
                        n_best_size, max_answer_length,
                        do_lower_case, output_prediction_file,
                        output_nbest_file)

    print("Generate prediction file sucefully.")

if __name__ == '__main__':


    from multiprocessing import freeze_support

    freeze_support()
    evaluate()
