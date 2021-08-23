# encoding: utf-8


# import torch
import paddle
import numpy as np
from typing import List


def collate_to_max_length(batch, max_len = None, fill_values = None):
    """
    pad to maximum length of this batch
    Args:
        batch: a batch of samples, each contains a list of field data(Tensor), which shape is [seq_length]
        max_len: specify max length
        fill_values: specify filled values of each field
    Returns:
        output: list of field batched data, which shape is [batch, max_length]
    """
    # [batch, num_fields]
    lengths = np.array([[len(field_data) for field_data in sample] for sample in batch])
    batch_size, num_fields = lengths.shape
    fill_values = fill_values or [0.0] * num_fields
    # [num_fields]
    max_lengths = lengths.max(axis=0)
    if max_len:
        assert max_lengths.max() <= max_len
        max_lengths = np.ones_like(max_lengths) * max_len

    output = [paddle.full([batch_size, max_lengths[field_idx]],
                         fill_value=fill_values[field_idx],
                         dtype=batch[0][field_idx].dtype)
              for field_idx in range(num_fields)]
    for sample_idx in range(batch_size):
        for field_idx in range(num_fields):
            # seq_length
            data = batch[sample_idx][field_idx]
            # print(data)
            # print("1"*20)
            # print(output[field_idx][sample_idx])
            # print("2"*20)
            # print(data.shape[0])
            # print("3"*20)
            # print(output[field_idx][sample_idx][: data.shape[0]])
            # print("*"*20)
            for i in range(0, data.shape[0]):
                output[field_idx][sample_idx,i] = data[i]
            # print(output[field_idx][sample_idx][: data.shape[0]])
            # output[field_idx][sample_idx][ : data.shape[0]] = data
    return output
