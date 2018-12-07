# coding: utf-8

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# pylint: disable=
"""Unit tests for data loading"""

from gluonnlp.data import FixedBucketSampler
from gluonnlp.data.batchify import Tuple, Stack, Pad
from mxnet.gluon.data import DataLoader

from sentiment_analysis.data_transformer import DataTransformer
from sentiment_analysis.dataset import NLPCCDataset

data_root_dir = '../data/'


def test_dataset():
    dataset = NLPCCDataset('train', data_root_dir)
    assert 6000 == len(dataset)


def test_transformation():
    dataset = NLPCCDataset('train', data_root_dir)
    transformer = DataTransformer(['train'])
    transformed_dataset = dataset.transform(transformer, lazy=False)

    assert 6000 == len(transformed_dataset)


def test_dataloader():
    batch_size = 128
    dataset = NLPCCDataset('train', data_root_dir)
    transformer = DataTransformer(['train'])

    word_vocab = transformer._word_vocab
    transformed_dataset = dataset.transform(transformer, lazy=False)

    batchify_fn = Tuple(Stack(),
                        Pad(axis=0, pad_val=word_vocab[word_vocab.padding_token],
                            ret_length=True),
                        Stack())
    sampler = FixedBucketSampler(lengths=[len(item[1]) for item in transformed_dataset],
                                 batch_size=batch_size,
                                 shuffle=True,
                                 num_buckets=30)

    data_loader = DataLoader(transformed_dataset, batchify_fn=batchify_fn, batch_sampler=sampler)

    for i, (rec_id, (data, original_length), label) in enumerate(data_loader):
        print(data.shape)
        assert data.shape[0] <= 128

