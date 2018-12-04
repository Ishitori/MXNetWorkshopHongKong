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
"""Transformer of INSPEC dataset."""
from gluonnlp.data.batchify import Pad


class DataTransformer:
    def __init__(self, vocab, max_length):
        self._vocab = vocab
        self._max_length = max_length
        self._data_padding_token = vocab[vocab.padding_token]
        self._label_padding_token = 0

    def _pad_to_max_length(self, item, pad_val):
        return item.copy() + [pad_val] * (self._max_length - len(item))

    def __call__(self, data, label):
        tokens = [self._vocab.token_to_idx[x] for x in data]

        if len(tokens) < self._max_length:  # padding 0
            tokens = self._pad_to_max_length(tokens, self._data_padding_token)
            label = self._pad_to_max_length(label, self._label_padding_token)
        else:
            tokens = tokens[:self._max_length]
            label = label[:self._max_length]

        return tokens, label
