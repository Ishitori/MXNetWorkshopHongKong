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
"""Data transformer for NLU Benchmark 2017 dataset"""


class DataTransformer:
    def __init__(self, vocab):
        self._vocab = vocab

    def __call__(self, text, tokens, entities, intent):
        processed_tokens = ['<bos>'] + tokens + ['<eos>']
        # 1 and 2 are bos and eos tags
        processed_entities = [1] + entities + [2]

        return self._vocab[processed_tokens], len(processed_tokens), processed_entities, intent
