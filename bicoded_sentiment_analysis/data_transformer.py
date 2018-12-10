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
"""Transformer that create loadable dataset from original NLPCC 2018 Sentiment analysis data"""
import gluonnlp
import nltk
from mxnet import nd

from gluonnlp import Vocab, data

from dataset import NLPCCDataset


class DataTransformer:
    def __init__(self, segments):
        self._tokenizer = lambda s: s.split(' ')
        # self._tokenizer = gluonnlp.data.JiebaTokenizer()
        #self._tokenizer = nltk.parse.corenlp.CoreNLPParser(url='http://localhost:9000')
        self._datasets = [NLPCCDataset(segment) for segment in segments]
        self._word_vocab, self._char_vocab = self._get_vocabs()

    def get_word_vocab(self):
        return self._word_vocab

    def get_char_vocab(self):
        return self._char_vocab

    def _get_vocabs(self):
        word_list = []
        char_list = []

        for ds in self._datasets:
            for item in ds:
                words = self._get_word_tokens(item[1])
                word_list.extend(words)

                for word in words:
                    char_list.extend(iter(word))

        word_counter = data.count_tokens(word_list)
        char_counter = data.count_tokens(char_list)

        word_vocab = Vocab(word_counter)
        char_vocab = Vocab(char_counter)

        # embedding_zh = gluonnlp.embedding.create('fasttext', source='cc.zh.300')
        # embedding_eng = gluonnlp.embedding.create('fasttext', source='cc.en.300')
        # embedding_ko = gluonnlp.embedding.create('fasttext', source='cc.ko.300')
        # word_vocab.set_embedding(embedding_eng, embedding_zh, embedding_ko)
        #
        # count = 0
        # for token, times in word_counter.items():
        #     if (word_vocab.embedding[token].sum() != 0).asscalar():
        #         count += 1
        #     else:
        #         print(token)
        #
        # print("{}/{} words have embeddings".format(count, len(word_counter)))

        return word_vocab, char_vocab

    def __call__(self, rec_id, text, label):
        tokens = self._get_word_tokens(text)
        token_indices = self._word_vocab[tokens]
        # char_indices = self._char_vocab[[char for t in tokens for char in iter(t.strip())]]
        return rec_id, token_indices, label

    def _get_word_tokens(self, text):
        return [w.strip() for w in self._tokenizer(text.replace('\n\t', ''))
                if len(w.strip()) > 0]
