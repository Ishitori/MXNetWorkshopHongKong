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
"""Main file to run training of sentiment analysis code on NLPCC 2018 bilingual data."""
from gluonnlp.model import ConvolutionalEncoder
from mxnet import gluon


class SentimentNet(gluon.HybridBlock):
    """Network for sentiment analysis."""
    def __init__(self, word_vocab, embedding_size=300, sentiments=5,
                 prefix=None, params=None):
        super(SentimentNet, self).__init__(prefix=prefix, params=params)
        with self.name_scope():
            self.word_embedding = gluon.nn.Embedding(input_dim=len(word_vocab),
                                                     output_dim=embedding_size)
            self.dropout = gluon.nn.Dropout(0.1)
            self.word_encoder = ConvolutionalEncoder(embed_size=300,
                                                     num_filters=(100, 100, ),
                                                     ngram_filter_sizes=(1, 2, ),
                                                     conv_layer_activation='relu',
                                                     num_highway=None,
                                                     output_size=None)

            self.output = gluon.nn.HybridSequential()

            with self.output.name_scope():
                self.output.add(gluon.nn.Dense(sentiments, activation='sigmoid'))

    def hybrid_forward(self, F, word_data):
        word_embedded = self.word_embedding(word_data)
        word_embedded = self.dropout(word_embedded)

        # Encoder expect TNC, so we need to do transpose
        word_encoded = self.word_encoder(F.transpose(word_embedded, axes=(1, 0, 2)))
        out = self.output(word_encoded)
        return out
