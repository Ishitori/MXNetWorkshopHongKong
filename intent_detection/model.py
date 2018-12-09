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
from mxnet import nd
from gluonnlp.model import ConvolutionalEncoder, Highway
from mxnet.gluon import Block, nn


class OneNet(Block):
    def __init__(self, elmo, crf, intent_cnt, slots_cnt,
                 embedding_size=1024, prefix=None, params=None):
        super(OneNet, self).__init__(prefix, params)
        self._embedding_size = embedding_size
        self.elmo_container = [elmo]

        with self.name_scope():
            self.crf = crf
            self.elmo_s = self.params.get('elmo_s', shape=(3, 1, 1, 1))
            self.gamma = self.params.get('gamma', shape=(1, 1, 1))
            self.highway = Highway(input_size=embedding_size, num_layers=2)

            self.encoder = ConvolutionalEncoder(embed_size=embedding_size,
                                                num_filters=(100, 100, 100, ),
                                                ngram_filter_sizes=(2, 3, 4, ),
                                                conv_layer_activation='relu',
                                                num_highway=None,
                                                output_size=None)
            self.intent_dense = nn.Dense(units=intent_cnt)
            self.slot_dense = nn.Dense(units=slots_cnt, flatten=False)

    def forward(self, data, hidden_state, mask):
        features, _ = self.elmo_container[0](data, hidden_state, mask)
        combined_features = nd.concat(*[nd.expand_dims(f, axis=0) for f in features], dim=0)

        scaled_elmo_layers = nd.broadcast_mul(lhs=combined_features,
                                              rhs=self.elmo_s.data().softmax(axis=0))
        scaled_elmo_embedding = nd.broadcast_mul(lhs=nd.sum(scaled_elmo_layers, axis=0),
                                                 rhs=self.gamma.data())

        highway_output = self.highway(scaled_elmo_embedding.reshape(shape=(-1,
                                                                           self._embedding_size)))
        highway_output = highway_output.reshape(shape=scaled_elmo_embedding.shape)
        encoded_data = self.encoder(highway_output.transpose(axes=(1, 0, 2)))
        intents = self.intent_dense(encoded_data)
        slots = self.slot_dense(highway_output)

        return intents, slots
