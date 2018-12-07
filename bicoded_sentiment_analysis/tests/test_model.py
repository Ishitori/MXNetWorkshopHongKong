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
"""Unit tests for model"""
import gluonnlp
from gluonnlp import Vocab, data


# def test_embedding():
#     # readable list of wikipedia for different languages is here:
#     # https://en.wikipedia.org/wiki/List_of_Wikipedias
#     vocab = Vocab(data.Counter(["走", "走秀", "秀"]))
#     fasttext = gluonnlp.embedding.create('fasttext', source='wiki.zh')
#     vocab.set_embedding(fasttext)
#     print(vocab.embedding['秀'])
#
#
# def test_embedding_eng():
#     # readable list of wikipedia for different languages is here:
#     # https://en.wikipedia.org/wiki/List_of_Wikipedias
#     vocab = Vocab(data.Counter(["user", "ut", "vacation"]))
#     fasttext = gluonnlp.embedding.create('fasttext', source='wiki.simple')
#     vocab.set_embedding(fasttext)
#     print(vocab.embedding['vacation'])


def test_join_embedding():
    counter = data.Counter(["love", "走秀", "vacation"])
    vocab1 = Vocab(counter)
    vocab2 = Vocab(counter)
    chinese_embedding = gluonnlp.embedding.create('fasttext', source='wiki.zh')
    eng_embedding = gluonnlp.embedding.create('fasttext', source='wiki.simple')

    vocab1.set_embedding(chinese_embedding)
    vocab2.set_embedding(eng_embedding)

    print(vocab1.embedding['vacation'] + vocab2.embedding['vacation'])

