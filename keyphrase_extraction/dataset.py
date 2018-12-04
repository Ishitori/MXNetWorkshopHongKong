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
"""Dataset for INSPEC data source"""
import os
import re

from mxnet.gluon.data import ArrayDataset


class INSPECDataset(ArrayDataset):
    def __init__(self, segment='train', root_dir='./inspec/all'):
        self._root_dir = root_dir
        self._segment = segment
        self._data_split = {'train': lambda data: data[:1000],
                           'dev': lambda data: data[1000:1500],
                           'test': lambda data: data[1500:]}

        super(INSPECDataset, self).__init__(self._read_data())

    def _read_data(self):
        """Read INSPEC data from disk to memory """
        articles = os.listdir(self._root_dir)
        all_data = []

        text_articles = INSPECDataset._get_files_by_extension(articles, '.abstr')
        keyp_articles = INSPECDataset._get_files_by_extension(articles, '.uncontr')

        for article_id in range(len(text_articles)):
            a = text_articles[article_id].split('.')[0]
            b = keyp_articles[article_id].split('.')[0]
            assert a == b

            with open(os.path.join(self._root_dir, text_articles[article_id]), 'r') as article_file:
                article = article_file.read().strip()

            article_words = INSPECDataset._get_article_words(article)

            with open(os.path.join(self._root_dir, keyp_articles[article_id]), 'r') as keyp_file:
                keyp = keyp_file.read().strip().replace('; ', ' ')

            keyphrases = [x.lower() for x in keyp.split()]

            label = INSPECDataset._get_article_label(article_words, keyphrases)
            all_data.append((article_words, label))

        return self._data_split[self._segment](all_data)

    @staticmethod
    def _get_article_label(article_words, keyphrases):
        """Generate label [0, 1 or 2] for all words in article depending if this word is in the
        keyphrases or not"""
        label = []

        for i, word in enumerate(article_words):
            if word not in keyphrases:
                label.append(0)  # NO_KP

            elif word in keyphrases and article_words[i - 1] not in keyphrases:
                label.append(1)  # BEGIN_KP

            else:
                label.append(2)  # INSIDE_KP

        return label

    @staticmethod
    def _get_article_words(article):
        article = re.sub('[!"#$%&\'()*+,./:;<=>?@[\\]^_`{|}~]+', '', article)
        article = article.replace('\\n', ' ').replace('\\t', ' ').replace('\\r', ' ')
        words = [x.lower() for x in article.split()]
        return words

    @staticmethod
    def _get_files_by_extension(articles, extension):
        files_with_extension = [a for a in articles if a.endswith(extension)]
        files_with_extension.sort()
        return files_with_extension
