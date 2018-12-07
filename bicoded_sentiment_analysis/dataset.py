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
"""Dataset for loadding NLPCC 2018 Sentiment analysis task bilingual dataset"""
import os
from lxml import html

from mxnet.gluon.data import ArrayDataset


class NLPCCDataset(ArrayDataset):
    def __init__(self, segment='train', root=os.path.join('.', 'data')):
        self._segment = segment
        self._data_file = {'train': ('train.txt', '052a75bf8fdb3e843b8649971658eae8133f9b0e'),
                           'dev': ('dev.txt', 'e31ad736582b72a8eabd5c0b0a38cb779ed913d7'),
                           'test': ('test.txt', 'e31ad736582b72a8eabd5c0b0a38cb779ed913d7')}

        root = os.path.expanduser(root)

        if not os.path.isdir(root):
            os.makedirs(root)

        self._root = root

        super(NLPCCDataset, self).__init__(self._read_data())

    def _read_data(self):
        """Read tweet data into from disk  dataset"""
        records = []
        data_file_name, _ = self._data_file[self._segment]

        with open(os.path.join(self._root, data_file_name), encoding='utf-8', mode='r') as f:
            content = f.read()
            xml_data = html.fragments_fromstring(content)

        for item in xml_data:
            item_data = {}

            for data_tag in item.getchildren():
                item_data[data_tag.tag] = data_tag.text

            item_data = NLPCCDataset._convert_from_string_to_int(item_data)
            records.append((int(item.attrib['id']), item_data['content'],
                            [item_data['happiness'], item_data['sadness'],
                             item_data['anger'], item_data['fear'], item_data['surprise']]))
        return records

    @staticmethod
    def _convert_from_string_to_int(item_data):
        """Parse data from initial format of T/F to 1/0"""
        for k, v in item_data.items():
            if k != 'content':
                item_data[k] = 1 if 'T' in v else 0

        return item_data