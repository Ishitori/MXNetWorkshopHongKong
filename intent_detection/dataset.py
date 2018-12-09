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
"""Dataset for NLU Benchmark 2017"""
import json
import os
import re

from mxnet.gluon.data import ArrayDataset


class NLUBenchmarkDataset(ArrayDataset):
    """This class provides access to NLU Benchmark data.
    Each record is of a format: List of text tokens, List of entities for each token, Intent
    List of entities contains indices of intents, and mapping from the index to the name can be
    received from `get_entity_map()` method.

    All entities contains a postfix '_BEGIN' or '_INSIDE' depending on the position of a token:
    first token is always _BEGIN and all next are with _INSIDE

    """
    def __init__(self, tokenizer, segment='train_full', intent_map=None, slots_map=None,
                 intent_to_load=None, root_dir='./data/'):
        self._tokenizer = tokenizer
        self._root_dir = root_dir
        self._segment = segment
        self._intent_to_load = intent_to_load
        self._intent_map = intent_map if intent_map else {'UNKNOWN_INTENT': 0}
        self._entity_map = slots_map if slots_map else None
        self._data_segment_filter = {
            'train_small': lambda filename: re.match('^train_(?!.*_full\.json$)', filename),
            'train_full': lambda filename: re.match('^train_.*_full\.json$', filename),
            'val': lambda filename: re.match('^validate_.*\.json$', filename)
        }

        super(NLUBenchmarkDataset, self).__init__(self._read_data())

    def get_slots_map(self):
        return self._entity_map

    def get_intent_map(self):
        return self._intent_map

    def _read_data(self):
        """Read data from disk to memory"""
        all_data = []

        intent_directories = sorted([item for item in os.listdir(self._root_dir)
                                     if os.path.isdir(os.path.join(self._root_dir, item))])

        intent_index = len(self._intent_map)
        for intent in intent_directories:
            if intent not in self._intent_map:
                self._intent_map[intent] = intent_index
                intent_index += 1
            else:
                intent_index = self._intent_map[intent] + 1

            if self._intent_to_load is not None and self._intent_to_load != intent:
                continue

            intent_data_path = os.path.join(self._root_dir, intent)
            data_files = [f for f in os.listdir(intent_data_path)
                          if self._data_segment_filter[self._segment](f)]

            for file in data_files:
                intent_data = self._parse_file(os.path.join(intent_data_path, file), intent)
                all_data.extend(intent_data)

        records = self._process_entities(all_data)
        return records

    def _parse_file(self, file_path, intent):
        result = []

        with open(file_path, mode='rb') as f:
            content = f.read()

        json_data = json.loads(content.decode('utf8', 'replace'), encoding='utf-8')
        intent_data = json_data[intent]

        for record in intent_data:
            resulting_record = [(data['text'], data['entity'] if 'entity' in data else '')
                                for data in record['data']]
            result.append((resulting_record, self._intent_map[intent]))

        return result

    def _process_entities(self, all_data):
        """Resulting format going to be: (Text tokens), (Entity indices per token), (Intent)"""
        all_texts = []
        all_tokens_components = []
        all_entity_components = []

        for record in all_data:
            token_components = []
            entity_components = []
            record_text = ''
            for record_item in record[0]:
                record_text += record_item[0]
                tokens = self._tokenizer(record_item[0])
                entity = record_item[1]

                for i, token in enumerate(tokens):
                    token_components.append(token)

                    if len(entity) > 0:
                        entity_postfix = '_BEGIN' if i == 0 else '_INSIDE'
                        entity_components.append(entity + entity_postfix)
                    else:
                        entity_components.append('NO_ENTITY')
            all_texts.append(record_text)
            all_tokens_components.append(token_components)
            all_entity_components.append(entity_components)

        if not self._entity_map:
            self._entity_map = self._build_entity_map(all_entity_components)

        entity_indices = [[self._entity_map[component] for component in entity]
                          for entity in all_entity_components]

        result = [(text, tokens, entities, rec[1])
                  for rec, text, tokens, entities in zip(all_data, all_texts,
                                                         all_tokens_components, entity_indices)]

        return result

    def _build_entity_map(self, entities):
        entity_map = {'NO_ENTITY': 0, '<bos>': 1, '<eos>': 2}
        index = 3

        all_posible_entities = [component for entity in entities for component in entity]
        unique_entities = sorted(set(all_posible_entities))

        for entity in unique_entities:
            if entity not in entity_map:
                entity_map[entity] = index
                index += 1

        return entity_map
