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
"""Compare performance to Snips_metrics_full.json"""
import json
import multiprocessing
import os

import mxnet as mx
from gluonnlp.data.batchify import Tuple, Pad, Stack
from gluonnlp.vocab import ELMoCharVocab
from mxnet.gluon.data import DataLoader

from data_transformer import DataTransformer
from dataset import NLUBenchmarkDataset
from restore_text import get_text_result
from tokenizer import SacreMosesTokenizer
from utils import get_data_mask


def get_all_intents(root_dir='./data'):
    """Read and return list of all possible intents"""
    return [item for item in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, item))]


def get_snips_metrics_for_intent(intent, root_dir='./data'):
    """Read and return SNIPS official evaluation result for intent. Samples_count used to determine
    which file to load and which item in the file to read. The resulting data has a format:
    Tuple(intent_recall, Dict<slot_name, Tuple(precision, recall)>"""
    intent_file_path = os.path.join(root_dir, intent, 'Snips_metrics_full.json')

    with open(intent_file_path, mode='rb') as f:
        content = f.read()

    json_data = json.loads(content.decode('utf8', 'replace'), encoding='utf-8')
    intent_data = json_data['intents'][intent]
    samples_count_key = list(intent_data['recall'].keys())[0]
    intent_recall = intent_data['recall'][samples_count_key][0]

    slots_data = intent_data['slots']

    slots_result = {}

    for slot_name in slots_data.keys():
        slot_precision = slots_data[slot_name]['precision'][samples_count_key][0]
        slot_recall = slots_data[slot_name]['recall'][samples_count_key][0]
        slots_result[slot_name] = (slot_precision, slot_recall)

    return intent_recall, slots_result


def get_ground_truth(intent, root_dir='./data'):
    """Get bits of texts and associated slot with the text. In case no slot is associated with that
    text bit, empty string is returned. The result format is List[List[Tuple(Text, Slot)]]"""
    val_intent_file_path = os.path.join(root_dir, intent, 'validate_{}.json'.format(intent))

    with open(val_intent_file_path, mode='rb') as f:
        content = f.read()

    json_data = json.loads(content.decode('utf8', 'replace'), encoding='utf-8')
    ground_truth = []

    for i, item in enumerate(json_data[intent]):
        ground_truth.append(
            [(data['text'],
              data['entity'] if 'entity' in data else '') for data in item['data']]
        )

    return ground_truth


def calculate_intent_recall(predictions, ground_truth_intent):
    """Calculate recall for intent"""
    tp = len([p for p in predictions if p == ground_truth_intent])
    return tp / len(predictions)


def get_precision_recall_per_slot(ground_truth, predictions, slot_name, slot_code):
    # ground truth item is item id and String
    ground_truth_slots = {item_id: slot_item[0].strip() for item_id, item in
                          enumerate(ground_truth) for slot_item in item
                          if slot_item[1] == slot_name}

    # no such slots in original item
    if len(ground_truth_slots) == 0:
        return float('nan'), float('nan')

    # prediction item is item_id and String
    prediction_slots = {item_id: get_text_result(slot_item[0], item[2], item[3])
                        for item_id, item in enumerate(predictions) for slot_item in item[1]
                        if slot_item[1] == slot_code}

    def dict_compare(d1, d2):
        d1_keys = set(d1.keys())
        d2_keys = set(d2.keys())
        intersect_keys = d1_keys.intersection(d2_keys)
        added = d2_keys - d1_keys
        removed = d1_keys - d2_keys
        modified = {o: (d1[o], d2[o]) for o in intersect_keys if d1[o] != d2[o]}
        same = set(o for o in intersect_keys if d1[o] == d2[o])
        return added, removed, modified, same

    added, removed, modified, same = dict_compare(ground_truth_slots, prediction_slots)
    p_denominator = (len(same) + len(modified) + len(added))
    r_denominator = (len(same) + len(modified) + len(removed))

    precision = float('nan') if p_denominator == 0 else len(same) / p_denominator
    recall = float('nan') if r_denominator == 0 else len(same) / r_denominator
    return precision, recall


def get_model_metrics_for_intent(predictions, ground_truth, ground_truth_intent, slots_map):
    """Compare predictions with ground truth. The prediction considered to be correct only
     on the exact match of the prediction string and ground_truth.

     Parameters
     ----------
         predictions : List[Tuple(Predicted_Intent, List[(List[bits of text], slot)], text]
            Predictions of the model
         ground_truth : List[List[Tuple(Text, Slot)]]
            Ground truth

     Returns
     -------
         Tuple(intent_recall, Dict<slot_name, Tuple(precision, recall)>"""

    assert len(ground_truth) == len(predictions)
    model_intent_recall = calculate_intent_recall([prediction[0] for prediction in predictions],
                                                  ground_truth_intent)

    precision_recall_per_slot = {}

    for slot in slots_map.keys():
        if slot == 'NO_ENTITY' or slot.endswith('_INSIDE'):
            continue

        slot_name = slot.replace('_BEGIN', '')
        precision_slot, recall_slot = get_precision_recall_per_slot(ground_truth,
                                                                    predictions,
                                                                    slot_name,
                                                                    slots_map[slot])
        # snips call this slot differently
        if slot_name == 'timeRange':
            slot_name = 'snips/datetime'

        precision_recall_per_slot[slot_name] = (precision_slot, recall_slot)

    return model_intent_recall, precision_recall_per_slot


def get_predictions(net, true_intent, intent_map, slots_map, context, batch_size):
    """Get predictions for every item in the intent.
    It returns a list where index is same as in validation item. Each record is of following format:
    Tuple(Predicted_Intent, List[(List[bits of text], slot)]"""
    result = []
    idx_to_slot = {v: k for k, v in slots_map.items()}
    idx_to_intent = {v: k for k, v in intent_map.items()}

    intent_dev_dataset = NLUBenchmarkDataset(SacreMosesTokenizer(), 'val', intent_map,
                                             slots_map, intent_to_load=true_intent)
    transformer = DataTransformer(ELMoCharVocab())
    transformed_dev_dataset = intent_dev_dataset.transform(transformer, lazy=False)
    batchify_fn = Tuple(Pad(), Stack(), Pad(), Stack())
    dev_dataloader = DataLoader(transformed_dev_dataset, batch_size=batch_size,
                                num_workers=multiprocessing.cpu_count() - 3,
                                batchify_fn=batchify_fn)

    for i, (data, valid_lengths, entities, intent) in enumerate(dev_dataloader):
        items_per_iteration = data.shape[0]
        length = data.shape[1]

        data = data.as_in_context(context)

        hidden_state = net.elmo_container[0].begin_state(mx.nd.zeros,
                                                         batch_size=items_per_iteration,
                                                         ctx=context)
        mask = get_data_mask(length, valid_lengths, items_per_iteration, context)

        intents, slots = net(data, hidden_state, mask)
        score, slots_seq = net.crf(slots.transpose(axes=(1, 0, 2)))

        intents_prediction = intents.argmax(axis=1).asnumpy()
        slots_prediction = slots_seq.asnumpy()

        for rec_id, pred_intent in enumerate(intents_prediction):
            text = intent_dev_dataset[rec_id][0]
            tokens = intent_dev_dataset[rec_id][1]
            slot_prediction = slots_prediction[rec_id]

            prediction_item = get_prediction_item(idx_to_slot, slot_prediction, tokens)
            result.append((idx_to_intent[pred_intent], prediction_item, text, tokens))

    return result


def get_prediction_item(idx_to_slot, slot_prediction, text):
    prediction_item = []
    # we have added <bos> and <eos> to string. So we remove padding from prediction
    slot_prediction_no_padding = slot_prediction[1:len(text)+1]
    prev_entity = 'FAKE_STRING'
    for token_idx, (text_bit, entity) in enumerate(zip(text, slot_prediction_no_padding)):
        current_entity = idx_to_slot[entity]

        if (prev_entity == current_entity and current_entity.endswith('_INSIDE')) or \
                (prev_entity.replace('_BEGIN', '') == current_entity.replace('_INSIDE', '')):
            prediction_item[len(prediction_item) - 1][0].append(token_idx)

        else:
            prediction_item.append(([token_idx], entity))

        prev_entity = current_entity
    return prediction_item


def compare_metrics(snips_intent_metrics, model_metrics):
    """Compare snips metrics with model metrics

    Parameters
    ----------
    snips_intent_metrics : Tuple(intent_recall, Dict<slot_name, Tuple(precision, recall)>
        SNIPS metrics

    model_metrics : Tuple(intent_recall, Dict<slot_name, Tuple(precision, recall)>
        Model metrics

    Returns
    -------
        Tuple(model, snips, intent_recall_diff), Dict<slot_name, Tuple(model_f1, snips_f1, diff_f1)>

    """
    def get_f1(precision, recall):
        f1_denominator = precision + recall

        if f1_denominator == 0:
            return float('nan')

        return 2 * precision * recall / f1_denominator

    intent_recall_portion = (model_metrics[0],
                             snips_intent_metrics[0],
                             model_metrics[0] - snips_intent_metrics[0])

    slots_f1_portion = {}
    model_slots = model_metrics[1]
    snips_slots = snips_intent_metrics[1]

    for slot_name in snips_slots.keys():
        model_slot = model_slots[slot_name]
        snips_slot = snips_slots[slot_name]
        model_f1 = get_f1(model_slot[0], model_slot[1])
        snips_f1 = get_f1(snips_slot[0], snips_slot[1])
        slots_f1_portion[slot_name] = (model_f1, snips_f1, model_f1 - snips_f1)

    return intent_recall_portion, slots_f1_portion


def run_evaluation(net, intent_map, slots_map, context, batch_size):
    """Compare model performance by each intent and each slot"""
    result = []
    intents = get_all_intents()
    for intent in intents:
        snips_intent_metrics = get_snips_metrics_for_intent(intent)
        ground_truth = get_ground_truth(intent)
        predictions = get_predictions(net, intent, intent_map,
                                      slots_map, context, batch_size)
        model_metrics = get_model_metrics_for_intent(predictions, ground_truth, intent, slots_map)
        metrics = compare_metrics(snips_intent_metrics, model_metrics)
        result.append((intent, metrics))

    return result
