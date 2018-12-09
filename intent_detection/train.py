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
"""Main file to run training of intent detection and slot filling example."""
import warnings
import argparse
import multiprocessing

import mxnet as mx
from gluonnlp.data.batchify import Tuple, Pad, Stack
from gluonnlp.model import get_model
from gluonnlp.vocab import ELMoCharVocab
from mxnet import autograd
from mxnet.gluon import Trainer
from mxnet.gluon.data import DataLoader
from mxnet.metric import Accuracy

from crf import CRF
from compare_performance import run_evaluation
from dataset import NLUBenchmarkDataset
from data_transformer import DataTransformer
from model import OneNet
from tokenizer import SacreMosesTokenizer
from utils import get_data_mask
warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=None, help='Index of GPU to use')
    parser.add_argument('--logging_path', default="./log.txt",
                        help='logging file path')
    parser.add_argument('--model_path', default='./model',
                        help='saving model in model_path')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=20, help='training epochs')
    parser.add_argument('--batch_size', type=int, default=100, help='batch size')
    return parser.parse_args()


def print_eval_results(eval_result):
    print('                                |  Model  |  SNIPS  |  Diff   |')
    print('--------------------------------+---------+---------+---------+')

    for intent in eval_result:
        intent_name = intent[0]
        intent_metrics = intent[1][0]
        print('{:<32}|  {:+.3f} |  {:+.3f} |  {:+5.3f} |'.format(intent_name,
                                                                 intent_metrics[0],
                                                                 intent_metrics[1],
                                                                 intent_metrics[2]))

        for slot_name, slot_metrics in intent[1][1].items():
            print('    {:<28}|  {:+.3f} |  {:+.3f} |  {:+5.3f} |'.format(slot_name,
                                                                         slot_metrics[0],
                                                                         slot_metrics[1],
                                                                         slot_metrics[2]))


def run_training(net, trainer, train_dataloader, val_dataloader, intents_count,
                 epochs, model_path, context):
    intent_loss_fn = mx.gluon.loss.SoftmaxCrossEntropyLoss()
    max_val_accuracy = 0
    best_model_path = ''

    for e in range(epochs):
        intent_train_acc = Accuracy()
        slot_train_acc = Accuracy()

        intent_val_acc = Accuracy()
        slot_val_acc = Accuracy()

        train_loss = 0.
        total_items = 0

        for i, (data, valid_lengths, entities, intent) in enumerate(train_dataloader):
            length = data.shape[1]
            items_per_iteration = data.shape[0]
            total_items += items_per_iteration

            data = data.as_in_context(context)
            intent = intent.as_in_context(context)
            entities = entities.as_in_context(context)

            hidden_state = net.elmo_container[0].begin_state(mx.nd.zeros,
                                                             batch_size=items_per_iteration,
                                                             ctx=context)
            mask = get_data_mask(length, valid_lengths, items_per_iteration, context)

            with autograd.record():
                intents, slots = net(data, hidden_state, mask)
                intents = intents.reshape((-1, intents_count))
                intent = intent.reshape((-1, 1))
                loss_intent = intent_loss_fn(intents, intent)

                # crf accepts seq_len x bs x channels
                score, slots_seq = net.crf(slots.transpose(axes=(1, 0, 2)))
                neg_log_likelihood = net.crf.neg_log_likelihood(slots.transpose(axes=(1, 0, 2)),
                                                                entities)
                loss = 0.1 * loss_intent.mean() + 0.9 * neg_log_likelihood.mean()

            loss.backward()
            trainer.step(1)

            train_loss += loss.mean().asscalar()
            intent_train_acc.update(intent.flatten(), intents.argmax(axis=1).flatten())
            slot_train_acc.update(entities, slots_seq)

        for i, (data, valid_lengths, entities, intent) in enumerate(val_dataloader):
            items_per_iteration = data.shape[0]
            length = data.shape[1]

            data = data.as_in_context(context)
            intent = intent.as_in_context(context)
            entities = entities.as_in_context(context)

            hidden_state = net.elmo_container[0].begin_state(mx.nd.zeros,
                                                             batch_size=items_per_iteration,
                                                             ctx=context)
            mask = get_data_mask(length, valid_lengths, items_per_iteration, context)

            intents, slots = net(data, hidden_state, mask)
            intents = intents.reshape((-1, intents_count))
            intent = intent.reshape((-1, 1))

            score, slots_seq = net.crf(slots.transpose(axes=(1, 0, 2)))

            intent_val_acc.update(intent.flatten(), intents.argmax(axis=1).flatten())
            slot_val_acc.update(entities, slots_seq)

        print("Epoch {}. Current Loss: {:.5f}. \n"
              "Intent train accuracy: {:.3f}, Slots train accuracy: {:.3f}, \n"
              "Intent valid accuracy: {:.3f}, Slot val accuracy: {:.3f}"
              .format(e, train_loss / total_items,
                      intent_train_acc.get()[1], slot_train_acc.get()[1],
                      intent_val_acc.get()[1], slot_val_acc.get()[1]))

        if max_val_accuracy < slot_val_acc.get()[1]:
            max_val_accuracy = slot_val_acc.get()[1]
            best_model_path = model_path + '_{:04d}.params'.format(e)
            net.save_parameters(best_model_path)
            print("Improvement observed")
        else:
            print("No improvement")

    return best_model_path


if __name__ == '__main__':
    args = parse_args()
    context = mx.cpu(0) if args.gpu is None else mx.gpu(args.gpu)

    train_dataset = NLUBenchmarkDataset(SacreMosesTokenizer(), 'train_full')
    print(train_dataset.get_intent_map())
    print(train_dataset.get_slots_map())
    dev_dataset = NLUBenchmarkDataset(SacreMosesTokenizer(), 'val',
                                      train_dataset.get_intent_map(), train_dataset.get_slots_map())

    transformer = DataTransformer(ELMoCharVocab())
    transformed_train_dataset = train_dataset.transform(transformer, lazy=False)
    transformed_dev_dataset = dev_dataset.transform(transformer, lazy=False)

    batchify_fn = Tuple(Pad(), Stack(), Pad(), Stack())

    train_dataloader = DataLoader(transformed_train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=multiprocessing.cpu_count() - 3,
                                  batchify_fn=batchify_fn)
    dev_dataloader = DataLoader(transformed_dev_dataset, batch_size=args.batch_size,
                                shuffle=True, num_workers=multiprocessing.cpu_count() - 3,
                                batchify_fn=batchify_fn)

    slots_count = len(train_dataset.get_slots_map())
    intents_count = len(train_dataset.get_intent_map())

    assert len(set(train_dataset.get_slots_map()) ^ set(dev_dataset.get_slots_map())) == 0

    crf = CRF(train_dataset.get_slots_map(), ctx=context)
    elmo, _ = get_model('elmo_2x4096_512_2048cnn_2xhighway',
                        dataset_name='gbw',
                        pretrained=True,
                        ctx=context)
    model = OneNet(elmo, crf, intents_count, slots_count)
    model.initialize(mx.init.Xavier(magnitude=2.24), ctx=context)

    trainer = Trainer(model.collect_params(), 'ftml', {'learning_rate': args.learning_rate})
    best_model_path = run_training(model, trainer, train_dataloader, dev_dataloader,
                                   intents_count, args.epochs, args.model_path, context)

    print('Model to use: {}'.format(best_model_path))
    model.load_parameters(best_model_path, ctx=context)
    eval_result = run_evaluation(model, train_dataset.get_intent_map(),
                                 train_dataset.get_slots_map(), context, args.batch_size)

    print_eval_results(eval_result)
