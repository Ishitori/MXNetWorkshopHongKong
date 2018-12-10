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
"""Main file to run training of keyphrase extraction example."""
import argparse
import multiprocessing
import mxnet as mx
from mxnet import autograd
from mxnet.gluon import Trainer
from mxnet.gluon.data import DataLoader
from mxnet.gluon.nn import HybridSequential, Embedding, Dropout, Dense
from mxnet.gluon.rnn import LSTM
from mxnet.metric import Accuracy

from gluonnlp import data, Vocab, embedding
from dataset import INSPECDataset
from data_transformer import DataTransformer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=None, help='Index of GPU to use')
    parser.add_argument('--embedding_dim', type=int, default=100, help='glove embedding dim')
    parser.add_argument('--logging_path', default="./log_glove_100.txt",
                        help='logging file path')
    parser.add_argument('--model_path', default='./model_glove_100.params',
                        help='saving model in model_path')
    parser.add_argument('--hidden', type=int, default=300, help='hidden units in bilstm')
    parser.add_argument('--lstm_dropout', type=float, default=0.5,
                        help='dropout applied to lstm layers ')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=14, help='training epochs')
    parser.add_argument('--seq_len', type=int, default=500, help='max length of sequences')
    parser.add_argument('--batch_size', type=int, default=100, help='batch size')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='dropout applied to fully connected layers')
    return parser.parse_args()


def get_vocab(datasets):
    all_words = [word for dataset in datasets for item in dataset for word in item[0]]
    vocab = Vocab(data.count_tokens(all_words))
    glove = embedding.create('glove', source='glove.6B.' + str(args.embedding_dim) + 'd')
    vocab.set_embedding(glove)
    return vocab


def get_model(vocab_size, embedding_size, hidden_size, dropout_rate, classes=3):
    net = HybridSequential()

    with net.name_scope():
        net.add(Embedding(vocab_size, embedding_size))
        net.add(Dropout(args.dropout))
        net.add(LSTM(hidden_size=hidden_size // 2,
                     num_layers=1,
                     layout='NTC',
                     bidirectional=True,
                     dropout=dropout_rate))
        net.add(Dense(units=classes, flatten=False))

    return net


def run_training(net, trainer, train_dataloader, val_dataloader, epochs, model_path, context):
    loss_fn = mx.gluon.loss.SoftmaxCrossEntropyLoss()

    for e in range(epochs):
        train_acc = Accuracy()
        val_acc = Accuracy()
        train_loss = 0.
        total_items = 0

        for i, (data, label) in enumerate(train_dataloader):
            items_per_iteration = data.shape[0]
            total_items += items_per_iteration

            data = data.as_in_context(context)
            label = label.as_in_context(context)

            with autograd.record():
                output = net(data)
                output = output.reshape((-1, 3))
                label = label.reshape((-1, 1))
                loss = loss_fn(output, label)

            loss.backward()
            trainer.step(items_per_iteration)

            train_loss += loss.mean().asscalar()
            train_acc.update(label.flatten(), output.argmax(axis=1).flatten())

        for i, (data, label) in enumerate(val_dataloader):
            data = data.as_in_context(context)
            label = label.as_in_context(context)

            output = net(data)
            output = output.reshape((-1, 3))
            val_acc.update(label.reshape(-1, 1).flatten(), output.argmax(axis=1).flatten())

        print("Epoch {}. Current Loss: {:.5f}. Train accuracy: {:.3f}, Validation accuracy: {:.3f}."
              .format(e, train_loss / total_items, train_acc.get()[1], val_acc.get()[1]))

    net.save_parameters(model_path)
    return model_path


def run_evaluation(net, test_dataloader, context):
    correct = 0
    extract = 0
    standard = 0

    for i, (data, label) in enumerate(test_dataloader):
        data = data.as_in_context(context)
        label = label.as_in_context(context)

        output = net(data).reshape((-1, 3))

        pred = mx.nd.argmax(output, axis=1).asnumpy().flatten()
        label = label.asnumpy().flatten()

        pred2 = [str(int(x)) for x in pred]
        label2 = [str(x) for x in label]

        predstr = ''.join(pred2).replace('0', ' ').split()
        labelstr = ''.join(label2).replace('0', ' ').split()

        extract += len(predstr)
        standard += len(labelstr)

        i = 0
        while i < len(label):
            if label[i] != 0:
                while i < len(label) and label[i] != 0 and pred[i] == label[i]:
                    i += 1

                if i < len(label) and label[i] == pred[i] == 0 or i == len(label):
                    correct += 1
            i += 1

    precision = 1.0 * correct / extract
    recall = 1.0 * correct / standard
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1, correct, extract, standard


if __name__ == '__main__':
    args = parse_args()
    context = mx.cpu(0) if args.gpu is None else mx.gpu(args.gpu)

    train_dataset = INSPECDataset('train')
    dev_dataset = INSPECDataset('dev')
    test_dataset = INSPECDataset('test')

    vocab = get_vocab([train_dataset, dev_dataset])
    transformer = DataTransformer(vocab, args.seq_len)

    train_dataloader = DataLoader(train_dataset.transform(transformer), batch_size=args.batch_size,
                                  shuffle=True, num_workers=multiprocessing.cpu_count() - 3)
    dev_dataloader = DataLoader(dev_dataset.transform(transformer), batch_size=args.batch_size,
                                shuffle=True, num_workers=multiprocessing.cpu_count() - 3)
    test_dataloader = DataLoader(test_dataset.transform(transformer), batch_size=args.batch_size,
                                 shuffle=True, num_workers=multiprocessing.cpu_count() - 3)

    model = get_model(len(vocab), args.embedding_dim, args.hidden, args.lstm_dropout)
    model.initialize(mx.init.Normal(sigma=0.1), ctx=context)
    model[0].weight.set_data(vocab.embedding.idx_to_vec)

    trainer = Trainer(model.collect_params(), 'adam', {'learning_rate': args.learning_rate})
    best_model_path = run_training(model, trainer, train_dataloader, dev_dataloader,
                                   args.epochs, args.model_path, context)

    model_for_eval = get_model(len(vocab), args.embedding_dim, args.hidden, args.lstm_dropout)
    model_for_eval.load_parameters(best_model_path, ctx=context)

    precision, recall, f1, _, _, _ = run_evaluation(model_for_eval, test_dataloader, context)
    print("Test done. Precision: {:.3f}, Recall: {:.3f}, F1: {:.3f}".format(precision, recall, f1))
