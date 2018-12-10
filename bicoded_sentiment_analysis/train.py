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
from subprocess import call

import numpy as np
import time

import argparse
import gluonnlp
import mxnet as mx

from gluonnlp.data import FixedBucketSampler
from gluonnlp.data.batchify import Tuple, Stack, Pad
from mxnet import autograd, gluon
from mxnet.gluon.data import DataLoader
from mxnet.gluon.loss import SigmoidBinaryCrossEntropyLoss

from data_transformer import DataTransformer
from dataset import NLPCCDataset
from model import SentimentNet
from result_generator import ResultFileGenerator


def transform_segment(transformer, segment, options):
    dataset = NLPCCDataset(segment, './data/')
    transformed_dataset = dataset.transform(transformer, lazy=False)

    word_vocab = transformer.get_word_vocab()

    batchify_fn = Tuple(Stack(),
                        Pad(axis=0, pad_val=word_vocab[word_vocab.padding_token],
                            ret_length=True),
                        Stack())

    sampler = FixedBucketSampler(lengths=[len(item[1]) for item in transformed_dataset],
                                 batch_size=options.batch_size,
                                 shuffle=True,
                                 num_buckets=options.num_buckets)
    return DataLoader(transformed_dataset, batchify_fn=batchify_fn, batch_sampler=sampler)


def get_model(word_vocab, char_vocab, options):
    model = SentimentNet(word_vocab, sentiments=options.sentiments)
    model.initialize(mx.init.Xavier(), ctx=context)

    en_embedding = gluonnlp.embedding.FastText.from_file('./cc.en.300.aligned.to.zh.vec.npz')
    # en_embedding = gluonnlp.embedding.create('fasttext', source='cc.en.300')
    zh_embedding = gluonnlp.embedding.create('fasttext', source='cc.zh.300')
    en_embedding._unknown_lookup = zh_embedding

    word_vocab.set_embedding(en_embedding)

    model.word_embedding.weight.set_data(word_vocab.embedding.idx_to_vec)
    model.word_embedding.collect_params().setattr('grad_req', 'null')

    # model.hybridize(static_alloc=True)
    model.summary(mx.nd.random.uniform(shape=(args.batch_size, 50), ctx=context))

    return model


def run_training(net, trainer, train_dataloader, val_dataloader, options):
    stop_early = 0
    best_metric = 0
    best_model_name = ''
    loss_fn = SigmoidBinaryCrossEntropyLoss(from_sigmoid=True)

    for epoch in range(args.epochs):
        start_epoch_time = time.time()
        epoch_L = 0.0
        epoch_sent_num = 0
        epoch_wc = 0
        # Log interval training stats
        start_log_interval_time = time.time()
        log_interval_wc = 0
        log_interval_sent_num = 0
        log_interval_L = 0.0

        for i, (rec_id, (data, original_length), label) in enumerate(train_dataloader):
            data = data.as_in_context(context)
            label = label.as_in_context(context).astype(np.float32)
            original_length = original_length.as_in_context(context).astype(np.float32)

            wc = original_length.sum().asscalar()
            log_interval_wc += wc
            epoch_wc += wc
            log_interval_sent_num += data.shape[1]
            epoch_sent_num += data.shape[1]

            with autograd.record():
                output = net(data)
                loss = loss_fn(output, label).mean()
            loss.backward()
            trainer.step(1)

            log_interval_L += loss.asscalar()
            epoch_L += loss.asscalar()

            if (i + 1) % options.log_interval == 0:
                print('[Epoch %d Batch %d/%d] avg loss %g, throughput %gK wps' % (
                    epoch, i + 1, len(train_dataloader),
                    log_interval_L / log_interval_sent_num,
                    log_interval_wc / 1000 / (time.time() - start_log_interval_time)))
                # Clear log interval training stats
                start_log_interval_time = time.time()
                log_interval_wc = 0
                log_interval_sent_num = 0
                log_interval_L = 0

        end_epoch_time = time.time()
        _, train_acc, train_em, train_f1, _ = run_evaluate(net, train_dataloader, options)
        valid_avg_L, valid_acc, valid_em, valid_f1, _ = run_evaluate(net, val_dataloader, options)

        print('[Epoch %d] '
              'train acc %.4f, train EM %.4f, train F1 %.4f, train avg loss %g, '
              'valid acc %.4f, valid EM %.4f, valid F1 %.4f, valid avg loss %g, '
              'throughput %gK wps' % (
                  epoch,
                  train_acc, train_em, train_f1, epoch_L / epoch_sent_num,
                  valid_acc, valid_em, valid_f1, valid_avg_L,
                  epoch_wc / 1000 / (end_epoch_time - start_epoch_time)))

        if valid_f1 < best_metric:
            print('No Improvement.')
            stop_early += 1
            if options.early_stop and stop_early == 5:
                print('No improvement for 5 times. Stop training. '
                      'Best valid F1 found: %.4f' % best_metric)
                break
        else:
            # Reset stop_early if the validation loss finds a new low value
            print('Observed Improvement.')
            stop_early = 0
            best_model_name = options.save_prefix + '_{:04d}.params'.format(epoch)
            net.save_parameters(best_model_name)
            best_metric = valid_f1

    print('Stop training. Best valid F1: %.4f, best model: %s' % (best_metric, best_model_name))
    return best_model_name


def run_evaluate(net, dataloader, options, return_predictions=False):
    """Evaluate network on the specified dataset"""
    total_L = 0.0
    total_sample_num = 0
    total_correct_classes = 0
    exact_match = 0
    prediction_results = []
    f1s = [mx.metric.F1(average='micro') for i in range(options.sentiments)]

    start_log_interval_time = time.time()

    loss_fn = SigmoidBinaryCrossEntropyLoss(from_sigmoid=True)

    print('Begin Testing...')

    for i, (rec_id, (data, original_length), label) in enumerate(dataloader):
        data = data.as_in_context(context)
        original_length = original_length.as_in_context(context).astype(np.float32)
        label = label.as_in_context(context).astype(np.float32)

        output = net(data)
        L = loss_fn(output, label)

        total_L += L.sum().asscalar()
        total_sample_num += label.shape[0]
        total_class_num = label.shape[1]

        pred = output > options.threshold
        total_correct_classes += (pred == label).sum().asscalar()
        exact_match += int(((pred == label).sum(axis=1) == total_class_num).sum().asscalar())

        for j, f1 in enumerate(f1s):
            emotion_pred = pred[:, j].reshape(0, 1)
            emotion_pred_neg = (1 - pred[:, j]).reshape(0, 1)
            pred_for_emotion = [mx.nd.concat(*[emotion_pred_neg, emotion_pred], dim=1)]
            label_for_emotion = [label[:, j]]
            f1.update(label_for_emotion, pred_for_emotion)

        if return_predictions:
            for ri, pr in zip(rec_id, pred):
                item = {'ri': ri.asscalar(),
                        'happiness': pr[0].asscalar(),
                        'sadness': pr[1].asscalar(),
                        'anger': pr[2].asscalar(),
                        'fear': pr[3].asscalar(),
                        'surprise': pr[4].asscalar()}
                prediction_results.append(item)

        if (i + 1) % args.log_interval == 0:
            print('[Batch {}/{}] elapsed {:.2f} s'.format(
                i + 1, len(dataloader), time.time() - start_log_interval_time))
            start_log_interval_time = time.time()

    avg_L = total_L / float(total_sample_num)
    # we need to divide by number of classes,
    acc = total_correct_classes / float(total_sample_num) / float(total_class_num)
    em = exact_match / float(total_sample_num)
    f1_avg = mx.nd.array([f1.get()[1] for f1 in f1s]).mean().asscalar()
    return avg_L, acc, em, f1_avg, prediction_results


def parse_args():
    parser = argparse.ArgumentParser(description='Sentiment analysis with the textCNN model on\
                                     various datasets.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='initial learning rate')
    parser.add_argument('--early_stop', type=int, default=0,
                        help='Do early stop')
    parser.add_argument('--epochs', type=int, default=40,
                        help='upper epoch limit')
    parser.add_argument('--sentiments', type=int, default=5,
                        help='Total number of sentiments')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold to use between positive/negative class difference')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='batch size')
    parser.add_argument('--num_buckets', type=int, default=30, metavar='N',
                        help='Number of buckets to reduce padding')
    parser.add_argument('--dropout', type=float, default=.5,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--log-interval', type=int, default=30, metavar='N',
                        help='report interval')
    parser.add_argument('--save-prefix', type=str, default='sa-model',
                        help='path to save the final model')
    parser.add_argument('--gpu', type=int, default=0,
                        help='id of the gpu to use. Set it to empty means to use cpu.')
    arg = parser.parse_args()
    return arg


if __name__ == '__main__':
    args = parse_args()
    context = mx.cpu(0) if args.gpu is None else mx.gpu(args.gpu)

    segments = ['train', 'dev']
    transformer = DataTransformer(segments)
    dataloaders = [transform_segment(transformer, segment, args) for segment in segments]

    model = get_model(transformer._word_vocab, transformer._char_vocab, args)

    trainer = gluon.Trainer(model.collect_params(), 'ftml', {'learning_rate': args.lr})
    best_model_name = run_training(model, trainer, dataloaders[0], dataloaders[1], args)

    model.load_parameters(best_model_name, ctx=context)
    avg_L, acc, em, f1, predictions = run_evaluate(model, dataloaders[1],
                                                   args, return_predictions=True)
    result_generator = ResultFileGenerator()
    result_generator.write_results(predictions, './dev-predictions.txt')
    call(['python', './evaluate_tool/evaluate.py', './data/dev.txt',  './dev-predictions.txt'])

    test_dataloader = transform_segment(transformer, 'test', args)
    _, _, _, test_f1, predictions = run_evaluate(model, test_dataloader,
                                                 args, return_predictions=True)
    result_generator.write_results(predictions, './test-predictions.txt')
    call(['python', './evaluate_tool/evaluate.py', './data/test.txt',  './test-predictions.txt'])
    print('Macro F1 for test dataset is: {}'.format(test_f1))
