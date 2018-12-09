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
import mxnet as mx


def get_data_mask(length, valid_lengths, items_per_iteration, context):
    mask = mx.nd.arange(length).expand_dims(0).broadcast_axes(axis=(0,),
                                                              size=(items_per_iteration,))
    mask = mask < valid_lengths.expand_dims(1).astype('float32')
    mask = mask.as_in_context(context)
    return mask
