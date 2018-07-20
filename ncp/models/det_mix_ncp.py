# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf

from ncp import tools

tfd = tf.contrib.distributions


def network(inputs, config):
  hidden = inputs
  for size in config.layer_sizes:
    hidden = tf.layers.dense(hidden, size, tf.nn.leaky_relu)
  mean = tf.layers.dense(hidden, 1)
  noise = tf.layers.dense(hidden, 1, tf.nn.softplus) + 1e-6
  uncertainty = tf.layers.dense(hidden, 1, None)
  return mean, noise, uncertainty


def define_graph(config):
  network_tpl = tf.make_template('network', network, config=config)
  inputs = tf.placeholder(tf.float32, [None, config.num_inputs])
  targets = tf.placeholder(tf.float32, [None, 1])
  num_visible = tf.placeholder(tf.int32, [])
  batch_size = tf.to_float(tf.shape(inputs)[0])
  data_mean, data_noise, data_uncertainty = network_tpl(inputs)
  back_inputs = inputs + tf.random_normal(
      tf.shape(inputs), 0.0, config.noise_std)
  back_mean, back_noise, back_uncertainty = network_tpl(back_inputs)
  losses = [
      -tfd.Normal(data_mean, data_noise).log_prob(targets),
      -tfd.Bernoulli(data_uncertainty).log_prob(0),
      -tfd.Bernoulli(back_uncertainty).log_prob(1),
  ]
  loss = sum(tf.reduce_sum(loss) for loss in losses) / batch_size
  optimizer = tf.train.AdamOptimizer(config.learning_rate)
  gradients, variables = zip(*optimizer.compute_gradients(
      loss, colocate_gradients_with_ops=True))
  if config.clip_gradient:
    gradients, _ = tf.clip_by_global_norm(gradients, config.clip_gradient)
  optimize = optimizer.apply_gradients(zip(gradients, variables))
  data_uncertainty = tf.sigmoid(data_uncertainty)
  data_mean = (1 - data_uncertainty) * data_mean + data_uncertainty * 0
  data_noise = (1 - data_uncertainty) * data_noise + data_uncertainty * 0.1
  return tools.AttrDict(locals())
