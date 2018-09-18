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

from tensorflow_probability import distributions as tfd
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from ncp import tools


def network(inputs, config):
  init_std = np.log(np.exp(config.weight_std) - 1).astype(np.float32)
  hidden = inputs
  for size in config.layer_sizes:
    hidden = tf.layers.dense(hidden, size, tf.nn.leaky_relu)
  kernel_posterior = tfd.Independent(tfd.Normal(
      tf.get_variable(
          'kernel_mean', (hidden.shape[-1].value, 1), tf.float32,
          tf.random_normal_initializer(0, config.weight_std)),
      tf.nn.softplus(tf.get_variable(
          'kernel_std', (hidden.shape[-1].value, 1), tf.float32,
          tf.constant_initializer(init_std)))), 2)
  kernel_prior = tfd.Independent(tfd.Normal(
      tf.zeros_like(kernel_posterior.mean()),
      tf.zeros_like(kernel_posterior.mean()) + tf.nn.softplus(init_std)), 2)
  bias_prior = None
  bias_posterior = tfd.Deterministic(tf.get_variable(
      'bias_mean', (1,), tf.float32, tf.constant_initializer(0.0)))
  tf.add_to_collection(
      tf.GraphKeys.REGULARIZATION_LOSSES,
      tfd.kl_divergence(kernel_posterior, kernel_prior))
  mean = tfp.layers.DenseReparameterization(
      1,
      kernel_prior_fn=lambda *args, **kwargs: kernel_prior,
      kernel_posterior_fn=lambda *args, **kwargs: kernel_posterior,
      bias_prior_fn=lambda *args, **kwargs: bias_prior,
      bias_posterior_fn=lambda *args, **kwargs: bias_posterior)(hidden)
  mean_dist = tfd.Normal(
      tf.matmul(hidden, kernel_posterior.mean()) + bias_posterior.mean(),
      tf.sqrt(tf.matmul(hidden ** 2, kernel_posterior.variance())))
  std = tf.layers.dense(hidden, 1, tf.nn.softplus) + 1e-6
  data_dist = tfd.Normal(mean, std)
  return data_dist, mean_dist


def define_graph(config):
  network_tpl = tf.make_template('network', network, config=config)
  inputs = tf.placeholder(tf.float32, [None, config.num_inputs])
  targets = tf.placeholder(tf.float32, [None, 1])
  num_visible = tf.placeholder(tf.int32, [])
  batch_size = tf.shape(inputs)[0]
  data_dist, mean_dist = network_tpl(inputs)
  ood_inputs = inputs + tf.random_normal(
      tf.shape(inputs), 0.0, config.noise_std)
  ood_data_dist, ood_mean_dist = network_tpl(ood_inputs)
  assert len(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
  divergence = sum([
      tf.reduce_sum(tensor) for tensor in
      tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)])
  num_batches = tf.to_float(num_visible) / tf.to_float(batch_size)
  if config.center_at_target:
    ood_mean_prior = tfd.Normal(targets, 1.0)
  else:
    ood_mean_prior = tfd.Normal(0.0, 1.0)
  losses = [
      config.divergence_scale * divergence / num_batches,
      -data_dist.log_prob(targets),
      config.ncp_scale * tfd.kl_divergence(ood_mean_prior, ood_mean_dist),
  ]
  if config.ood_std_prior:
    sg = tf.stop_gradient
    ood_std_dist = tfd.Normal(sg(ood_mean_dist.mean()), ood_data_dist.stddev())
    ood_std_prior = tfd.Normal(sg(ood_mean_dist.mean()), config.ood_std_prior)
    divergence = tfd.kl_divergence(ood_std_prior, ood_std_dist)
    losses.append(config.ncp_scale * divergence)
  loss = sum(tf.reduce_sum(loss) for loss in losses) / tf.to_float(batch_size)
  optimizer = tf.train.AdamOptimizer(config.learning_rate)
  gradients, variables = zip(*optimizer.compute_gradients(
      loss, colocate_gradients_with_ops=True))
  if config.clip_gradient:
    gradients, _ = tf.clip_by_global_norm(gradients, config.clip_gradient)
  optimize = optimizer.apply_gradients(zip(gradients, variables))
  data_mean = mean_dist.mean()
  data_noise = data_dist.stddev()
  data_uncertainty = mean_dist.stddev()
  return tools.AttrDict(locals())
