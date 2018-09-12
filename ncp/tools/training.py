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

import collections
import os
import sys

import numpy as np
import scipy.stats
import tensorflow as tf

from ncp.tools import attrdict
from ncp.tools import plotting


def run_experiment(
    logdir, graph, dataset, num_epochs, num_initial, num_select,
    select_after_epochs, eval_after_epochs, log_after_epochs,
    visualize_after_epochs, batch_size, temperature=0.5,
    has_uncertainty=True, drop_remainder=True, evaluate_unseen_train=False,
    filetype='pdf', seed=0):

  if drop_remainder:
    assert num_initial >= batch_size
  logdir = os.path.expanduser(logdir)
  tf.gfile.MakeDirs(logdir)
  random = np.random.RandomState(seed)
  metrics = attrdict.AttrDict(
      num_visible=[],
      train_likelihoods=[],
      train_distances=[],
      test_likelihoods=[],
      test_distances=[])
  visibles = random.choice(len(dataset.train.inputs), num_initial).tolist()

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(num_epochs):
      visible = np.array(visibles)

      # Shuffle, batch, drop remainder.
      indices = random.permutation(np.arange(len(visible)))
      limit = len(visible) - batch_size + 1 if drop_remainder else len(visible)
      for index in range(0, limit, batch_size):
        current = visible[indices[index: index + batch_size]]
        sess.run(graph.optimize, {
            graph.inputs: dataset.train.inputs[current],
            graph.targets: dataset.train.targets[current],
            graph.num_visible: len(visible)})

      if epoch in eval_after_epochs:
        target_scale = dataset.get('target_scale', 1)
        metrics.num_visible.append(len(visibles))
        likelihood, distance = evaluate_model(
            sess, graph, has_uncertainty, dataset.train.inputs[visible],
            dataset.train.targets[visible], target_scale)
        metrics.train_likelihoods.append(likelihood)
        metrics.train_distances.append(distance)
        test_inputs = dataset.test.inputs
        test_targets = dataset.test.targets
        if evaluate_unseen_train:
          unseen = ~np.isin(np.arange(len(dataset.train.inputs)), visible)
          test_inputs = np.concatenate(
              [test_inputs, dataset.train.inputs[unseen]], 0)
          test_targets = np.concatenate(
              [test_targets, dataset.train.targets[unseen]], 0)
        likelihood, distance = evaluate_model(
            sess, graph, has_uncertainty, test_inputs, test_targets,
            target_scale)
        metrics.test_likelihoods.append(likelihood)
        metrics.test_distances.append(distance)

      if epoch in log_after_epochs:
        print(
            'Epoch', epoch,
            'visible', metrics.num_visible[-1],
            'train nlpd {:.2f}'.format(-metrics.train_likelihoods[-1]),
            'train rmse {:.2f}'.format(metrics.train_distances[-1]),
            'test nlpd {:.2f}'.format(-metrics.test_likelihoods[-1]),
            'test rmse {:.2f}'.format(metrics.test_distances[-1]))
        sys.stdout.flush()

      if epoch in visualize_after_epochs:
        filename = os.path.join(logdir, 'epoch-{}.{}'.format(
            epoch, filetype))
        plotting.visualize_model(
            filename, sess, graph, has_uncertainty, dataset, visibles)

      if epoch in select_after_epochs:
        visibles += select_next_target(
            random, sess, graph, has_uncertainty, dataset, visibles,
            num_select, temperature)

  metrics = {key: np.array(value) for key, value in metrics.items()}
  np.savez_compressed(os.path.join(logdir, 'metrics.npz'), **metrics)
  return metrics


def evaluate_model(
    sess, graph, has_uncertainty, inputs, targets, target_scale,
    batch_size=100):
  likelihoods, squared_distances = [], []
  for index in range(0, len(inputs), batch_size):
    target = targets[index: index + batch_size]
    mean, noise, uncertainty = sess.run(
        [graph.data_mean, graph.data_noise, graph.data_uncertainty],
        {graph.inputs: inputs[index: index + batch_size]})
    squared_distances.append((target_scale * (target - mean)) ** 2)
    if has_uncertainty:
      std = np.sqrt(noise ** 2 + uncertainty ** 2 + 1e-8)
    else:
      std = noise
    # Subtracting the log target scale is equivalent to evaluting the
    # log-probability of the unnormalized targets under the scaled predicted
    # mean and standard deviation.
    # likelihood = scipy.stats.norm(
    #     target_scale * mean, target_scale * std).logpdf(
    #         target_scale * target)
    likelihood = scipy.stats.norm(mean, std).logpdf(
        target) - np.log(target_scale)
    likelihoods.append(likelihood)
  likelihood = np.concatenate(likelihoods, 0).sum(1).mean(0)
  distance = np.sqrt(np.concatenate(squared_distances, 0).sum(1).mean(0))
  return likelihood, distance


def select_next_target(
    random, sess, graph, has_uncertainty, dataset, visibles,
    num_select, temperature, batch_size=100):
  values = []
  for index in range(0, len(dataset.train.inputs), batch_size):
    mean, noise, uncertainty = sess.run(
        [graph.data_mean, graph.data_noise, graph.data_uncertainty],
        {graph.inputs: dataset.train.inputs[index: index + batch_size]})
    if has_uncertainty:
      value = 0.5 * np.log(1 + (uncertainty ** 2 / (noise ** 2 + 1e-8)))
    else:
      value = noise
    values.append(value)
  value = np.concatenate(values, 0)
  assert len(value) == len(dataset.train.inputs)
  logit = value.mean(-1) / temperature
  logit -= logit.max()
  logit[np.array(visibles)] = -np.inf
  prob = np.exp(logit)
  prob /= prob.sum()
  return random.choice(len(value), num_select, p=prob).tolist()


def load_results(pattern):
  results = collections.defaultdict(list)
  for filepath in tf.gfile.Glob(pattern):
    metrics = np.load(filepath)
    for key in metrics.keys():
      results[key].append(metrics[key])
  for key, value in results.items():
    results[key] = np.array(value)
  return attrdict.AttrDict(results)
