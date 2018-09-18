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

import functools

from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


palette = [
    '#1f78b4',
    '#33a02c',
    '#e31a1c',
    '#ff7f00',
    '#6a3d9a',
    '#b15928',
    '#a6cee3',
    '#b2df8a',
    '#fb9a99',
    '#fdbf6f',
    '#cab2d6',
    '#ffff99',
]


def visualize_model(
    filename, sess, graph, has_uncertainty, dataset, visibles,
    batch_size=100):
  means, noises, uncertainties = [], [], []
  for index in range(0, len(dataset.domain), batch_size):
    mean, noise, uncertainty = sess.run(
        [graph.data_mean, graph.data_noise, graph.data_uncertainty],
        {graph.inputs: dataset.domain[index: index + batch_size]})
    means.append(mean)
    noises.append(noise)
    uncertainties.append(uncertainty)
  mean = np.concatenate(means, 0)
  noise = np.concatenate(noises, 0)
  if has_uncertainty:
    uncertainty = np.concatenate(uncertainties, 0)
  visible = np.array(visibles)
  not_visible = np.ones(len(dataset.train.targets), dtype=bool)
  not_visible[visibles] = False
  fig, axes = plt.subplots(
      ncols=dataset.domain.shape[1], squeeze=False,
      figsize=(3 * dataset.domain.shape[1], 4))
  axes = axes[0]  # Only one row.
  for index, ax in enumerate(axes):
    # Predictive distribution.
    if dataset.domain.shape[1] > 1:
      ax.set_title('Input variable {}'.format(index + 1))
    ax.set_ylabel('dist')
    std = np.sqrt(noise ** 2 + uncertainty ** 2) if has_uncertainty else noise
    ax.scatter(
        dataset.test.inputs[:, index],
        dataset.test.targets[:, 0], c='#dddddd', lw=0, s=3)
    ax.scatter(
        dataset.train.inputs[not_visible, index],
        dataset.train.targets[not_visible, 0], c='#dddddd', lw=0, s=3)
    plot_prediction(ax, dataset.domain[:, index], mean[:, 0], std[:, 0])
    ax.scatter(
        dataset.train.inputs[visible, index],
        dataset.train.targets[visible, 0], c='#000000', lw=0, s=4)
    ax.set_xlim(dataset.domain[0, index], dataset.domain[-1, index])
    min_ = min(dataset.train.targets.min(), dataset.test.targets.min())
    max_ = max(dataset.train.targets.max(), dataset.test.targets.max())
    padding = 0.2 * (max_ - min_) + 1e-6
    ax.set_ylim(min_ - padding, max_ + padding)
    ax.xaxis.set_ticks([])
    # Uncertainty.
    divider = make_axes_locatable(ax)
    if has_uncertainty:
      ax = divider.append_axes('bottom', size='50%', pad=0)
      ax.set_ylabel('uncert', labelpad=1)
      ax.xaxis.set_ticks([])
      plot_std_area(ax, dataset.domain[:, index], uncertainty[:, 0])
    # Predicted noise.
    ax = divider.append_axes('bottom', size='50%', pad=0)
    ax.set_ylabel('noise', labelpad=5)
    plot_std_area(ax, dataset.domain[:, index], noise[:, 0], color='red')
    # Expected information gain.
    if has_uncertainty:
      infogain = 0.5 * np.log(1 + (uncertainty ** 2 / (noise ** 2 + 1e-8)))
      ax = divider.append_axes('bottom', size='50%', pad=0)
      ax.set_ylabel('infogain', labelpad=10)
      plot_std_area(ax, dataset.domain[:, index], infogain[:, 0], color='gray')
  fig.tight_layout(pad=0, h_pad=0, w_pad=1)
  fig.savefig(filename)
  plt.close(fig)


def plot_prediction(ax, inputs, mean, std, area=2):
  order = np.argsort(inputs)
  inputs, mean, std = inputs[order], mean[order], std[order]
  ax.plot(inputs, mean, alpha=0.5)
  ax.fill_between(
      inputs, mean - area * std, mean + area * std, alpha=0.1, lw=0)
  ax.yaxis.set_major_locator(plt.MaxNLocator(5, prune='both'))
  ax.set_xlim(inputs.min(), inputs.max())
  min_, max_ = inputs.min(), inputs.max()
  min_ -= 0.1 * (max_ - min_ + 1e-6)
  max_ += 0.1 * (max_ - min_ + 1e-6)
  ax.set_xlim(min_, max_)
  ax.yaxis.tick_right()
  ax.yaxis.set_label_coords(-0.05, 0.5)


def plot_std_area(ax, inputs, std, **kwargs):
  kwargs['alpha'] = kwargs.get('alpha', 0.5)
  kwargs['lw'] = kwargs.get('lw', 0.0)
  order = np.argsort(inputs)
  inputs, std = inputs[order], std[order]
  ax.fill_between(inputs, std, 0 * std, **kwargs)
  ax.set_xlim(inputs.min(), inputs.max())
  ax.set_ylim(0, std.max())
  ax.yaxis.set_major_locator(plt.MaxNLocator(4, prune='both'))
  ax.yaxis.tick_right()
  ax.yaxis.set_label_coords(-0.05, 0.5)


def plot_likelihood(ax, results, key, baselines):
  for index, (model, metrics) in enumerate(results):
    if not metrics.keys():
      continue
    def formatter(metrics, x, _):
      index = min(int(x), len(metrics.num_visible[0]) - 1)
      return metrics.num_visible[0][index]
    ax.xaxis.set_major_formatter(
        mpl.ticker.FuncFormatter(functools.partial(formatter, metrics)))
    domain = np.arange(metrics[key].shape[1])
    values = -metrics[key]
    ax.plot(domain, values.mean(0), c=palette[index], label=model)
    ax.fill_between(
        domain, values.mean(0) - values.std(0), values.mean(0) + values.std(0),
        color=palette[index], alpha=0.2)
    # ax.set_xlim(domain[0], domain[-1])
    ax.set_xlim(domain[0], domain[-20])
  x_scale = ax.get_xlim()[1] - ax.get_xlim()[0]
  y_scale = ax.get_ylim()[1] - ax.get_ylim()[0]
  for name, value in baselines.items():
    right = ax.get_xlim()[1] - 0.05 * x_scale
    bottom = value + 0.03 * y_scale
    if name.startswith('MF'):
      bottom -= 0.09 * y_scale
    ax.axhline(value, c='gray', ls='--')
    ax.text(right, bottom, name, color='gray', horizontalalignment='right')


def plot_distance(ax, results, key, baselines):
  for index, (model, metrics) in enumerate(results):
    if not metrics.keys():
      continue
    def formatter(metrics, x, _):
      index = min(int(x), len(metrics.num_visible[0]) - 1)
      return metrics.num_visible[0][index]
    ax.xaxis.set_major_formatter(
        mpl.ticker.FuncFormatter(functools.partial(formatter, metrics)))
    domain = np.arange(metrics[key].shape[1])
    values = metrics[key]
    ax.plot(domain, values.mean(0), c=palette[index], label=model)
    ax.fill_between(
        domain, values.mean(0) - values.std(0), values.mean(0) + values.std(0),
        color=palette[index], alpha=0.2)
    # ax.set_xlim(domain[0], domain[-1])
    ax.set_xlim(domain[0], domain[-20])
  x_scale = ax.get_xlim()[1] - ax.get_xlim()[0]
  y_scale = ax.get_ylim()[1] - ax.get_ylim()[0]
  for name, value in baselines.items():
    right = ax.get_xlim()[1] - 0.05 * x_scale
    bottom = value + 0.03 * y_scale
    if name.startswith('MF'):
      bottom -= 0.09 * y_scale
    ax.axhline(value, c='gray', ls='--')
    ax.text(right, bottom, name, color='gray', horizontalalignment='right')
