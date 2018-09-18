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

import argparse
import itertools
import os
import warnings

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import ruamel.yaml as yaml
import tensorflow as tf

from ncp import datasets
from ncp import models
from ncp import tools


def default_schedule(model):
  config = tools.AttrDict()
  config.num_epochs = 2000
  config.num_initial = 10
  config.num_select = 10
  config.select_after_epochs = range(50, 2000, 50)
  config.eval_after_epochs = range(0, 2000, 50)
  config.log_after_epochs = range(0, 2000, 500)
  config.visualize_after_epochs = range(50, 2000, 500)
  config.batch_size = 10
  config.temperature = 0.5
  config.filetype = 'png'
  if model == 'det':
    config.has_uncertainty = False
  return config


def default_config(model):
  config = tools.AttrDict()
  config.num_inputs = 8
  config.layer_sizes = [50, 50]
  if model == 'bbb':
    config.divergence_scale = 1.0
  if model == 'bbb_ncp':
    config.noise_std = 0.1
    config.ncp_scale = 0.1
    config.divergence_scale = 0
    config.ood_std_prior = None
    config.center_at_target = True
  if model == 'det_mix_ncp':
    config.noise_std = 0.1
    config.center_at_target = True
  config.learning_rate = 1e-4
  config.weight_std = 0.1
  config.clip_gradient = 100.0
  return config


def plot_results(args):
  load_results = lambda x: tools.load_results(
      os.path.join(args.logdir, x) + '-*/*.npz')
  results = [
      ('BBB+NCP', load_results('bbb_ncp')),
      ('ODC+NCP', load_results('det_mix_ncp')),
      ('BBB', load_results('bbb')),
      ('Det', load_results('det')),
  ]
  fig, ax = plt.subplots(ncols=4, figsize=(8, 2))
  tools.plot_distance(ax[0], results, 'train_distances', {})
  ax[0].set_xlabel('Data points seen')
  ax[0].set_title('Train RMSE')
  ax[0].set_ylim(15, 40)
  tools.plot_likelihood(ax[1], results, 'train_likelihoods', {})
  ax[1].set_xlabel('Data points seen')
  ax[1].set_title('Train NLPD')
  ax[1].set_ylim(3.2, 5.0)
  tools.plot_distance(ax[2], results, 'test_distances', {})
  ax[2].set_xlabel('Data points seen')
  ax[2].set_title('Test RMSE')
  ax[2].set_ylim(29.5, 32)
  tools.plot_likelihood(ax[3], results, 'test_likelihoods', {})
  ax[3].set_xlabel('Data points seen')
  ax[3].set_title('Test NLPD')
  ax[3].set_ylim(4.6, 5.6)
  ax[3].legend(frameon=False, labelspacing=0.2, borderpad=0)
  fig.tight_layout(pad=0, w_pad=0.5)
  filename = os.path.join(args.logdir, 'results.pdf')
  fig.savefig(filename)


def main(args):
  if args.replot:
    plot_results(args)
    return
  warnings.filterwarnings('ignore', category=DeprecationWarning)  # TensorFlow.
  dataset = datasets.load_numpy_dataset(
      args.dataset, args.train_amount, args.test_amount)
  models_ = [
      ('bbb_ncp', models.bbb_ncp.define_graph),
      ('det_mix_ncp', models.det_mix_ncp.define_graph),
      ('bbb', models.bbb.define_graph),
      ('det', models.det.define_graph),
  ]
  experiments = itertools.product(range(args.seeds), models_)
  for seed, (model, define_graph) in experiments:
    schedule = globals()[args.schedule](model)
    config = globals()[args.config](model)
    logdir = os.path.join(args.logdir, '{}-{}'.format(model, seed))
    tf.gfile.MakeDirs(logdir)
    if os.path.exists(os.path.join(logdir, 'metrics.npz')):
      if args.resume:
        continue
      raise RuntimeError('The log directory is not empty.')
    with open(os.path.join(logdir, 'schedule.yaml'), 'w') as file_:
      yaml.dump(schedule.copy(), file_)
    with open(os.path.join(logdir, 'config.yaml'), 'w') as file_:
      yaml.dump(config.copy(), file_)
    message = '\n{0}\n# Model {1} seed {2}\n{0}'
    print(message.format('#' * 79, model, seed))
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    graph = define_graph(config)
    tools.run_experiment(logdir, graph, dataset, **schedule, seed=seed)
    plot_results(args)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--schedule', default='default_schedule')
  parser.add_argument('--config', default='default_config')
  parser.add_argument('--logdir', required=True)
  parser.add_argument('--dataset', required=True)
  parser.add_argument('--seeds', type=int, default=5)
  parser.add_argument('--train_amount', type=int)
  parser.add_argument('--test_amount', type=int)
  parser.add_argument('--resume', action='store_true', default=False)
  parser.add_argument('--replot', action='store_true', default=False)
  args = parser.parse_args()
  args.logdir = os.path.expanduser(args.logdir)
  main(args)
