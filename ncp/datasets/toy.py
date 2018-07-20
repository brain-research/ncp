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

import numpy as np

from ncp import tools


def generate_vargrad_dataset(length=1000, noise_slope=0.2):
  random = np.random.RandomState(0)
  inputs = np.linspace(-1, 1, length)
  noise_std = np.maximum(0, (inputs + 1) * noise_slope)
  targets = 0.5 * + np.sin(25 * inputs) + random.normal(0, noise_std)
  targets += 0.5 * inputs
  domain = np.linspace(-1.2, 1.2, 1000)
  train_split = np.repeat([False, True, False, True, False], 200)
  test_split = (1 - train_split).astype(bool)
  domain, inputs, targets = domain[:, None], inputs[:, None], targets[:, None]
  test_inputs, test_targets = inputs[test_split], targets[test_split]
  train_inputs, train_targets = inputs[train_split], targets[train_split]
  train = tools.AttrDict(inputs=train_inputs, targets=train_targets)
  test = tools.AttrDict(inputs=test_inputs, targets=test_targets)
  return tools.AttrDict(domain=domain, train=train, test=test, target_scale=1)
