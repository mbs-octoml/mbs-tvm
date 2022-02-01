from pathlib import Path
from collage.utils import extract_attrs, get_input_shape
import json
import pickle
from os import path
import logging

# @sunggg: [TODO] Need to check hash conflict
# configuration includes operator name, operator type (backend operators from different targets might have the same type),
# data shape of all free variables, and node attributes
class Config(object):
  # We have data_shape and attrs as arguments for debugging purpose
  def __init__(self, op_name, pattern, expr, data_shape=None, attrs=None):
    self._op_name = op_name
    self._pattern = pattern

    if expr != None:
      self._data_shape = get_input_shape(expr)
      self._attrs = extract_attrs(expr)

    else:
      # Debugging purpose
      self._data_shape = data_shape
      self._attrs = attrs

  def __hash__(self):
    return hash((self._op_name, self._pattern, self._data_shape, self._attrs))

  def __eq__(self, other):
#     print(f"Check equality, {type(self._op_name)}, {type(self._pattern)}, {type(self._data_shape)}, {type(self._attrs)}")
    return (self._op_name == other._op_name and self._pattern == other._pattern
    and self._data_shape == other._data_shape and self._attrs == other._attrs)

  def __repr__(self):
    return "op_name: {0}, pattern: {1}, data_shape: {2}, attrs: {3}".format(
      self._op_name, self._pattern, self._data_shape, self._attrs)

  def __str__(self):
    return "pattern: {0}, data_shape: {1}, attrs: {2}, op_name: {3}".format(
      self._pattern, self._data_shape, self._attrs, self._op_name)


# @sunggg: Do we need this per backend?

# class to save costs of already evaluated configurations so we do not need to reevaluate them
class OpCostLogger(object):
  def __init__(self, log_path = None, dump_readable = False):
    # maps configurations already measured to the measured cost (in ms)
    self.measured_configs = dict()
    self.log_path = "operator_cost.log" if log_path is None else log_path
    self.log_path_readable = "readable_" + self.log_path + ".json"
    self.dump_readable = dump_readable

  def get_cost(self, config):
    if config in self.measured_configs:
      return self.measured_configs[config]
    return None

  # cost is (mean(cost), std(cost))
  def save_cost(self, config, cost):
    self.measured_configs[config] = cost

  def save_to_log(self):
    with open(self.log_path, 'wb+') as log:
      pickle.dump(self.measured_configs, log)

    if self.dump_readable:
      str_configs = dict()
      for key, perf in self.measured_configs.items():
        str_configs[str(key)] = perf

      with open(self.log_path_readable, 'w+') as log:
        json.dump(str_configs, log, sort_keys=True, indent=4)

  # If log doesn't exist, it uses default empty dictionary.
  def load_from_log(self):
    if path.exists(self.log_path):
      with open(self.log_path, 'rb') as log:
        logging.info(">> Start with previous op cost log")
        self.measured_configs = pickle.load(log)
    else:
       logging.info(">> Start from scratch")