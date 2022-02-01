from tvm import relay
from collections import defaultdict

from .default_patterns import str_to_pattern
from .pattern_language import Pattern, name_relay_pattern

#from tvm.relay.dataflow_pattern import *
from collage.utils import (
        no_constraints_func,
        get_op_pattern,
        get_op_pattern, 
        get_args,
        is_var_node, 
        is_constant_node, 
        is_tuple_node, 
        is_tuplegetitem_node,
        no_constraints_func
      )

from collage.backend import BackendKind
from collage.measurer.op_cost_logger import Config
import logging

# Backend pattern is a basic unit that pattern registry uses to manage (backend, pattern) pair
class BackendPattern(object):
  def __init__(self, backend, pattern, op_cost_logger, constraint_func = no_constraints_func,):
    self._backend = backend
    self._pattern = pattern 
    self._op_cost_logger = op_cost_logger 
    self._constraint_func = constraint_func

    self._op_name = pattern.get_name()
    self._name = backend.name + "_" + self._op_name
    self._depth = pattern.get_depth()
    
  def __hash__(self):
      return hash((self._name, self._constraint_func))

  def __eq__(self, other):
      return self._name == other._name and self._constraint_func == other._constraint_func

  def __repr__(self):
    return self._name

  def get_pattern(self):
    return self._pattern

  def get_backend(self):
    return self._backend

  # max depth is the depth of the longest branch
  # (chain pattern has a single branch, diamond pattern has 2 branchese)
  def get_depth(self):
    return self._depth

  def get_cost(self, expr, build_target):
    # configuration: backend operator name, operator type (which encode pattern), data shape, node attributes
    config = Config(self._name, self._op_name, expr)
    # print(config)

    # For Tuple, we do not need to measure it
    if is_tuple_node(expr) or is_tuplegetitem_node(expr):
      return 0

    # if constraints are not satisfied, return infinite cost
    if not self._constraint_func(config):
      return float('inf')#, 0

    cost_info = self._op_cost_logger.get_cost(config)

    if cost_info != None:
      logging.info("REUSED RESULT!!!!")
    else:
      logging.info("NOT REUSED!!!!")

      cost_info = self._backend.measure_cost(expr, build_target, name = self._name)
      self._op_cost_logger.save_cost(config, cost_info)

    # We use mean as a cost instead of sampling for now
    mean_cost, std_cost = cost_info
    return mean_cost


# library class (singleton) representing all backend patterns
class PatternRegistry(object):
  __instance = None

  @staticmethod
  def destroy():
      """ Static access method. """
      PatternRegistry.__instance = None

  def __init__(self, backend_registry, op_cost_logger):
    """ Virtually private constructor. """
    if PatternRegistry.__instance != None:
      raise Exception("This class should be a singleton!")


    self.backend_registry = backend_registry
    self.op_cost_logger = op_cost_logger
    self.update_backend_registry_info()
   
    # Manage its instance
    PatternRegistry.__instance = self

  def update_backend_registry_info(self):
    # Graph-level backends will be fine-tuned with graph-level optimizer
    self.graph_level_backends = set()
    # Pattern generate will automatically add legal patterns right before optimizers
    self.backends_with_pattern_generators = set()
    # Unique set of backend patterns (backend+pattern)
    self.all_backend_patterns = set()
    # Manages a pattern <-> bakcend patterns relations
    self.pattern_to_backend_patterns = defaultdict(set)

    for name, backend_obj in self.backend_registry.items():
        # Add enumerated patterns to pattern registry
        for pattern, constraint_func in backend_obj.patterns:
          self.add_backend_pattern(backend_obj, pattern, constraint_func)

        # Add backends with pattern generator
        if backend_obj.pattern_generator is not None:
          self.backends_with_pattern_generators.add(backend_obj)
        
        # Add graph-level backends
        if backend_obj.kind == BackendKind.GRAPH_LEVEL:
          self.graph_level_backends.add(backend_obj)


  def get_registered_backends(self):
    return list(self.backend_registry.keys())

  def add_backend_pattern(self, backend, pattern, constraint_func):
    constraint_func = no_constraints_func if constraint_func is None else constraint_func
    backend_pattern = BackendPattern(
                                      backend, 
                                      pattern,
                                      self.op_cost_logger,
                                      constraint_func,
                                  )
    self.all_backend_patterns.add(backend_pattern)
    self.pattern_to_backend_patterns[backend_pattern.get_pattern()].add(backend_pattern)
  
  def get_backend_patterns(self, pattern):
    return self.pattern_to_backend_patterns[pattern]

  # save newly measured op perfs to the log
  def save_to_log(self):
    return self.op_cost_logger.save_to_log()
