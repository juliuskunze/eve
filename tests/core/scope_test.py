# Copyright 2021 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from flax import errors
from flax.core import Scope, scope, freeze, init, apply, nn

from jax import random

import numpy as np


from absl.testing import absltest

class ScopeTest(absltest.TestCase):

  def test_rng(self):
    def f(scope):
      self.assertTrue(scope.has_rng('params'))
      self.assertFalse(scope.has_rng('dropout'))
      rng = scope.make_rng('params')
      self.assertTrue(np.all(rng == random.fold_in(random.PRNGKey(0), 1)))
    init(f)(random.PRNGKey(0))

  def test_in_filter(self):
    filter_true = lambda x, y : self.assertTrue(scope.in_filter(x, y))
    filter_false = lambda x, y : self.assertFalse(scope.in_filter(x, y))

    filter_true(True, 'any_string1')
    filter_false(False, 'any_string2')
    filter_true('exact_match', 'exact_match')
    filter_false('no_match1', 'no_match2')
    filter_true(['one', 'two'], 'one')
    filter_false(['one', 'two'], 'three')
    filter_false([], 'one')
    filter_false([], None)

  def test_union_filter(self):
    def union_check(a, b, ans):
      self.assertEqual(scope.union_filters(a, b), ans)
      self.assertEqual(scope.union_filters(b, a), ans)

    union_check(['a', 'b'], ['b', 'c'], set(['a', 'b', 'c']))
    union_check(True, False, True)
    union_check(False, False, set())
    union_check(True, True, True)
    union_check(scope.DenyList(['a', 'b']), scope.DenyList(['b', 'c']), scope.DenyList(set(['b'])))
    union_check(scope.DenyList(['a', 'b']), ['b', 'c'], scope.DenyList(set(['a'])))
  
  def test_intersect_filter(self):
    def intersect_check(a, b, ans):
      self.assertEqual(scope.intersect_filters(a, b), ans)
      self.assertEqual(scope.intersect_filters(b, a), ans)

    intersect_check(['a', 'b'], ['b', 'c'], set(['b']))
    intersect_check(True, False, False)
    intersect_check(False, False, set())
    intersect_check(True, True, True)
    intersect_check(scope.DenyList(['a', 'b']), scope.DenyList(['b', 'c']), scope.DenyList(set(['a', 'b', 'c'])))
    intersect_check(scope.DenyList(['a', 'b']), ['b', 'c'], set(['c']))
  
  def test_subtract_filter(self):
    def subtract_check(a, b, ans):
      self.assertEqual(scope.subtract_filters(a, b), ans)

    subtract_check(['a', 'b'], ['b', 'c'], set(['a']))
    subtract_check(True, False, scope.DenyList(False))
    subtract_check(False, False, set())
    subtract_check(True, True, False)
    subtract_check(True, 'a', scope.DenyList('a'))
    subtract_check(scope.DenyList(['a', 'b']), scope.DenyList(['b', 'c']), set(['c']))
    subtract_check(scope.DenyList(['a', 'b']), ['b', 'c'], scope.DenyList(set(['a', 'b', 'c'])))


  def test_group_collections(self):
    params = { 'dense1': { 'x': [10, 20] } }
    batch_stats = { 'dense1': { 'ema': 5 } }
    xs = { 'params': params, 'batch_stats': batch_stats }

    # Retrieve all keys only once.
    group = scope.group_collections(xs, ['params', 'params'])
    self.assertEqual(group, ({'params': params}, {}))

    # Ignore non-existing keys.
    self.assertEqual(scope.group_collections(xs, ['vars']), ({},))

    # False gets nothing and True retrieves all keys once.
    self.assertEqual(scope.group_collections(xs, [False, True, True]), 
                                             ({}, xs, {}))

  def test_inconsistent_param_shapes(self):
    def f(scope):
      scope.param('test', nn.initializers.ones, (4,))
    
    msg = r'Inconsistent shapes between value and initializer for parameter "test" in "/": \(2,\), \(4,\).'
    with self.assertRaisesRegex(errors.ScopeParamShapeError, msg):
      apply(f)(freeze({'params': {'test': np.ones((2,))}}))

  def test_mutate_undefined_collection(self):
    def f(scope):
      scope.put_variable('state', 'test', 123)

    msg = r'Cannot update variable "test" in "/" because collection "state" is immutable.'
    with self.assertRaisesRegex(errors.ModifyScopeVariableError, msg):
      init(f, mutable='params')(random.PRNGKey(0))

  def test_undefined_param(self):
    def f(scope):
      nn.dense(scope.push('dense'), np.ones((1, 2)), 2)

    msg = r'No parameter named "kernel" exists in "/dense".'
    with self.assertRaisesRegex(errors.ScopeParamNotFoundError, msg):
      apply(f)({})

  def test_variable_is_mutable(self):
    def f(scope, should_be_mutable):
      test = scope.variable('state', 'test', lambda: 1)
      self.assertEqual(test.is_mutable(), should_be_mutable)

    _, variables = apply(f, mutable='state')({}, True)
    apply(f, mutable=False)(variables, False)


if __name__ == '__main__':
  absltest.main()
