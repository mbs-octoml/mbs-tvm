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

import tvm
import logging

logging.basicConfig(level=logging.INFO)

capture_index_in_spans = tvm._ffi.get_global_func("relay.collage.capture_index_in_spans")
partition_on_indexes_for_testing = tvm._ffi.get_global_func("relay.collage.partition_on_indexes_for_testing")


def print_with_indexes(mod):
    mod = capture_index_in_spans()(mod)
    print(mod)


def process(mod, max_outputs, allow_taps, indexes, labels=None):
    mod = tvm.relay.transform.InferType()(mod)
    mod = capture_index_in_spans()(mod)
    mod = partition_on_indexes_for_testing(max_outputs, allow_taps, indexes, labels)(mod)
    return mod


def assert_eq(in_mod, expected_mod, actual_mod):
    if not tvm.ir.structural_equal(actual_mod, expected_mod, True):
        # Print everything in full so we can see what's going on when things fail.
        print("Input module:")
        print(in_mod)
        print("Expected module:")
        print(expected_mod)
        print("Actual module:")
        print(actual_mod)
        # Assert again so as to see the actual disagreeing sub-expressions.
        tvm.ir.assert_structural_equal(actual_mod, expected_mod, True)


def test_single_op():
    def input():
        return tvm.parser.fromtext(
            """
            #[version = "0.0.5"]
            def @main(%a: Tensor[(5, 7), float32], %b: Tensor[(5, 7), float32],
                      %c: Tensor[(5, 7), float32], %d: Tensor[(5, 7), float32]) {
              %0 = add(%a, %b);
              %1 = add(%c, %d);   // node 7
              subtract(%0, %1)
            }
        """)

    def expected():
        return tvm.parser.fromtext(
            """
            #[version = "0.0.5"]
            def @main(%a: Tensor[(5, 7), float32], %b: Tensor[(5, 7), float32],
                      %c: Tensor[(5, 7), float32], %d: Tensor[(5, 7), float32]) {
              %0 = add(%a, %b);
              %1 = (fn(%x, %y) { add(%x, %y) })(%c, %d);
              subtract(%0, %1)
            }
        """)

    assert_eq(input(), expected(), process(input(), 1, False, [7]))


def test_multi_output():
    def input():
        return tvm.parser.fromtext(
            """
            #[version = "0.0.5"]
            def @main(%a: Tensor[(5, 7), float32], %b: Tensor[(5, 7), float32],
                      %c: Tensor[(5, 7), float32], %d: Tensor[(5, 7), float32]) {
              %0 = add(%a, %b);   // node 6
              %1 = add(%c, %d);   // node 7
              subtract(%0, %1)
            }
        """)

    def expected():
        return tvm.parser.fromtext(
            """
            #[version = "0.0.5"]
            def @main(%a: Tensor[(5, 7), float32], %b: Tensor[(5, 7), float32],
                      %c: Tensor[(5, 7), float32], %d: Tensor[(5, 7), float32]) {
              %0 = (fn(%w, %x, %y, %z) { (add(%y, %z), add(%w, %x)) })(%c, %d, %a, %b);
              %1 = %0.0;
              %2 = %0.1;
              subtract(%1, %2)
            }
        """)

    # No rewrite since 2 outputs
    assert_eq(input(), input(), process(input(), 1, False, [6, 7]))
    # Rewrite
    assert_eq(input(), expected(), process(input(), 2, False, [6, 7]))


def test_classic_conv2d_add_relu():
    def input():
        return tvm.parser.fromtext(
            """
            #[version = "0.0.5"]
            def @main(%a: Tensor[(5, 3, 32, 32), float32], %b: Tensor[(2, 3, 5, 5), float32],
                      %c: Tensor[(5, 2, 28, 28), float32], %d: Tensor[(5, 2, 28, 28), float32]) {
              %0 = nn.conv2d(%a, %b); // node 8
              %1 = add(%0, %c);       // node 9
              %2 = nn.relu(%1);       // node 10
              subtract(%2, %d)
            }
        """)

    def expected():
        return tvm.parser.fromtext(
            """
            #[version = "0.0.5"]
            def @main(%a: Tensor[(5, 3, 32, 32), float32], %b: Tensor[(2, 3, 5, 5), float32],
                      %c: Tensor[(5, 2, 28, 28), float32], %d: Tensor[(5, 2, 28, 28), float32]) {
              %2 = (fn(%x, %y, %z) {
                %0 = nn.conv2d(%x, %y);
                %1 = add(%0, %z);
                nn.relu(%1)
              })(%a, %b, %c);           
              subtract(%2, %d)
            }
        """)

    assert_eq(input(), expected(), process(input(), 1, False, [8, 9, 10]))


def test_diamond_single_output():
    def input():
        return tvm.parser.fromtext(
            """
            #[version = "0.0.5"]
            def @main(%a: Tensor[(5, 3, 32, 32), float32], %b: Tensor[(2, 3, 5, 5), float32]) {
              %0 = nn.conv2d(%a, %b, padding=[0, 0, 0, 0]); // node 5
              %1 = nn.relu(%0);                             // node 6
              %2 = nn.relu(%1);                             // node 7
              %3 = nn.leaky_relu(%0, alpha=0f);             // node 9
              add(%2, %3)                                   // node 10
            }   
        """)

    def expected():
        return tvm.parser.fromtext(
            """
            #[version = "0.0.5"]
            def @main(%a: Tensor[(5, 3, 32, 32), float32], %b: Tensor[(2, 3, 5, 5), float32]) {
              (fn (%x: Tensor[(5, 3, 32, 32), float32], %y: Tensor[(2, 3, 5, 5), float32]) {
                %0 = nn.conv2d(%x, %y, padding=[0, 0, 0, 0]);
                %1 = nn.relu(%0);
                %2 = nn.relu(%1);
                %3 = nn.leaky_relu(%0, alpha=0f);
                add(%2, %3)
              })(%a, %b)
            }
        """)

    assert_eq(input(), expected(), process(input(), 1, False, [5, 6, 7, 9, 10]))


def test_diamond_multi_output():
    def input():
        return tvm.parser.fromtext(
            """
            #[version = "0.0.5"]
            def @main(%a: Tensor[(5, 3, 32, 32), float32], %b: Tensor[(2, 3, 5, 5), float32]) {
              %0 = nn.conv2d(%a, %b, padding=[0, 0, 0, 0]); // node 5
              %1 = nn.relu(%0);                             // node 6
              %2 = nn.relu(%1);                             // node 7
              %3 = nn.leaky_relu(%0, alpha=0f);             // node 9
              add(%2, %3)
            }   
        """)

    def expected():
        return tvm.parser.fromtext(
            """
            #[version = "0.0.5"]
            def @main(%a: Tensor[(5, 3, 32, 32), float32], %b: Tensor[(2, 3, 5, 5), float32]) {
              %4 = (fn (%x: Tensor[(5, 3, 32, 32), float32], %y: Tensor[(2, 3, 5, 5), float32]) {
                %0 = nn.conv2d(%x, %y, padding=[0, 0, 0, 0]);
                %1 = nn.relu(%0);
                %2 = nn.relu(%1);
                %3 = nn.leaky_relu(%0, alpha=0f);
                (%2, %3)
              })(%a, %b);
              %5 = %4.0;
              %6 = %4.1;
              add(%5, %6)
            }
        """)

    assert_eq(input(), expected(), process(input(), 2, False, [5, 6, 7, 9]))


def test_with_tap():
    def input():
        return tvm.parser.fromtext(
            """
            #[version = "0.0.5"]
            def @main(%a: Tensor[(5, 3, 32, 32), float32], %b: Tensor[(2, 3, 5, 5), float32]) {
              %0 = nn.conv2d(%a, %b, padding=[0, 0, 0, 0]); // node 5
              %1 = nn.relu(%0);                             // node 6
              add(%1, %0)
            }            
        """)

    def expected():
        return tvm.parser.fromtext(
            """
            #[version = "0.0.5"]
            def @main(%a: Tensor[(5, 3, 32, 32), float32], %b: Tensor[(2, 3, 5, 5), float32]) {
              %2 = (fn (%x, %y) {
                %0 = nn.conv2d(%x, %y, padding=[0, 0, 0, 0]);
                %1 = nn.relu(%0);
                (%0, %1)
              })(%a, %b);
              %3 = %2.1;
              %4 = %2.0; 
              add(%3, %4)
            }            
        """)

    # No rewrite since has tap
    assert_eq(input(), input(), process(input(), 2, False, [5, 6]))
    # Rewrite
    assert_eq(input(), expected(), process(input(), 2, True, [5, 6]))


def test_no_cycles():
    def input():
        return tvm.parser.fromtext(
            """
            #[version = "0.0.5"]
            def @main(%a: Tensor[(5, 7), float32], %b: Tensor[(5, 7), float32]) {
              %0 = add(%a, %b); // node 3 
              %1 = add(%0, %b);
              add(%1, %b)       // node 5
            }            
        """)

    def expected():
        return tvm.parser.fromtext(
            """
            #[version = "0.0.5"]
            def @main(%a: Tensor[(5, 7), float32], %b: Tensor[(5, 7), float32]) {
              (fn(%x, %y) {
                %0 = add(%x, %y);
                %1 = add(%0, %y);
                add(%1, %y)
              })(%a, %b) 
            }            
        """)

    # No rewrite since would create cycle
    assert_eq(input(), input(), process(input(), 2, False, [3, 5]))
    # No cycle
    assert_eq(input(), expected(), process(input(), 2, False, [3, 4, 5]))


def test_labels_direct_connection():
    def input():
        return tvm.parser.fromtext(
            """
            #[version = "0.0.5"]
            def @main(%a: Tensor[(5, 7), float32]) {
              %0 = nn.relu(%a);  // node 3
              %1 = nn.relu(%0);  // node 4
              %2 = nn.relu(%1);  // node 5
              %3 = nn.relu(%1);  // node 6
              %4 = add(%2, %3);  // node 7
              %5 = nn.relu(%4);  // node 8
              %6 = nn.relu(%4);  // node 9
              %7 = add(%5, %6);  // node 10
              nn.relu(%7)        // node 11  
            }            
        """)

    def expected():
        return tvm.parser.fromtext(
            """
            #[version = "0.0.5"]
            def @main(%a: Tensor[(5, 7), float32]) {
              (fn(%x) {
                %0 = nn.relu(%x);
                %4 = (fn(%y, Composite="a") { 
                  %1 = nn.relu(%y);
                  %2 = nn.relu(%1);
                  %3 = nn.relu(%1);
                  add(%2, %3)
                })(%0);
                %7 = (fn(%z, Composite="b") {
                  %5 = nn.relu(%z);
                  %6 = nn.relu(%z);
                  add(%5, %6)
                })(%4);
                nn.relu(%7)
              })(%a)  
            }
        """)

    assert_eq(input(), expected(), process(input(), 1, False,
                                           [3, 4, 5, 6, 7, 8, 9, 10, 11],
                                           ["", "a", "a", "a", "a", "b", "b", "b", ""]))


def test_labels_nested_tap():
    def input():
        return tvm.parser.fromtext(
            """
            #[version = "0.0.5"]
            def @main(%a: Tensor[(5, 7), float32]) {
              %0 = nn.relu(%a);  // node 3
              %1 = nn.relu(%0);  // node 4
              %2 = nn.relu(%1);  // node 5
              %3 = nn.relu(%1);  // node 6
              %4 = add(%2, %3);  // node 7
              %5 = nn.relu(%4);  // node 8
              %6 = nn.relu(%4);  // node 9
              %7 = add(%5, %6);  // node 10
              add(%2, %7)        // node 11  
            }            
        """)

    def expected():
        return tvm.parser.fromtext(
            """
            #[version = "0.0.5"]
            def @main(%a: Tensor[(5, 7), float32]) {
              %0 = nn.relu(%a); 
              %9 = (fn(%x) {
                %5 = (fn(%y, Composite="a") { 
                  %1 = nn.relu(%y);
                  %2 = nn.relu(%1);
                  %3 = nn.relu(%1);
                  %4 = add(%2, %3);
                  (%2, %4)
                })(%x);
                %8 = (fn(%z, Composite="b") {
                  %6 = nn.relu(%z);
                  %7 = nn.relu(%z);
                  add(%6, %7)
                })(%5.1);
                (%5.0, %8)
              })(%0);
              add(%9.0, %9.1)
            }
        """)

    assert_eq(input(), expected(), process(input(), 2, True,
                                           [4, 5, 6, 7, 8, 9, 10],
                                           ["a", "a", "a", "a", "b", "b", "b"]))


if __name__ == "__main__":
    import sys
    import pytest

    sys.exit(pytest.main([__file__] + sys.argv[1:]))
