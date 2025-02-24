:py:mod:`quark.onnx.graph_transformations.model_transformer_test`
=================================================================

.. py:module:: quark.onnx.graph_transformations.model_transformer_test

.. autoapi-nested-parse::

   Tests for Model Transformation.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   quark.onnx.graph_transformations.model_transformer_test.ModelTransformerTest



Functions
~~~~~~~~~

.. autoapisummary::

   quark.onnx.graph_transformations.model_transformer_test.generate_input_initializer



.. py:function:: generate_input_initializer(tensor_shape: List[int], tensor_dtype: type, input_name: str) -> onnx.TensorProto

   Helper function to generate initializers for test inputs


.. py:class:: ModelTransformerTest(methodName='runTest')




   A class whose instances are single test cases.

   By default, the test code itself should be placed in a method named
   'runTest'.

   If the fixture may be used for many test cases, create as
   many test methods as are needed. When instantiating such a TestCase
   subclass, specify in the constructor arguments the name of the test method
   that the instance is to execute.

   Test authors should subclass TestCase for their own tests. Construction
   and deconstruction of the test's environment ('fixture') can be
   implemented by overriding the 'setUp' and 'tearDown' methods respectively.

   If it is necessary to override the __init__ method, the base class
   __init__ method must always be called. It is important that subclasses
   should not change the signature of their __init__ method, since instances
   of the classes are instantiated automatically by parts of the framework
   in order to be run.

   When subclassing TestCase, you can set these attributes:
   * failureException: determines which exception will be raised when
       the instance's assertion methods fail; test methods raising this
       exception will be deemed to have 'failed' rather than 'errored'.
   * longMessage: determines whether long messages (including repr of
       objects used in assert methods) will be printed on failure in *addition*
       to any explicit message passed.
   * maxDiff: sets the maximum length of a diff in failure messages
       by assert methods using difflib. It is looked up as an instance
       attribute so can be configured by individual tests if required.

   .. py:class:: ReplaceWholeModel




      Defines a transform to be applied to a onnx model graph.

      A transform is a combination of 'Find + Replace' which describes how to find
      a pattern of nodes in a model, and what to replace those nodes with.

      A pattern is described using `OpTypePattern`. The replacement function receives
      a `NodeTree` which contains the matched nodes and should return a
      `NodeTree` which contains the set of nodes which replaced the matched
      nodes.

      .. py:method:: pattern() -> OpTypePattern

         Return the `OpTypePattern` to find in the model graph.


      .. py:method:: replacement(match_node: NodeTree) -> NodeTree

         Generate a replacement sub-graph for the matched sub-graph.

         The fundamental constraint of the replacement is that the replacement
         sub-graph should consume the same input tensors as the original sub-graph
         and also produce a final list of tensors which are same in number and shape
         as the original sub-graph. Not following this could crash model creation,
         or introduce bugs in the new model graph.

         Args:
           match_nodes: Matched NodeTree based on `self.pattern()`.



   .. py:class:: RemoveRelu




      Defines a transform to be applied to a onnx model graph.

      A transform is a combination of 'Find + Replace' which describes how to find
      a pattern of nodes in a model, and what to replace those nodes with.

      A pattern is described using `OpTypePattern`. The replacement function receives
      a `NodeTree` which contains the matched nodes and should return a
      `NodeTree` which contains the set of nodes which replaced the matched
      nodes.

      .. py:method:: pattern() -> OpTypePattern

         Return the `OpTypePattern` to find in the model graph.


      .. py:method:: replacement(match_node: NodeTree) -> NodeTree

         Generate a replacement sub-graph for the matched sub-graph.

         The fundamental constraint of the replacement is that the replacement
         sub-graph should consume the same input tensors as the original sub-graph
         and also produce a final list of tensors which are same in number and shape
         as the original sub-graph. Not following this could crash model creation,
         or introduce bugs in the new model graph.

         Args:
           match_nodes: Matched NodeTree based on `self.pattern()`.




