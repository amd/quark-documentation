:py:mod:`quark.onnx.optimizations.convert_transforms`
=====================================================

.. py:module:: quark.onnx.optimizations.convert_transforms

.. autoapi-nested-parse::

   Graph transforms for the conversion of onnx models.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   quark.onnx.optimizations.convert_transforms.ConvQDQToQOPTransform
   quark.onnx.optimizations.convert_transforms.MatMulQDQToQOPTransform
   quark.onnx.optimizations.convert_transforms.AddQDQToQOPTransform
   quark.onnx.optimizations.convert_transforms.MulQDQToQOPTransform
   quark.onnx.optimizations.convert_transforms.SigmoidQDQToQOPTransform
   quark.onnx.optimizations.convert_transforms.RemoveQDQTransform




.. py:class:: ConvQDQToQOPTransform




   Defines a transform to be applied to a onnx model graph.

   A transform is a combination of 'Find + Replace' which describes how to find
   a pattern of nodes in a model, and what to replace those nodes with.

   A pattern is described using `OpTypePattern`. The replacement function receives
   a `NodeTree` which contains the matched nodes and should return a
   `NodeTree` which contains the set of nodes which replaced the matched
   nodes.

   .. py:method:: pattern() -> OpTypePattern

      Return the `OpTypePattern` to find in the model graph.


   .. py:method:: replacement(match_node: NodeTree) -> Any

      Generate a replacement sub-graph for the matched sub-graph.

      The fundamental constraint of the replacement is that the replacement
      sub-graph should consume the same input tensors as the original sub-graph
      and also produce a final list of tensors which are same in number and shape
      as the original sub-graph. Not following this could crash model creation,
      or introduce bugs in the new model graph.

      Args:
        match_nodes: Matched NodeTree based on `self.pattern()`.



.. py:class:: MatMulQDQToQOPTransform




   Defines a transform to be applied to a onnx model graph.

   A transform is a combination of 'Find + Replace' which describes how to find
   a pattern of nodes in a model, and what to replace those nodes with.

   A pattern is described using `OpTypePattern`. The replacement function receives
   a `NodeTree` which contains the matched nodes and should return a
   `NodeTree` which contains the set of nodes which replaced the matched
   nodes.

   .. py:method:: pattern() -> OpTypePattern

      Return the `OpTypePattern` to find in the model graph.


   .. py:method:: replacement(match_node: NodeTree) -> Any

      Generate a replacement sub-graph for the matched sub-graph.

      The fundamental constraint of the replacement is that the replacement
      sub-graph should consume the same input tensors as the original sub-graph
      and also produce a final list of tensors which are same in number and shape
      as the original sub-graph. Not following this could crash model creation,
      or introduce bugs in the new model graph.

      Args:
        match_nodes: Matched NodeTree based on `self.pattern()`.



.. py:class:: AddQDQToQOPTransform




   Defines a transform to be applied to a onnx model graph.

   A transform is a combination of 'Find + Replace' which describes how to find
   a pattern of nodes in a model, and what to replace those nodes with.

   A pattern is described using `OpTypePattern`. The replacement function receives
   a `NodeTree` which contains the matched nodes and should return a
   `NodeTree` which contains the set of nodes which replaced the matched
   nodes.

   .. py:method:: pattern() -> OpTypePattern

      Return the `OpTypePattern` to find in the model graph.


   .. py:method:: replacement(match_node: NodeTree) -> Any

      Generate a replacement sub-graph for the matched sub-graph.

      The fundamental constraint of the replacement is that the replacement
      sub-graph should consume the same input tensors as the original sub-graph
      and also produce a final list of tensors which are same in number and shape
      as the original sub-graph. Not following this could crash model creation,
      or introduce bugs in the new model graph.

      Args:
        match_nodes: Matched NodeTree based on `self.pattern()`.



.. py:class:: MulQDQToQOPTransform




   Defines a transform to be applied to a onnx model graph.

   A transform is a combination of 'Find + Replace' which describes how to find
   a pattern of nodes in a model, and what to replace those nodes with.

   A pattern is described using `OpTypePattern`. The replacement function receives
   a `NodeTree` which contains the matched nodes and should return a
   `NodeTree` which contains the set of nodes which replaced the matched
   nodes.

   .. py:method:: pattern() -> OpTypePattern

      Return the `OpTypePattern` to find in the model graph.


   .. py:method:: replacement(match_node: NodeTree) -> Any

      Generate a replacement sub-graph for the matched sub-graph.

      The fundamental constraint of the replacement is that the replacement
      sub-graph should consume the same input tensors as the original sub-graph
      and also produce a final list of tensors which are same in number and shape
      as the original sub-graph. Not following this could crash model creation,
      or introduce bugs in the new model graph.

      Args:
        match_nodes: Matched NodeTree based on `self.pattern()`.



.. py:class:: SigmoidQDQToQOPTransform




   Defines a transform to be applied to a onnx model graph.

   A transform is a combination of 'Find + Replace' which describes how to find
   a pattern of nodes in a model, and what to replace those nodes with.

   A pattern is described using `OpTypePattern`. The replacement function receives
   a `NodeTree` which contains the matched nodes and should return a
   `NodeTree` which contains the set of nodes which replaced the matched
   nodes.

   .. py:method:: pattern() -> OpTypePattern

      Return the `OpTypePattern` to find in the model graph.


   .. py:method:: replacement(match_node: NodeTree) -> Any

      Generate a replacement sub-graph for the matched sub-graph.

      The fundamental constraint of the replacement is that the replacement
      sub-graph should consume the same input tensors as the original sub-graph
      and also produce a final list of tensors which are same in number and shape
      as the original sub-graph. Not following this could crash model creation,
      or introduce bugs in the new model graph.

      Args:
        match_nodes: Matched NodeTree based on `self.pattern()`.



.. py:class:: RemoveQDQTransform




   Defines a transform to be applied to a onnx model graph.

   A transform is a combination of 'Find + Replace' which describes how to find
   a pattern of nodes in a model, and what to replace those nodes with.

   A pattern is described using `OpTypePattern`. The replacement function receives
   a `NodeTree` which contains the matched nodes and should return a
   `NodeTree` which contains the set of nodes which replaced the matched
   nodes.

   .. py:method:: pattern() -> OpTypePattern

      Return the `OpTypePattern` to find in the model graph.


   .. py:method:: replacement(match_node: NodeTree) -> Any

      Generate a replacement sub-graph for the matched sub-graph.

      The fundamental constraint of the replacement is that the replacement
      sub-graph should consume the same input tensors as the original sub-graph
      and also produce a final list of tensors which are same in number and shape
      as the original sub-graph. Not following this could crash model creation,
      or introduce bugs in the new model graph.

      Args:
        match_nodes: Matched NodeTree based on `self.pattern()`.



