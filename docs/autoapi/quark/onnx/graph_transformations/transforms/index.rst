:py:mod:`quark.onnx.graph_transformations.transforms`
=====================================================

.. py:module:: quark.onnx.graph_transformations.transforms

.. autoapi-nested-parse::

   Defines core classes for expressing onnx model transformations.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   quark.onnx.graph_transformations.transforms.OpTypePattern
   quark.onnx.graph_transformations.transforms.NodeTree
   quark.onnx.graph_transformations.transforms.Transform




.. py:class:: OpTypePattern(op_type: str = '', inputs: Optional[List[OpTypePattern]] = None, config: Optional[Dict[str, Any]] = None)




   Defines a tree sub-graph pattern of onnx nodes to match in a model.

   `OpTypePattern` can be used to describe various common patterns in model
   graphs that we need to find.

   Examples:
       Matches a Conv+BN+ReLU6 and DepthwiseConv+BN+ReLU6 pattern.
       pattern = OpTypePattern('ReLU', {'max_value': 6.0}, [
           OpTypePattern('BatchNormalization', {}, [
               OpTypePattern('Conv2D|DepthwiseConv2D', {} [])
           ])
       ])

       Matches multiple Conv2Ds feeding into a Concat.
       pattern = OpTypePattern('Concat', {}, [
           OpTypePattern('Conv2D', {}, []),
           OpTypePattern('Conv2D', {}, [])
       ])



.. py:class:: NodeTree(node: Optional[onnx.NodeProto] = None, weights: Union[OrderedDict[str, onnx.TensorProto], List[Any], None] = None, input_nodes: Optional[List[NodeTree]] = None, metadata: Optional[Dict[str, Any]] = None)




   Represents a pattern matching results in a node containing a tree.

   `NodeTree` is used to represent a tree of nodes in a model. It contains
   the NodeDef which describes the node, and other input nodes feeding into it.

   It is used as a generic class to represent both sets of nodes which have
   been found in a model, and nodes which should be replaced inside the model.


.. py:class:: Transform




   Defines a transform to be applied to a onnx model graph.

   A transform is a combination of 'Find + Replace' which describes how to find
   a pattern of nodes in a model, and what to replace those nodes with.

   A pattern is described using `OpTypePattern`. The replacement function receives
   a `NodeTree` which contains the matched nodes and should return a
   `NodeTree` which contains the set of nodes which replaced the matched
   nodes.

   .. py:property:: allow_multi_consumers
      :type: bool

      Whether to allow the internal node have multiple consuming nodes.

      E.g.
            B                B
          //                //
      A --        to   E --
          \                \
            C --> D          F

      Should set allow_mulit_consumers if you want to match pattern "A --> C --> D".
      Please be careful to handle the transformation to not break the input connection
      of consumers outside the pattern, otherwise will lead to unknown input tensors.


   .. py:method:: pattern() -> OpTypePattern
      :abstractmethod:

      Return the `OpTypePattern` to find in the model graph.


   .. py:method:: replacement(match_node: NodeTree) -> Any
      :abstractmethod:

      Generate a replacement sub-graph for the matched sub-graph.

      The fundamental constraint of the replacement is that the replacement
      sub-graph should consume the same input tensors as the original sub-graph
      and also produce a final list of tensors which are same in number and shape
      as the original sub-graph. Not following this could crash model creation,
      or introduce bugs in the new model graph.

      Args:
        match_nodes: Matched NodeTree based on `self.pattern()`.



