:orphan:

:py:mod:`quark.torch.quantization.graph.optimization.opt_pass_manager`
======================================================================

.. py:module:: quark.torch.quantization.graph.optimization.opt_pass_manager


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   quark.torch.quantization.graph.optimization.opt_pass_manager.OptPassBase
   quark.torch.quantization.graph.optimization.opt_pass_manager.OptPassManager




.. py:class:: OptPassBase




   Base interface for implementing passes.

   It is required to implement the `call` function so that we can directly
   pass instances of the Pass directly to the PassManager and call them as a
   function.

   We can directly pass an instance of a class implementing this interface into
   the PassManager's `passes` attribute.

   .. py:method:: call(graph_module: torch.fx.graph_module.GraphModule) -> torch.fx.graph_module.GraphModule
      :abstractmethod:

      The pass that is run through the given graph module. To implement a
      pass, it is required to implement this function.

      Args:
          graph_module: The graph module we will run a pass on


   .. py:method:: requires(graph_module: torch.fx.graph_module.GraphModule) -> None

      This function will be called before the pass is run and will check that
      the given graph module contains the preconditions needed to run the
      pass. It is not required to implement this function.

      Args:
          graph_module: The graph module we will run checks on


   .. py:method:: ensures(graph_module: torch.fx.graph_module.GraphModule) -> None

      This function will be called after the pass is run and will check that
      the given graph module contains the postconditions needed to run the
      pass. It is not required to implement this function.

      Args:
          graph_module: The graph module we will run checks on



.. py:class:: OptPassManager(passes: Optional[List[Callable[[torch.fx.graph_module.GraphModule], torch.fx.graph_module.GraphModule]]] = None)


   Construct a OPTPassManager.

   Collects passes. This defines the pass schedule

   Args:
       passes (Optional[List[Callable]]): List of passes. A pass is a
           callable which modifies an object and returns a PassResu

   .. py:method:: add_pass(_pass: Callable[[torch.fx.graph_module.GraphModule], torch.fx.graph_module.GraphModule]) -> None

      Adds a pass into the current list of passes.


   .. py:method:: add_checks(check: Callable[[Any], Any]) -> None

      Adds a function which takes runs various checks on a given graph module.
      This function is run before and after each pass if the
      `run_checks_after_each_pass` flag is enabled.



